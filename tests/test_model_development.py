import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pipeline.main import build_parser
from pipeline.model_cli import job_script, stage_arguments, submit_model
from pipeline.model_development import ModelDevelopment, call, read_yaml, write_json, write_yaml
from pipeline.model_evaluation import frame_id, read_table, point_errors, metrics, write_report
from pipeline.model_inference import prediction_frame
from pipeline.model_inference import predict


def table(rows, parts=('nose',), confidence=False, index=None):
    coords = ['x', 'y', 'likelihood'] if confidence else ['x', 'y']
    columns = pd.MultiIndex.from_product([['human'], parts, coords], names=['scorer', 'bodyparts', 'coords'])
    return pd.DataFrame(rows, columns=columns, index=index)


class EvaluationTests(unittest.TestCase):
    def test_frame_identity_preserves_folder_and_normalizes_windows(self):
        self.assertEqual(frame_id(('labeled-data', 'a', 'img.png')), 'a/img.png')
        self.assertEqual(frame_id(r'C:\old\labeled-data\a\img.png'), 'a/img.png')
        self.assertEqual(frame_id('img.png', 'a'), 'a/img.png')
        self.assertNotEqual(frame_id('img.png', 'a'), frame_id('img.png', 'b'))
        self.assertEqual(frame_id('/data/a/img.png', 'a'), 'a/img.png')

    def test_hdf_and_csv_with_multi_index_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = table([[1, 2], [3, 4]], index=pd.MultiIndex.from_tuples([
                ('labeled-data', 'a', '1.png'), ('labeled-data', 'a', '2.png')]))
            hdf, csv = root / 'labels.h5', root / 'labels.csv'
            frame.to_hdf(hdf, key='df')
            frame.to_csv(csv)
            pd.testing.assert_frame_equal(read_table(hdf), read_table(csv))
            self.assertEqual(list(read_table(csv).index), ['a/1.png', 'a/2.png'])

    def test_multiple_animals_and_duplicate_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.h5'
            frame = table([[1, 2], [3, 4]], index=['a/1.png', 'a/1.png'])
            frame.to_hdf(path, key='df')
            with self.assertRaisesRegex(ValueError, 'Duplicate frame'):
                read_table(path)
            frame = pd.DataFrame([[1, 2, 3, 4]], columns=pd.MultiIndex.from_product(
                [['scorer'], ['a', 'b'], ['nose'], ['x', 'y']]))
            frame.to_hdf(path, key='df', mode='w')
            with self.assertRaisesRegex(ValueError, 'Multiple scorers or animals'):
                read_table(path)

    def test_missing_prediction_is_false_negative_and_pixel_error_is_separate(self):
        labels = table([[0, 0], [0, 0], [np.nan, np.nan]], index=['a/1', 'a/2', 'a/3']).droplevel(0, axis=1)
        pred = table([[3, 4, .9], [0, 0, .9]], confidence=True, index=['a/1', 'a/3']).droplevel(0, axis=1)
        points, coverage = point_errors(labels, pred, ['nose'])
        result = metrics(points, .5)
        self.assertEqual(coverage['missing_frames'], ['a/2'])
        self.assertEqual(result['mean_pixel_error_points'], 5)
        self.assertEqual(result['visibility_precision'], .5)
        self.assertEqual(result['visibility_recall'], .5)
        self.assertEqual(result['visibility_accuracy'], 1 / 3)
        self.assertEqual(result['localization_success'], .5)

    def test_nan_coordinates_not_detection_and_undefined_denominators(self):
        labels = table([[np.nan, np.nan]], index=['a/1']).droplevel(0, axis=1)
        pred = table([[np.nan, 2, .99]], confidence=True, index=['a/1']).droplevel(0, axis=1)
        points, _ = point_errors(labels, pred, ['nose'])
        result = metrics(points, .5)
        self.assertEqual(result['detected_points'], 0)
        self.assertTrue(np.isnan(result['visibility_precision']))
        self.assertTrue(np.isnan(result['visibility_recall']))
        self.assertEqual(result['visibility_accuracy'], 1)

    def test_mapping_unsupported_and_extra_frames(self):
        labels = table([[0, 0]], parts=['ear'], index=['a/1']).droplevel(0, axis=1)
        pred = table([[0, 1, .9], [2, 3, .9]], parts=['left_ear'], confidence=True,
                     index=['a/1', 'a/2']).droplevel(0, axis=1)
        points, coverage = point_errors(labels, pred, ['ear'], {'ear': 'left_ear'})
        self.assertEqual(metrics(points, .5)['mean_pixel_error_points'], 1)
        self.assertEqual(coverage['extra_frames'], ['a/2'])
        _, coverage = point_errors(labels, pred, ['ear'])
        self.assertEqual(coverage['unsupported_bodyparts'], ['ear'])

    def test_report_artifacts_and_groups(self):
        labels = table([[0, 0]], index=['a/1']).droplevel(0, axis=1)
        pred = table([[0, 0, .9]], confidence=True, index=['a/1']).droplevel(0, axis=1)
        points, coverage = point_errors(labels, pred, ['nose'])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(points, coverage, Path(tmp), groups={'head': ['nose']})
            self.assertTrue(all(p.is_file() and p.stat().st_size for p in paths))
            summary = pd.read_csv(paths[0])
            self.assertEqual(set(summary.scope), {'overall', 'folder', 'bodypart', 'group'})


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = dict(project='project', superanimal='superanimal_topviewmouse', mode='finetune',
                             bodyparts=['nose'], training=dict(epochs=1, batch_size=2, save_epochs=1))
        self.setup = dict(hosts={'local': {'development_root': str(self.root)}},
                          model_development={'experiments': {'top': self.settings}})
        self.dev = ModelDevelopment(self.setup, 'top')
        self.dev.project.mkdir()
        write_yaml(self.dev.config, dict(Task='top', date='Jan1', project_path='old', scorer='human',
                                        bodyparts=['nose'], TrainingFraction=[.8], iteration=0,
                                        video_sets={'a.avi': {}}, multianimalproject=False))
        folder = self.dev.project / 'labeled-data' / 'a'
        folder.mkdir(parents=True)
        (folder / '1.png').write_bytes(b'image1')
        (folder / '2.png').write_bytes(b'image2')
        table([[1, 2], [3, 4]], index=pd.MultiIndex.from_tuples([
            ('labeled-data', 'a', '1.png'), ('labeled-data', 'a', '2.png')])).to_hdf(folder / 'CollectedData_human.h5', key='df')
        self.mapping = self.root / 'reviewed.csv'
        self.mapping.write_text('gt,MasterName\nnose,nose\n')

    def fake_api(self, module, name):
        if name == 'load_super_animal_config':
            return lambda **kw: {'metadata': {'bodyparts': ['nose']}}
        if name == 'read_config':
            return lambda p: read_yaml(Path(p))
        if name == 'Engine':
            return SimpleNamespace(PYTORCH='pytorch')
        if name == 'build_weight_init':
            return self.weight
        if name == 'create_conversion_table':
            return Mock()
        if name in {'create_training_dataset', 'create_training_dataset_from_existing_split'}:
            def create(config, **kw):
                self.creations.append((name, kw))
                path = self.dev.model_folder() / 'train' / 'pytorch_config.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                write_yaml(path, {'metadata': {'bodyparts': ['nose']}})
            return create
        raise AssertionError((module, name))

    def test_adoption_explicit_and_runtime_config_relocated(self):
        with self.assertRaisesRegex(ValueError, '--adopt'):
            self.dev.initialize()
        result = self.dev.initialize(adopt=True)
        self.assertEqual(result.status, 'complete')
        runtime = read_yaml(self.dev.runtime_config())
        self.assertEqual(runtime['project_path'], str(self.dev.project))
        self.assertEqual(read_yaml(self.dev.config)['project_path'], 'old')
        moved = self.dev.relocated_config({'video_sets': {r'C:\old\videos\a.avi': {}}})
        self.assertEqual(list(moved['video_sets']), [str(self.dev.project / 'videos' / 'a.avi')])

    def test_mapping_missing_duplicate_unknown(self):
        self.assertEqual(self.dev.mapping(self.mapping, ['nose']), {'nose': 'nose'})
        self.mapping.write_text('gt,MasterName\nnose,ear\n')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.dev.mapping(self.mapping, ['nose'])
        self.mapping.write_text('gt,MasterName\nnose,nose\nnose,nose\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.dev.mapping(self.mapping)

    def test_prepare_modes_share_split_and_reuse(self):
        self.weight = Mock(return_value='weights')
        self.creations = []
        with patch('pipeline.model_development.api', side_effect=self.fake_api):
            self.dev.prepare(self.mapping)
            self.assertFalse(self.weight.call_args.kwargs['memory_replay'])
            self.assertEqual(self.dev.prepare(self.mapping).status, 'reused')
            self.settings.update(shuffle=1, mode='memory_replay')
            self.dev = ModelDevelopment(self.setup, 'top')
            self.dev.prepare(self.mapping)
            self.assertTrue(self.weight.call_args.kwargs['memory_replay'])
            self.assertEqual(self.creations[-1][0], 'create_training_dataset_from_existing_split')
            self.assertEqual(self.creations[-1][1]['from_shuffle'], 0)
            self.assertEqual(self.creations[-1][1]['shuffles'], [1])

    def test_changed_mapping_rejected_same_shuffle(self):
        self.weight, self.creations = Mock(), []
        with patch('pipeline.model_development.api', side_effect=self.fake_api):
            self.dev.prepare(self.mapping)
            self.mapping.write_text('gt,MasterName\nnose,ear\n')
            with self.assertRaisesRegex(ValueError, 'new variant'):
                self.dev.prepare(self.mapping)

    def test_interrupted_training_requires_explicit_checkpoint(self):
        self.weight, self.creations = Mock(), []
        with patch('pipeline.model_development.api', side_effect=self.fake_api):
            self.dev.prepare(self.mapping)
        with patch('pipeline.model_development.api', return_value=Mock(side_effect=RuntimeError('interrupted'))):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.dev.train()
        with self.assertRaisesRegex(ValueError, '--resume-checkpoint'):
            self.dev.train()

    def test_memory_replay_channels_and_multiple_animals(self):
        config = dict(metadata={'bodyparts': ['nose']}, train_settings={'weight_init': {
            'memory_replay': True, 'conversion_array': [1]}})
        result = prediction_frame({'img': {'bodyparts': np.array([[[1, 2, .3], [4, 5, .9]]])}}, config)
        self.assertEqual(result.iloc[0].tolist(), [4, 5, .9])
        with self.assertRaisesRegex(ValueError, 'one animal'):
            prediction_frame({'img': {'bodyparts': np.zeros((2, 1, 3))}}, config)

    def test_benchmark_leakage_blocked_before_matching(self):
        benchmark = self.root / 'benchmark' / 'b'
        benchmark.mkdir(parents=True)
        (benchmark / '1.png').write_bytes(b'image1')
        table([[1, 2]], index=['1.png']).to_hdf(benchmark / 'CollectedData_human.h5', key='df')
        self.settings['benchmark'] = {'labels': 'benchmark'}
        self.dev = ModelDevelopment(self.setup, 'top')
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.dev.match()

    def test_strict_api_does_not_drop_unsupported_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, 'Incompatible DLC API'):
            call(lambda config: None, 'config', snapshot_path='selected.pt')

    def test_cli_and_slurm_propagate_selection_and_mounts(self):
        args = build_parser().parse_args(['model', 'predict', 'top', '--inputs', 'benchmark',
            '--source', 'project', '--checkpoint', 'pose.pt', '--detector-checkpoint', 'detector.pt', '--overlays'])
        host = dict(repository_dir='/repo', sandbox='/dlc', cache_dir='/cache', tmp_dir='/temp', singularity_module='singularity')
        self.setup['hosts']['hpc'] = host
        self.settings['benchmark'] = {'labels': 'benchmark'}
        self.dev = ModelDevelopment(self.setup, 'top')
        script = job_script(self.setup, self.dev, args, self.root / 'job' / 'setup.json')
        self.assertIn('--checkpoint pose.pt', script)
        self.assertIn('--overlays', script)
        self.assertIn('--containall', script)
        self.assertIn(':ro', script)
        self.assertIn(':rw', script)
        self.assertNotIn('--slurm', stage_arguments(args, 'frozen.json'))

    def test_prediction_cache_invalidates_inputs_and_checkpoints(self):
        image = self.root / 'inspect.png'
        image.write_bytes(b'first')
        config = {'metadata': {'bodyparts': ['nose']}}
        inference = Mock(return_value={str(image): {'bodyparts': np.array([[[1., 2., .9]]])}})
        model = (config, self.root / 'pose.pt', self.root / 'det.pt', {'checkpoint': 'a'})
        with patch('pipeline.model_inference.resolve_model', return_value=model), patch('pipeline.model_inference.api', return_value=inference):
            first = predict(self.dev, inputs=[image])
            self.assertEqual(predict(self.dev, inputs=[image]).status, 'reused')
            self.assertEqual(inference.call_count, 1)
            image.write_bytes(b'changed')
            second = predict(self.dev, inputs=[image])
            self.assertNotEqual(first.artifacts[0], second.artifacts[0])
            model[3]['checkpoint'] = 'b'
            third = predict(self.dev, inputs=[image])
            self.assertNotEqual(second.artifacts[0], third.artifacts[0])

    def test_external_evaluation_reuses_predictions_when_threshold_changes(self):
        folder = self.root / 'benchmark' / 'other'
        folder.mkdir(parents=True)
        image = folder / '1.png'
        image.write_bytes(b'heldout')
        table([[0, 0]], index=['1.png']).to_hdf(folder / 'CollectedData_human.h5', key='df')
        self.settings['benchmark'] = {'labels': 'benchmark', 'bodyparts': ['nose']}
        dev = ModelDevelopment(self.setup, 'top')
        model = ({'metadata': {'bodyparts': ['nose']}}, self.root / 'pose.pt', self.root / 'det.pt', {'checkpoint': 'a'})
        inference = Mock(return_value={str(image): {'bodyparts': np.array([[[3., 4., .9]]])}})
        with patch('pipeline.model_inference.resolve_model', return_value=model), patch('pipeline.model_inference.api', return_value=inference):
            first = dev.evaluate(scope='external')
            dev.settings['benchmark']['confidence'] = .7
            second = dev.evaluate(scope='external')
        self.assertEqual(inference.call_count, 1)
        self.assertNotEqual(first.artifacts[0], second.artifacts[0])
        report = json.loads(first.artifacts[2].read_text())
        self.assertTrue(report['coverage']['held_out'])
        summary = pd.read_csv(first.artifacts[0])
        row = summary[(summary.scope == 'overall') & (summary.threshold == .5)].iloc[0]
        self.assertEqual(row.mean_pixel_error_points, 5)

    def test_project_lock_prevents_simultaneous_calls(self):
        with self.dev.lock():
            with self.assertRaisesRegex(RuntimeError, 'locked'):
                with self.dev.lock():
                    pass
        self.assertFalse((self.dev.project / '.model-development.lock').exists())

    def test_missing_review_table_rejected_by_cli(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['model', 'prepare', 'top'])

    def test_split_evaluation_isolated_and_pinned(self):
        self.dev.initialize(adopt=True)
        train = self.dev.model_folder() / 'train'
        train.mkdir(parents=True)
        pose, detector = train / 'snapshot-001.pt', train / 'snapshot-detector-001.pt'
        pose.write_bytes(b'pose')
        detector.write_bytes(b'detector')
        write_yaml(train / 'pytorch_config.yaml', {'metadata': {}})
        (self.dev.project / 'training-datasets').mkdir()
        seen = []
        def evaluate(config, *, shuffles, snapshots_to_evaluate, detector_snapshot_index, per_keypoint_evaluation):
            view = Path(config).parent
            self.assertNotEqual(view, self.dev.project)
            self.assertEqual(read_yaml(Path(config))['project_path'], str(view))
            self.assertEqual(snapshots_to_evaluate, [pose.stem])
            self.assertEqual(detector_snapshot_index, 0)
            seen.append(view)
            report = view / 'evaluation-results-pytorch' / 'summary.csv'
            report.parent.mkdir()
            report.write_text('error\n1\n')
        def dispatch(module, name):
            if name == 'Task':
                return SimpleNamespace(DETECT='detect')
            if name == 'get_model_snapshots':
                return lambda **kwargs: [SimpleNamespace(path=detector)]
            if name == 'evaluate_network':
                return evaluate
            raise AssertionError(name)
        with patch('pipeline.model_development.api', side_effect=dispatch):
            first = self.dev.evaluate(scope='split', checkpoint=pose, detector_checkpoint=detector)
            second = self.dev.evaluate(scope='split', checkpoint=pose, detector_checkpoint=detector)
            self.assertEqual(len(seen), 1)
            self.assertEqual(first.artifacts, second.artifacts)
            pose.write_bytes(b'new checkpoint')
            third = self.dev.evaluate(scope='split', checkpoint=pose, detector_checkpoint=detector)
            self.assertEqual(len(seen), 2)
            self.assertNotEqual(first.artifacts, third.artifacts)
        self.assertFalse((self.dev.project / 'evaluation-results-pytorch').exists())

    def test_slurm_freezes_setup_and_reviewed_table(self):
        repo = self.root / 'repo'
        (repo / 'pipeline').mkdir(parents=True)
        (repo / 'pipeline' / 'main.py').write_text('')
        sandbox = self.root / 'sandbox'
        sandbox.mkdir()
        self.setup['hosts']['hpc'] = dict(repository_dir=str(repo), sandbox=str(sandbox),
            cache_dir=str(self.root / 'cache'), tmp_dir=str(self.root / 'tmp'), singularity_module='singularity')
        self.setup['slurm'] = {'gres': 'gpu:1'}
        args = build_parser().parse_args(['model', 'prepare', 'top', '--conversion-table', str(self.mapping)])
        with patch('pipeline.model_cli._is_windows', return_value=False), patch('pipeline.model_cli._submit', return_value='123') as submit:
            result = submit_model(self.setup, self.dev, args)
        self.assertEqual(result.job_id, '123')
        frozen = next((self.root / '.submissions').glob('*/setup.json'))
        saved = json.loads(frozen.read_text())
        self.assertEqual(saved['model_development']['experiments']['top']['project'], 'project')
        self.assertIn('_model_submission_checks', saved)
        self.assertEqual((frozen.parent / 'reviewed.csv').read_text(), self.mapping.read_text())
        self.assertIn(str(frozen), submit.call_args.args[1])

    def test_new_project_imports_labels_and_relocates_video_paths(self):
        self.settings.update(project='new-project', scorer='human', task='top', videos=['a.avi'],
                             labeled_data=['project/labeled-data/a'])
        video = self.root / 'a.avi'
        video.write_bytes(b'video')
        dev = ModelDevelopment(self.setup, 'top')
        def create(task, scorer, videos, working_directory, copy_videos):
            project = Path(working_directory) / 'dated-project'
            (project / 'videos').mkdir(parents=True)
            (project / 'videos' / 'a.avi').write_bytes(b'video')
            (project / 'labeled-data' / 'a').mkdir(parents=True)
            write_yaml(project / 'config.yaml', dict(Task=task, scorer=scorer, date='Jan1',
                       video_sets={str(project / 'videos' / 'a.avi'): {}}, multianimalproject=False))
            return project / 'config.yaml'
        with patch('pipeline.model_development.api', return_value=create):
            result = dev.initialize()
        self.assertEqual(result.status, 'complete')
        self.assertTrue((dev.project / 'labeled-data' / 'a' / '1.png').is_file())
        self.assertTrue(all(Path(p).is_file() for p in dev.project_config()['video_sets']))

    def test_extra_object_annotations_can_be_excluded_from_project(self):
        path = self.dev.project / 'labeled-data' / 'a' / 'CollectedData_human.h5'
        annotated = pd.read_hdf(path)
        annotated[('human', 'lever', 'x')] = 10
        annotated[('human', 'lever', 'y')] = 20
        annotated.to_hdf(path, key='df', mode='w')
        selected, _ = self.dev.dataset()
        self.assertEqual(set(selected.columns.get_level_values('bodyparts')), {'nose'})


if __name__ == '__main__':
    unittest.main()
