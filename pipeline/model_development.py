"""Explicit, resumable SuperAnimal development stages (DLC imported lazily)."""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .model_evaluation import file_hash, frame_id, label_tables, read_table, point_errors, write_report


def digest(value) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def write_yaml(path: Path, value: dict) -> None:
    import yaml
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')


def api(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f'DLC PyTorch API {module}.{name} is unavailable. Run in a DLC 3 PyTorch environment/container.') from exc


def call(function, *args, **kwargs):
    """Never silently discard an unsupported checkpoint or training option."""
    try:
        inspect.signature(function).bind(*args, **kwargs)
    except TypeError as exc:
        raise RuntimeError(f'Incompatible DLC API {function.__name__}: {exc}. Use a compatible DLC 3 PyTorch environment.') from exc
    return function(*args, **kwargs)


@dataclass
class ModelStageResult:
    stage: str
    status: str
    artifacts: list[Path] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class ModelDevelopment:
    def __init__(self, setup: dict, experiment: str, environment: str = 'local'):
        try:
            self.settings = copy.deepcopy(setup['model_development']['experiments'][experiment])
            self.root = Path(setup['hosts'][environment]['development_root']).expanduser().resolve()
        except KeyError as exc:
            raise ValueError(f'Missing model experiment or hosts.{environment}.development_root: {exc}') from exc
        self.name = experiment
        self.settings.setdefault('mode', 'finetune')
        self.settings.setdefault('model_name', 'hrnet_w32')
        self.settings.setdefault('detector_name', 'fasterrcnn_resnet50_fpn_v2')
        self.settings.setdefault('seed', 0)
        self.settings.setdefault('train_fraction', .8)
        self.settings.setdefault('shuffle', 0)
        if self.settings['mode'] not in {'finetune', 'memory_replay'}:
            raise ValueError('mode must be finetune or memory_replay')
        if self.settings.get('max_individuals', 1) != 1:
            raise ValueError('Model development currently supports one animal')
        if not 0 < self.settings['train_fraction'] < 1:
            raise ValueError('train_fraction must lie between 0 and 1')
        if not isinstance(self.settings['shuffle'], int) or self.settings['shuffle'] < 0:
            raise ValueError('shuffle must be a nonnegative integer')
        self.project = self.path(self.settings['project'])
        self.config = self.project / 'config.yaml'
        self.artifacts = self.project / 'model-development'
        self.variant = self.artifacts / f"shuffle-{self.settings['shuffle']}"
        self.environment = environment

    def path(self, value) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()

    @contextmanager
    def lock(self):
        self.project.mkdir(parents=True, exist_ok=True)
        lock = self.project / '.model-development.lock'
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f'Project is locked: {lock}. After a crashed worker, verify it has stopped before removing this file.') from exc
        try:
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def project_config(self) -> dict:
        if not self.config.is_file():
            raise ValueError(f'Run model init first: {self.config}')
        cfg = read_yaml(self.config)
        if cfg.get('multianimalproject'):
            raise ValueError('Multi-animal projects are unsupported')
        parts = cfg.get('bodyparts', [])
        if not parts or len(parts) != len(set(parts)) or not cfg.get('scorer'):
            raise ValueError('Project needs a scorer and unique bodyparts')
        if self.settings.get('bodyparts', parts) != parts:
            raise ValueError('Configured bodyparts differ from the existing project')
        return cfg

    def runtime_config(self) -> Path:
        cfg = self.project_config()
        cfg = self.relocated_config(cfg)
        self.variant.mkdir(parents=True, exist_ok=True)
        path = self.variant / 'runtime-config.yaml'
        write_yaml(path, cfg)
        return path

    def relocated_config(self, cfg: dict) -> dict:
        cfg = copy.deepcopy(cfg)
        cfg['project_path'] = str(self.project)
        cfg['engine'] = 'pytorch'
        cfg['video_sets'] = {
            str(self.project / 'videos' / str(path).replace('\\', '/').rsplit('/', 1)[-1]): value
            for path, value in cfg.get('video_sets', {}).items()
        }
        return cfg

    def dataset(self) -> tuple[pd.DataFrame, dict[str, str]]:
        cfg = self.project_config()
        frames, hashes = [], {}
        for path in label_tables(self.project / 'labeled-data'):
            if path.stem != 'CollectedData_' + cfg['scorer']:
                raise ValueError(f'Label scorer does not match project: {path}')
            frame = read_table(path, folder=path.parent.name)
            if set(cfg['bodyparts']) - set(frame.columns.get_level_values('bodyparts')):
                raise ValueError(f'Labels lack configured project bodyparts: {path}')
            frame = frame.loc[:, frame.columns.get_level_values('bodyparts').isin(cfg['bodyparts'])]
            for identity in frame.index:
                image = self.project / 'labeled-data' / identity
                if not image.is_file():
                    raise FileNotFoundError(f'Labeled image missing: {image}')
                hashes[identity] = file_hash(image)
            frames.append(frame)
        combined = pd.concat(frames)
        if combined.empty:
            raise ValueError('Development dataset contains no labeled frames')
        if combined.index.has_duplicates:
            raise ValueError('Duplicate labeled frame identities across folders')
        return combined, hashes

    def data_identity(self) -> dict:
        _, hashes = self.dataset()
        return dict(images=hashes, labels={p.relative_to(self.project).as_posix(): file_hash(p)
                                          for p in label_tables(self.project / 'labeled-data')})

    def record(self, stage: str, identity: dict, outputs: list[Path], status='complete') -> ModelStageResult:
        from importlib.metadata import version, PackageNotFoundError
        try:
            dlc_version = version('deeplabcut')
        except PackageNotFoundError:
            dlc_version = 'unavailable'
        path = self.variant / f'{stage}.json'
        names = {p: p.relative_to(self.project).as_posix() for p in outputs}
        write_json(path, dict(identity=identity, status=status, dlc_version=dlc_version, outputs=list(names.values()),
                             output_hashes={names[p]: file_hash(p) for p in outputs if p.is_file()}))
        return ModelStageResult(stage, status, [path, *outputs])

    def previous(self, stage: str, identity: dict, *, immutable=True) -> ModelStageResult | None:
        path = self.variant / f'{stage}.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record['identity'] != identity:
            if immutable:
                raise ValueError(f'{stage} inputs changed; configure a new variant with a distinct shuffle')
            return None
        outputs = [self.project / p for p in record['outputs']]
        if record['status'] == 'complete' and all(p.exists() for p in outputs):
            if all(file_hash(self.project / p) == h for p, h in record.get('output_hashes', {}).items()):
                return ModelStageResult(stage, 'reused', [path, *outputs])
        if immutable and record['status'] == 'complete':
            raise ValueError(f'{stage} artifacts changed or are missing; restore them or use a new variant')
        return None

    def initialize(self, *, adopt=False) -> ModelStageResult:
        self.project.mkdir(parents=True, exist_ok=True)
        if self.config.exists():
            if not adopt and not (self.artifacts / 'initialized.json').exists():
                raise ValueError('Existing project requires explicit --adopt')
        else:
            if adopt:
                raise FileNotFoundError(self.config)
            videos = [self.path(v) for v in self.settings.get('videos', [])]
            if not videos or not all(p.is_file() for p in videos):
                raise ValueError('Creating a project requires existing configured videos')
            create = api('deeplabcut', 'create_new_project')
            # DLC chooses a dated directory; use a staging parent before assigning the configured path.
            import tempfile
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                created = Path(call(create, self.settings.get('task', self.name), self.settings['scorer'],
                                    [str(v) for v in videos], working_directory=temporary, copy_videos=True))
                for item in created.parent.iterdir():
                    target = self.project / item.name
                    if target.exists():
                        raise FileExistsError(target)
                    shutil.move(str(item), str(target))
            cfg = read_yaml(self.config)
            cfg.update(bodyparts=self.settings['bodyparts'], engine='pytorch', project_path=str(self.project),
                       TrainingFraction=[self.settings['train_fraction']])
            cfg['video_sets'] = {str(self.project / 'videos' / Path(p).name): v for p, v in cfg['video_sets'].items()}
            write_yaml(self.config, cfg)
            for source in self.settings.get('labeled_data', []):
                source = self.path(source)
                destination = self.project / 'labeled-data' / source.name
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(destination)
                shutil.copytree(source, destination, dirs_exist_ok=True)
        cfg = self.project_config()
        _, images = self.dataset()
        videos = {Path(p.replace('\\', '/')).stem for p in cfg.get('video_sets', {})}
        missing = {p.split('/')[0] for p in images} - videos
        if missing:
            raise ValueError(f'Labeled folders need matching video_sets entries: {sorted(missing)}')
        path = self.artifacts / 'initialized.json'
        write_json(path, dict(project=str(self.project), bodyparts=cfg['bodyparts'], images=images))
        return ModelStageResult('init', 'complete', [self.config, path])

    def base(self) -> dict:
        return {key: self.settings[key] for key in ('superanimal', 'model_name', 'detector_name')}

    def match(self) -> ModelStageResult:
        self.check_benchmark_separation()
        identity = dict(base=self.base(), data=self.data_identity())
        previous = self.previous('match', identity)
        if previous:
            return previous
        function = api('deeplabcut.utils.pseudo_label', 'keypoint_matching')
        call(function, str(self.runtime_config()), self.settings['superanimal'],
             self.settings['model_name'], self.settings['detector_name'], copy_images=True)
        outputs = [self.project / 'memory_replay' / name for name in
                   ('conversion_table.csv', 'confusion_matrix.png', 'pseudo_predictions.json')]
        if not all(p.is_file() for p in outputs):
            raise RuntimeError('DLC matching did not produce all expected diagnostics')
        return self.record('match', identity, outputs)

    def check_benchmark_separation(self):
        benchmark = self.settings.get('benchmark', {}).get('labels')
        if not benchmark:
            return
        root = self.path(benchmark)
        if root == self.project or self.project in root.parents:
            raise ValueError('External benchmark must be outside the development project')
        if not root.exists():
            raise FileNotFoundError(root)
        _, development = self.dataset()
        hashes = set(development.values())
        for path in label_tables(root):
            table = read_table(path, folder=path.parent.relative_to(root).as_posix())
            for identity in table.index:
                image = root / identity
                if not image.is_file():
                    raise FileNotFoundError(image)
                if identity in development or file_hash(image) in hashes:
                    raise ValueError(f'External benchmark overlaps development data: {identity}; separate the datasets before matching/training')

    def mapping(self, path: Path, available: list[str] | None = None, *, require_all=True) -> dict[str, str]:
        table = pd.read_csv(path, skipinitialspace=True).rename(columns=lambda s: s.strip())
        if not {'gt', 'MasterName'} <= set(table.columns):
            raise ValueError('Reviewed conversion CSV needs gt,MasterName columns')
        table = table.dropna(subset=['gt'])
        for column in ('gt', 'MasterName'):
            table[column] = table[column].str.strip()
        if (table['MasterName'].isna().any() or table['gt'].duplicated().any()
                or table['MasterName'].duplicated().any() or (table['MasterName'] == '').any()):
            raise ValueError('Mapping contains missing or duplicate assignments; explicitly review every target')
        mapping = dict(zip(table['gt'], table['MasterName']))
        parts = self.project_config()['bodyparts']
        if set(mapping) - set(parts):
            raise ValueError('Reviewed table contains unknown project bodyparts')
        if require_all and set(mapping) != set(parts):
            raise ValueError('Reviewed table must map every project bodypart exactly once')
        if available is not None and set(mapping.values()) - set(available):
            raise ValueError('Reviewed table contains unknown SuperAnimal bodyparts')
        return mapping

    def prepare(self, conversion_table: Path) -> ModelStageResult:
        self.check_benchmark_separation()
        conversion_table = conversion_table.resolve()
        data = self.data_identity()
        identity = dict(base=self.base(), mode=self.settings['mode'], data=data,
                        mapping=file_hash(conversion_table), seed=self.settings['seed'],
                        fraction=self.settings['train_fraction'], shuffle=self.settings['shuffle'])
        previous = self.previous('prepare', identity)
        if previous:
            return previous
        if self.model_folder().exists():
            raise ValueError('Selected shuffle already has DLC model files; choose a new shuffle instead of overwriting them')
        base = call(api('deeplabcut.pose_estimation_pytorch.modelzoo', 'load_super_animal_config'), super_animal=self.settings['superanimal'],
                    model_name=self.settings['model_name'], detector_name=self.settings['detector_name'], max_individuals=1)
        mapping = self.mapping(conversion_table, base['metadata']['bodyparts'])
        cfgpath = self.runtime_config()
        cfg = read_yaml(cfgpath)
        if cfg['TrainingFraction'] != [self.settings['train_fraction']]:
            raise ValueError('Project TrainingFraction must match the configured single train_fraction')
        call(api('deeplabcut.modelzoo.utils', 'create_conversion_table'), config=str(cfgpath),
             super_animal=self.settings['superanimal'], project_to_super_animal=mapping)
        cfg = api('deeplabcut.utils.auxiliaryfunctions', 'read_config')(str(cfgpath))
        weight = call(api('deeplabcut.modelzoo', 'build_weight_init'), cfg=cfg,
                      super_animal=self.settings['superanimal'], model_name=self.settings['model_name'],
                      detector_name=self.settings['detector_name'], with_decoder=True,
                      memory_replay=self.settings['mode'] == 'memory_replay')
        splitpath = self.artifacts / 'split.json'
        split_identity = dict(data=data, seed=self.settings['seed'], fraction=self.settings['train_fraction'])
        split = json.loads(splitpath.read_text()) if splitpath.exists() else None
        if split and split['identity'] != split_identity:
            raise ValueError('Development data/split changed; use a new project to preserve comparison splits')
        args = dict(net_type='top_down_' + self.settings['model_name'], detector_type=self.settings['detector_name'],
                    engine=api('deeplabcut', 'Engine').PYTORCH, weight_init=weight, userfeedback=False)
        shuffle = self.settings['shuffle']
        self.record('prepare', identity, [], status='started')
        if split:
            if split['shuffle'] == shuffle:
                raise ValueError('This split already owns the shuffle; recover the original preparation or use a new shuffle')
            call(api('deeplabcut', 'create_training_dataset_from_existing_split'), str(cfgpath),
                 from_shuffle=split['shuffle'], shuffles=[shuffle], **args)
        else:
            random.seed(self.settings['seed'])
            np.random.seed(self.settings['seed'])
            call(api('deeplabcut', 'create_training_dataset'), str(cfgpath), Shuffles=[shuffle], **args)
            write_json(splitpath, dict(identity=split_identity, shuffle=shuffle))
        reviewed = self.variant / 'reviewed-conversion.csv'
        shutil.copyfile(conversion_table, reviewed)
        # Preserve the prepared config separately from the relocatable runtime copy.
        prepared = self.variant / 'prepared-config.yaml'
        shutil.copyfile(cfgpath, prepared)
        model = self.model_folder() / 'train' / 'pytorch_config.yaml'
        if not model.is_file():
            raise RuntimeError(f'DLC did not create model configuration: {model}')
        return self.record('prepare', identity, [reviewed, prepared, splitpath])

    def model_folder(self) -> Path:
        cfg = self.project_config()
        name = f"{cfg['Task']}{cfg['date']}-trainset{int(self.settings['train_fraction'] * 100)}shuffle{self.settings['shuffle']}"
        return self.project / 'dlc-models-pytorch' / f"iteration-{cfg.get('iteration', 0)}" / name

    def prepared_config(self) -> Path:
        path = self.variant / 'prepared-config.yaml'
        if not path.exists():
            raise ValueError('Run model prepare with a reviewed conversion table first')
        cfg = self.relocated_config(read_yaml(path))
        target = self.variant / 'runtime-config.yaml'
        write_yaml(target, cfg)
        return target

    def evaluation_config(self) -> Path:
        if (self.variant / 'prepared-config.yaml').is_file():
            return self.prepared_config()
        # Adopted projects already have DLC split/model metadata of their own.
        if not (self.artifacts / 'initialized.json').exists():
            raise ValueError('Adopt the existing project with model init --adopt before split evaluation')
        return self.runtime_config()

    def train(self, resume_checkpoint: Path | None = None, resume_detector: Path | None = None) -> ModelStageResult:
        self.check_benchmark_separation()
        settings = dict(self.settings.get('training', {}))
        for key in ('epochs', 'batch_size', 'save_epochs'):
            if not isinstance(settings.get(key), int) or settings[key] < 1:
                raise ValueError(f'training.{key} must be an explicit positive integer')
        settings.setdefault('detector_epochs', 0)
        prepared = self.variant / 'prepare.json'
        if not prepared.exists():
            raise ValueError('Run model prepare first')
        record = json.loads(prepared.read_text())
        if record['status'] != 'complete' or record['identity']['data'] != self.data_identity():
            raise ValueError('Preparation is incomplete or development data changed')
        expected = record['identity']
        if (expected['base'] != self.base() or expected['mode'] != self.settings['mode']
                or expected['fraction'] != self.settings['train_fraction'] or expected['seed'] != self.settings['seed']):
            raise ValueError('Training variant differs from preparation; use a new shuffle and prepare it')
        for filename, expected_hash in record.get('output_hashes', {}).items():
            if not (self.project / filename).is_file() or file_hash(self.project / filename) != expected_hash:
                raise ValueError('Prepared artifacts changed; restore them or use a new variant')
        identity = dict(preparation=record['identity'], training=settings)
        previous = self.previous('train', identity)
        if previous:
            return previous
        if (self.variant / 'train.json').exists() and resume_checkpoint is None:
            raise ValueError('Interrupted training requires --resume-checkpoint (and --resume-detector when needed)')
        if (self.variant / 'train.json').exists() and settings['detector_epochs'] > 0 and resume_detector is None:
            raise ValueError('Interrupted detector training requires --resume-detector')
        for path in (resume_checkpoint, resume_detector):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        function = api('deeplabcut.pose_estimation_pytorch.apis', 'train_network')
        self.record('train', identity, [], status='started')
        if resume_checkpoint is not None:
            import uuid
            write_json(self.variant / 'resumes' / f'{uuid.uuid4().hex}.json',
                       {str(p): file_hash(p) for p in (resume_checkpoint, resume_detector) if p is not None})
        call(function, str(self.prepared_config()), shuffle=self.settings['shuffle'],
             snapshot_path=str(resume_checkpoint) if resume_checkpoint else None,
             detector_path=str(resume_detector) if resume_detector else None, **settings)
        snapshots = sorted((self.model_folder() / 'train').glob('snapshot*.pt'))
        if not any('detector' not in p.name for p in snapshots):
            raise RuntimeError('Training returned without a pose checkpoint')
        return self.record('train', identity, snapshots)

    def predict(self, *, source='project', inputs=None, kind='images', checkpoint=None,
                detector_checkpoint=None, overlays=False) -> ModelStageResult:
        from .model_inference import predict
        return predict(self, source=source, inputs=inputs, kind=kind, checkpoint=checkpoint,
                       detector_checkpoint=detector_checkpoint, overlays=overlays)

    def evaluate(self, *, source='project', scope='both', checkpoint=None, detector_checkpoint=None) -> ModelStageResult:
        if scope not in {'split', 'external', 'both'}:
            raise ValueError('Evaluation scope must be split, external, or both')
        outputs = []
        if scope in {'split', 'both'}:
            if source != 'project':
                raise ValueError('Stock models have no DLC train/test split; use --scope external')
            if checkpoint is None or detector_checkpoint is None:
                raise ValueError('Split evaluation requires explicit pose and detector checkpoints')
            checkpoint, detector_checkpoint = Path(checkpoint).resolve(), Path(detector_checkpoint).resolve()
            for path in (checkpoint, detector_checkpoint):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if checkpoint.parent != self.model_folder() / 'train' or detector_checkpoint.parent != checkpoint.parent:
                raise ValueError('Split evaluation checkpoints must belong to the configured shuffle train directory')
            snapshots = api('deeplabcut.pose_estimation_pytorch.apis.utils', 'get_model_snapshots')
            detector_list = call(snapshots, index='all', model_folder=checkpoint.parent,
                                 task=api('deeplabcut.pose_estimation_pytorch.task', 'Task').DETECT)
            detector_index = next((i for i, s in enumerate(detector_list) if Path(s.path).resolve() == detector_checkpoint), None)
            if detector_index is None:
                raise ValueError('Detector checkpoint not found by DLC snapshot resolver')
            evaluation_identity = dict(checkpoint=file_hash(checkpoint), detector=file_hash(detector_checkpoint),
                                       data=self.data_identity(), config=file_hash(self.evaluation_config()),
                                       model_config=file_hash(checkpoint.parent / 'pytorch_config.yaml'),
                                       shuffle=self.settings['shuffle'])
            workspace = self.variant / 'split-evaluation' / digest(evaluation_identity)[:20]
            completion = workspace / 'complete.json'
            completed = json.loads(completion.read_text()) if completion.exists() else None
            if completed and all((workspace / p).is_file() and file_hash(workspace / p) == h
                                 for p, h in completed['hashes'].items()):
                outputs.extend(workspace / p for p in completed['hashes'])
            else:
                # DLC itself caches by scorer/filename, not checkpoint contents. A fresh
                # project view prevents stale split scores if weights are replaced.
                import tempfile
                workspace.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=workspace, prefix='attempt-') as temporary:
                    view = Path(temporary)
                    def link_or_copy(source, destination):
                        if Path(source).suffix == '.pt':
                            try:
                                os.link(source, destination)
                                return destination
                            except OSError:
                                pass
                        return shutil.copy2(source, destination)
                    for name in ('labeled-data', 'training-datasets'):
                        shutil.copytree(self.project / name, view / name, copy_function=link_or_copy)
                    relative_model = self.model_folder().relative_to(self.project)
                    shutil.copytree(self.model_folder(), view / relative_model, copy_function=link_or_copy)
                    cfg = read_yaml(self.evaluation_config())
                    cfg['project_path'] = str(view)
                    write_yaml(view / 'config.yaml', cfg)
                    model_config = view / relative_model / 'train' / 'pytorch_config.yaml'
                    model_cfg = read_yaml(model_config)
                    model_cfg.setdefault('metadata', {}).update(project_path=str(view), pose_config_path=str(model_config))
                    write_yaml(model_config, model_cfg)
                    call(api('deeplabcut.pose_estimation_pytorch.apis', 'evaluate_network'), str(view / 'config.yaml'),
                         shuffles=[self.settings['shuffle']], snapshots_to_evaluate=[checkpoint.stem],
                         detector_snapshot_index=detector_index, per_keypoint_evaluation=True)
                    reports = sorted((view / 'evaluation-results-pytorch').rglob('*.csv'))
                    if not reports:
                        raise RuntimeError('DLC split evaluation returned without CSV reports')
                    saved = []
                    for report in reports:
                        target = workspace / report.relative_to(view)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(report, target)
                        saved.append(target)
                    write_json(completion, dict(identity=evaluation_identity,
                               hashes={p.relative_to(workspace).as_posix(): file_hash(p) for p in saved}))
                    outputs.extend(saved)
        if scope in {'external', 'both'}:
            settings = self.settings.get('benchmark', {})
            root = self.path(settings['labels'])
            label_paths = label_tables(root)
            labels = pd.concat([read_table(p, folder=p.parent.relative_to(root).as_posix()) for p in label_paths])
            if labels.index.has_duplicates:
                raise ValueError('Duplicate benchmark frame identities')
            images = [root / name for name in labels.index]
            if not all(p.is_file() for p in images):
                raise ValueError('Benchmark labels reference missing images')
            prediction = self.predict(source=source, inputs=images, checkpoint=checkpoint,
                                      detector_checkpoint=detector_checkpoint)
            predictions = read_table(prediction.artifacts[0], normalize_index=False)
            # Prediction manifest stores full source paths; convert them to dataset-relative IDs.
            predictions.index = [Path(p).relative_to(root).as_posix() for p in predictions.index]
            mapping = None
            mapping_hash = None
            if source == 'stock':
                mapping_path = self.path(settings['conversion_table'])
                mapping = self.mapping(mapping_path, require_all=False)
                mapping_hash = file_hash(mapping_path)
            points, coverage = point_errors(labels, predictions, settings['bodyparts'], mapping)
            _, development = self.dataset()
            overlap = [p.relative_to(root).as_posix() for p in images
                       if file_hash(p) in set(development.values()) or p.relative_to(root).as_posix() in development]
            coverage.update(overlap_frames=overlap, held_out=not bool(overlap))
            provenance = dict(prediction=prediction.details, benchmark=settings, mapping_hash=mapping_hash,
                              labels={str(p): file_hash(p) for p in label_paths})
            destination = self.variant / 'evaluation' / digest(provenance)[:16]
            outputs.extend(write_report(points, coverage, destination, threshold=settings.get('confidence', .5),
                                        radius=settings.get('radius_px', 5), groups=settings.get('groups'), provenance=provenance))
        return ModelStageResult('evaluate', 'complete', outputs)
