"""CLI and frozen Slurm execution for model development."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import uuid

from .config import Setup
from .job_manager import JobResult, _slurm_options, _submit, _is_windows
from .model_development import ModelDevelopment, write_json
from .model_evaluation import file_hash


def add_model_parser(commands, default_setup):
    group = commands.add_parser('model', help='Develop and benchmark SuperAnimal models')
    stages = group.add_subparsers(dest='model_stage', required=True)
    for stage in ('init', 'match', 'prepare', 'train', 'predict', 'evaluate'):
        sub = stages.add_parser(stage)
        sub.add_argument('experiment', help='Named model_development experiment')
        sub.add_argument('--config', type=Path, default=default_setup)
        sub.add_argument('--environment', choices=('auto', 'local', 'hpc'), default='auto')
        sub.add_argument('--slurm', action='store_true', help='Submit in the HPC DLC container')
        sub.add_argument('--preview', action='store_true', help='Print resolved settings/job without executing')
        if stage == 'init':
            sub.add_argument('--adopt', action='store_true')
        if stage == 'prepare':
            sub.add_argument('--conversion-table', type=Path, required=True, help='Explicitly reviewed gt,MasterName CSV')
        if stage == 'train':
            sub.add_argument('--resume-checkpoint', type=Path)
            sub.add_argument('--resume-detector', type=Path)
        if stage in ('predict', 'evaluate'):
            sub.add_argument('--source', choices=('stock', 'project'), default='project')
            sub.add_argument('--checkpoint', type=Path)
            sub.add_argument('--detector-checkpoint', type=Path)
        if stage == 'predict':
            sub.add_argument('--inputs', nargs='+', type=Path)
            sub.add_argument('--kind', choices=('images', 'videos'), default='images')
            sub.add_argument('--overlays', action='store_true')
        if stage == 'evaluate':
            sub.add_argument('--scope', choices=('split', 'external', 'both'), default='both')


def stage_arguments(args, config):
    command = ['model', args.model_stage, args.experiment, '--config', str(config), '--environment', 'hpc']
    for key in ('conversion_table', 'resume_checkpoint', 'resume_detector', 'source', 'checkpoint',
                'detector_checkpoint', 'kind', 'scope'):
        value = getattr(args, key, None)
        if value is not None:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    for key in ('adopt', 'overlays'):
        if getattr(args, key, False):
            command.append('--' + key)
    if getattr(args, 'inputs', None):
        command.extend(['--inputs', *map(str, args.inputs)])
    return command


def job_script(setup, development, args, frozen):
    host = setup['hosts']['hpc']
    repo = Path(host['repository_dir'])
    rw = {development.root, development.project, Path(host['cache_dir']), Path(host['tmp_dir']), frozen.parent}
    ro = {repo}
    for name in ('videos', 'labeled_data', 'prediction_inputs'):
        for value in development.settings.get(name, []):
            path = development.path(value)
            ro.add(path if path.is_dir() else path.parent)
    benchmark = development.settings.get('benchmark', {})
    for key in ('labels', 'conversion_table'):
        if benchmark.get(key):
            path = development.path(benchmark[key])
            ro.add(path if path.is_dir() else path.parent)
    for key in ('conversion_table', 'checkpoint', 'detector_checkpoint', 'resume_checkpoint', 'resume_detector'):
        if getattr(args, key, None):
            ro.add(development.path(getattr(args, key)).parent)
    for path in getattr(args, 'inputs', None) or []:
        path = development.path(path)
        ro.add(path if path.is_dir() else path.parent)
    # The project contains copied training data; benchmark/source mounts remain read-only.
    ro = {p for p in ro if p not in rw and p != development.project and development.project not in p.parents}
    command = ['singularity', 'exec', '--nv', '--containall']
    for path in sorted(rw):
        command.extend(['--bind', f'{path}:{path}:rw'])
    for path in sorted(ro):
        command.extend(['--bind', f'{path}:{path}:ro'])
    command.extend([host['sandbox'], 'env', f'PYTHONPATH={repo}',
                    f"DLC_MODELZOO_PATH={host['cache_dir']}/dlc_modelzoo",
                    f"TORCH_HOME={host['cache_dir']}/torch_cache", f"HF_HOME={host['cache_dir']}/hf_cache",
                    f"TMPDIR={host['tmp_dir']}", 'python3', '-m', 'pipeline.main', *stage_arguments(args, frozen)])
    return '#!/bin/bash\nset -euo pipefail\nmodule load ' + shlex.quote(host['singularity_module']) + '\nexec ' + shlex.join(command) + '\n'


def submit_model(setup, development, args):
    if _is_windows():
        raise RuntimeError('Slurm submission must run on the HPC host; use --preview locally')
    host = setup['hosts']['hpc']
    for path in (Path(host['repository_dir']) / 'pipeline' / 'main.py', Path(host['sandbox'])):
        if not path.exists():
            raise FileNotFoundError(path)
    submission = development.root / '.submissions' / uuid.uuid4().hex
    submission.mkdir(parents=True)
    frozen = submission / 'setup.json'
    # Freeze a selected reviewed table as well as the setup; workers verify checkpoint hashes.
    if getattr(args, 'conversion_table', None):
        import shutil
        source = development.path(args.conversion_table)
        target = submission / 'reviewed.csv'
        shutil.copyfile(source, target)
        args = copy.copy(args)
        args.conversion_table = target
    frozen_setup = copy.deepcopy(setup)
    checks = {}
    for key in ('checkpoint', 'detector_checkpoint', 'resume_checkpoint', 'resume_detector'):
        if getattr(args, key, None):
            path = development.path(getattr(args, key))
            checks[str(path)] = file_hash(path)
    if development.config.exists():
        checks[str(development.config)] = file_hash(development.config)
    for path in (development.model_folder() / 'train' / 'pytorch_config.yaml',
                 development.variant / 'prepared-config.yaml') if development.config.exists() else []:
        if path.is_file():
            checks[str(path)] = file_hash(path)
    if development.config.exists() and args.model_stage in {'match', 'prepare', 'train'}:
        frozen_setup['_model_submission_data'] = development.data_identity()
    frozen_setup['_model_submission_checks'] = checks
    write_json(frozen, frozen_setup)
    for folder in (development.project, Path(host['cache_dir']), Path(host['tmp_dir'])):
        folder.mkdir(parents=True, exist_ok=True)
    script = job_script(setup, development, args, frozen)
    script_path = submission / 'job.sh'
    script_path.write_text(script, encoding='utf-8', newline='\n')
    command = ['sbatch', '--parsable', *_slurm_options(setup), '--output', str(submission / '%j.out'),
               '--error', str(submission / '%j.err'), '--chdir', host['repository_dir']]
    job_id = _submit(command, script)
    write_json(submission / 'submission.json', dict(job_id=job_id, stage=args.model_stage, experiment=args.experiment))
    return JobResult('submitted', shlex.join(command), job_id,
                     str(submission / f'{job_id}.out'), str(submission / f'{job_id}.err'))


def run_model_command(args):
    setup = Setup.load(args.config)
    for path, expected in setup.data.get('_model_submission_checks', {}).items():
        if not Path(path).is_file() or file_hash(Path(path)) != expected:
            raise ValueError(f'Queued model input changed since submission: {path}')
    environment = setup.host_name(args.environment)
    development = ModelDevelopment(setup.data, args.experiment, environment)
    if ('_model_submission_data' in setup.data
            and setup.data['_model_submission_data'] != development.data_identity()):
        raise ValueError('Queued development dataset changed since submission')
    # Resolve user-specified paths consistently in direct and queued execution.
    for key in ('conversion_table', 'resume_checkpoint', 'resume_detector', 'checkpoint', 'detector_checkpoint'):
        if getattr(args, key, None) is not None:
            setattr(args, key, development.path(getattr(args, key)))
    if args.preview:
        print(json.dumps(dict(project=str(development.project), settings=development.settings), indent=2))
        if environment == 'hpc':
            print(job_script(setup.data, development, args, development.root / '.submissions' / 'PREVIEW' / 'setup.json'))
        return
    if args.slurm:
        if environment != 'hpc':
            raise ValueError('--slurm requires --environment hpc')
        result = submit_model(setup.data, development, args)
        print(f'Slurm {result.status}: {result.job_id}\n{result.output_log}\n{result.error_log}')
        return
    with development.lock():
        stage = args.model_stage
        if stage == 'init':
            result = development.initialize(adopt=args.adopt)
        elif stage == 'match':
            result = development.match()
        elif stage == 'prepare':
            result = development.prepare(args.conversion_table)
        elif stage == 'train':
            result = development.train(args.resume_checkpoint, args.resume_detector)
        elif stage == 'predict':
            result = development.predict(source=args.source, inputs=args.inputs, kind=args.kind,
                                         checkpoint=args.checkpoint, detector_checkpoint=args.detector_checkpoint,
                                         overlays=args.overlays)
        else:
            result = development.evaluate(source=args.source, scope=args.scope,
                                          checkpoint=args.checkpoint, detector_checkpoint=args.detector_checkpoint)
    print(f'{result.stage}: {result.status}')
    for path in result.artifacts:
        print(path)
