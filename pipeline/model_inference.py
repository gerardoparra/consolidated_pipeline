"""One fixed-checkpoint inference path for stock and developing DLC models."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_development import ModelStageResult, api, call, digest, read_yaml, write_json
from .model_evaluation import file_hash


def resolve_model(development, source, checkpoint, detector_checkpoint):
    settings = development.settings
    if source == 'stock':
        config = call(api('deeplabcut.pose_estimation_pytorch.modelzoo', 'load_super_animal_config'), super_animal=settings['superanimal'],
                      model_name=settings['model_name'], detector_name=settings['detector_name'], max_individuals=1)
        checkpoint = call(api('deeplabcut.pose_estimation_pytorch.modelzoo', 'get_super_animal_snapshot_path'), dataset=settings['superanimal'], model_name=settings['model_name'])
        detector_checkpoint = call(api('deeplabcut.pose_estimation_pytorch.modelzoo', 'get_super_animal_snapshot_path'), dataset=settings['superanimal'], model_name=settings['detector_name'])
    elif source == 'project':
        if checkpoint is None or detector_checkpoint is None:
            raise ValueError('Project inference requires --checkpoint and --detector-checkpoint; latest is never implicit')
        config_path = development.model_folder() / 'train' / 'pytorch_config.yaml'
        config = read_yaml(config_path)
    else:
        raise ValueError('source must be stock or project')
    checkpoint, detector_checkpoint = Path(checkpoint).resolve(), Path(detector_checkpoint).resolve()
    for path in (checkpoint, detector_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = dict(source=source, base=development.base(), config=digest(config),
                    checkpoint=file_hash(checkpoint), detector_checkpoint=file_hash(detector_checkpoint),
                    dlc_version=importlib.import_module('deeplabcut').__version__)
    return config, checkpoint, detector_checkpoint, identity


def prediction_frame(predictions: dict, config: dict) -> pd.DataFrame:
    """Project memory replay uses the training conversion array, never guessed names."""
    parts = config['metadata']['bodyparts']
    initialization = config.get('train_settings', {}).get('weight_init', {}) or {}
    conversion = initialization.get('conversion_array') if initialization.get('memory_replay') else None
    rows, indices = [], []
    for name, prediction in predictions.items():
        values = np.asarray(prediction['bodyparts'], dtype=float)
        if values.ndim != 3 or values.shape[0] > 1 or values.shape[-1] != 3:
            raise ValueError(f'Expected one animal with x/y/confidence: {name}, {values.shape}')
        if values.shape[0] == 0:
            values = np.full((1, len(parts), 3), np.nan)
        elif conversion is not None:
            if any(i < 0 or i >= values.shape[1] for i in conversion):
                raise ValueError('Invalid memory-replay conversion indices')
            values = values[:, conversion, :]
        if values.shape[1] != len(parts):
            raise ValueError('Prediction channels do not match model metadata')
        rows.append(values.reshape(-1))
        indices.append(str(name).replace('\\', '/'))
    columns = pd.MultiIndex.from_product([['model-development'], parts, ['x', 'y', 'likelihood']],
                                         names=['scorer', 'bodyparts', 'coords'])
    return pd.DataFrame(rows, index=indices, columns=columns)


def render_overlay(source: Path, frame: pd.DataFrame, destination: Path, video: bool):
    import cv2
    def paint(image, row):
        points = row.to_numpy().reshape(-1, 3)
        for x, y, confidence in points:
            if np.isfinite([x, y, confidence]).all() and confidence > .5:
                cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        return image
    if not video:
        image = cv2.imread(str(source))
        if image is None or not cv2.imwrite(str(destination), paint(image, frame.iloc[0])):
            raise RuntimeError(f'Cannot render image overlay: {source}')
        return
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS),
                             (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError(f'Cannot render video overlay: {source}')
        for _, row in frame.iterrows():
            ok, image = capture.read()
            if not ok:
                raise RuntimeError('Video shorter than prediction table')
            writer.write(paint(image, row))
    finally:
        capture.release()
        writer.release()


def predict(development, *, source='project', inputs=None, kind='images', checkpoint=None,
            detector_checkpoint=None, overlays=False):
    if kind not in {'images', 'videos'}:
        raise ValueError('kind must be images or videos')
    if inputs is None:
        inputs = development.settings.get('prediction_inputs', [])
    suffixes = {'.png', '.jpg', '.jpeg'} if kind == 'images' else {'.mp4', '.avi', '.mkv', '.mov'}
    paths = []
    for value in inputs:
        path = development.path(value)
        paths.extend(sorted(p for p in path.rglob('*') if p.suffix.lower() in suffixes) if path.is_dir() else [path])
    paths = sorted(set(paths))
    if not paths or any(not p.is_file() or p.suffix.lower() not in suffixes for p in paths):
        raise ValueError(f'Prediction requires existing {kind} inputs')
    config, checkpoint, detector_checkpoint, identity = resolve_model(development, source, checkpoint, detector_checkpoint)
    identity.update(inputs={str(p): file_hash(p) for p in paths}, kind=kind, overlays=overlays)
    output = development.variant / 'predictions' / digest(identity)[:20]
    manifest = output / 'prediction.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        files = [Path(p) for p in previous['outputs']]
        if previous['identity'] == identity and all(p.is_file() and file_hash(p) == previous['hashes'][str(p)] for p in files):
            return ModelStageResult('predict', 'reused', files, identity)
    # A failed attempt must not reuse DLC's partial outputs.
    import tempfile
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    with tempfile.TemporaryDirectory(dir=output, prefix='attempt-') as temporary:
        work = Path(temporary)
        if kind == 'images':
            function = api('deeplabcut.pose_estimation_pytorch.apis', 'analyze_image_folder')
            predictions = call(function, model_cfg=config, images=[str(p) for p in paths],
                               snapshot_path=checkpoint, detector_path=detector_checkpoint, max_individuals=1)
            table = prediction_frame(predictions, config)
            file = work / 'predictions.h5'
            table.to_hdf(file, key='predictions', mode='w')
            produced.append(file)
            if overlays:
                for i, path in enumerate(paths):
                    key = path.as_posix()
                    if key in table.index:
                        overlay = work / f'{i:06d}-{path.stem}.png'
                        render_overlay(path, table.loc[[key]], overlay, False)
                        produced.append(overlay)
        else:
            pose = call(api('deeplabcut.pose_estimation_pytorch.apis', 'get_pose_inference_runner'),
                        model_config=config, snapshot_path=checkpoint, max_individuals=1)
            detector = call(api('deeplabcut.pose_estimation_pytorch.apis', 'get_detector_inference_runner'),
                            model_config=config, snapshot_path=detector_checkpoint, max_individuals=1)
            function = api('deeplabcut.pose_estimation_pytorch.apis', 'video_inference')
            for i, path in enumerate(paths):
                predictions = call(function, str(path), pose_runner=pose, detector_runner=detector)
                table = prediction_frame(dict(enumerate(predictions)), config)
                file = work / f'{i:06d}-{path.stem}.h5'
                table.to_hdf(file, key='predictions', mode='w')
                produced.append(file)
                if overlays:
                    overlay = file.with_suffix('.mp4')
                    render_overlay(path, table, overlay, True)
                    produced.append(overlay)
        if not produced or any(p.stat().st_size == 0 for p in produced):
            raise RuntimeError('Inference did not produce valid outputs')
        final = []
        for file in produced:
            destination = output / file.name
            file.replace(destination)
            final.append(destination)
    write_json(manifest, dict(identity=identity, outputs=[str(p) for p in final],
                              hashes={str(p): file_hash(p) for p in final}))
    return ModelStageResult('predict', 'complete', final, identity)
