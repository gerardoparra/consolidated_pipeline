"""DLC-independent benchmark metrics.

Consolidates the evaluation workflow in Niek Andresen's evaluate.py and
evaluate_superanimal.py (August 2023), with explicit coverage and frame identity.
"""
from __future__ import annotations

import csv
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_id(value, folder: str = '') -> str:
    text = '/'.join(map(str, value)) if isinstance(value, tuple) else str(value)
    text = text.replace('\\', '/')
    if 'labeled-data/' in text:
        text = text.split('labeled-data/', 1)[1]
    elif folder:
        token = folder.strip('/') + '/'
        if token in text:
            text = token + text.rsplit(token, 1)[1]
        elif '/' not in text:
            text = token + text
        else:
            raise ValueError(f'Frame path cannot be resolved within {folder}: {text}')
    if '..' in text.split('/'):
        raise ValueError(f'Unsafe frame identity: {text}')
    return text


def read_table(path: str | Path, *, folder: str = '', normalize_index: bool = True) -> pd.DataFrame:
    """Read one scorer/animal, retaining folder-qualified frame identities."""
    path = Path(path)
    if path.suffix.lower() in {'.h5', '.hdf5'}:
        frame = pd.read_hdf(path)
    else:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(itertools.islice(csv.reader(stream), 5))
        coordinate_row = next((i for i, row in enumerate(rows) if row and row[0] == 'coords'), None)
        if coordinate_row not in (2, 3):
            raise ValueError(f'Expected DLC multi-header CSV: {path}')
        row = rows[coordinate_row]
        first = next(i for i, value in enumerate(row) if value in {'x', 'y', 'likelihood'})
        frame = pd.read_csv(path, header=list(range(coordinate_row + 1)), index_col=list(range(first)))
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels not in (2, 3, 4):
        raise ValueError(f'Invalid DLC columns: {path}')
    while frame.columns.nlevels > 2:
        if len(frame.columns.get_level_values(0).unique()) != 1:
            raise ValueError(f'Multiple scorers or animals in {path}')
        frame.columns = frame.columns.droplevel(0)
    frame.columns = frame.columns.set_names(['bodyparts', 'coords'])
    if frame.columns.has_duplicates:
        raise ValueError(f'Duplicate keypoint columns: {path}')
    if normalize_index:
        frame.index = pd.Index([frame_id(i, folder) for i in frame.index], name='frame')
    if frame.index.has_duplicates:
        raise ValueError(f'Duplicate frame identities: {path}')
    return frame.apply(pd.to_numeric, errors='raise')


def label_tables(root: Path) -> list[Path]:
    """Prefer HDF5 over its paired CSV, rejecting ambiguous label scorers."""
    folders = sorted({p.parent for p in root.rglob('CollectedData_*') if p.suffix in {'.h5', '.csv'}})
    result = []
    for folder in folders:
        candidates = sorted(folder.glob('CollectedData_*.h5')) or sorted(folder.glob('CollectedData_*.csv'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one label scorer in {folder}')
        result.append(candidates[0])
    if not result:
        raise ValueError(f'No DLC labels under {root}')
    return result


def point_errors(labels: pd.DataFrame, predictions: pd.DataFrame, parts: list[str],
                 mapping: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    if not parts or len(parts) != len(set(parts)):
        raise ValueError('Benchmark bodyparts must be a nonempty unique list')
    mapping = {part: part for part in parts} if mapping is None else mapping
    supported = set(predictions.columns.get_level_values('bodyparts'))
    unsupported = [part for part in parts if mapping.get(part) not in supported]
    rows = []
    pred = predictions.reindex(labels.index)
    for part in parts:
        if not all((part, c) in labels for c in ('x', 'y')):
            raise ValueError(f'Ground truth lacks x/y for {part}')
        actual = labels.loc[:, [(part, 'x'), (part, 'y')]].to_numpy(float)
        target = mapping.get(part)
        values = np.full((len(labels), 3), np.nan)
        if part not in unsupported:
            if not all((target, c) in pred for c in ('x', 'y', 'likelihood')):
                raise ValueError(f'Prediction lacks x/y/likelihood for {target}')
            values = pred.loc[:, [(target, c) for c in ('x', 'y', 'likelihood')]].to_numpy(float)
        visible = np.isfinite(actual).all(axis=1)
        finite = np.isfinite(values).all(axis=1)
        distance = np.linalg.norm(values[:, :2] - actual, axis=1)
        for i, name in enumerate(labels.index):
            rows.append(dict(frame=name, folder=name.rsplit('/', 1)[0] if '/' in name else '',
                             bodypart=part, visible=bool(visible[i]), finite=bool(finite[i]),
                             likelihood=values[i, 2], error_px=distance[i], supported=part not in unsupported))
    coverage = dict(label_frames=len(labels), prediction_frames=len(predictions),
                    missing_frames=sorted(set(labels.index) - set(predictions.index)),
                    extra_frames=sorted(set(predictions.index) - set(labels.index)),
                    unsupported_bodyparts=unsupported)
    return pd.DataFrame(rows), coverage


def metrics(points: pd.DataFrame, threshold: float, radius: float = 5.0) -> dict:
    visible = points.visible.to_numpy(bool)
    detected = points.finite.to_numpy(bool) & (points.likelihood.to_numpy(float) > threshold)
    tp = visible & detected
    errors = points.loc[tp, 'error_px']
    ratio = lambda a, b: float(a / b) if b else float('nan')
    per_part = points.loc[tp].groupby('bodypart').error_px.mean()
    return dict(threshold=threshold, possible_points=len(points), visible_points=int(visible.sum()),
                detected_points=int(detected.sum()), true_positives=int(tp.sum()),
                mean_pixel_error_points=float(errors.mean()), mean_pixel_error_parts=float(per_part.mean()),
                visibility_precision=ratio(tp.sum(), detected.sum()),
                visibility_recall=ratio(tp.sum(), visible.sum()),
                visibility_accuracy=ratio((visible == detected).sum(), len(points)),
                outlier_fraction_50px=ratio((errors > 50).sum(), len(errors)),
                localization_success=ratio((errors <= radius).sum(), visible.sum()), radius_px=radius)


def write_report(points: pd.DataFrame, coverage: dict, output: Path, *, threshold: float = .5,
                 radius: float = 5, groups: dict[str, list[str]] | None = None,
                 provenance: dict | None = None) -> list[Path]:
    if not 0 <= threshold <= 1 or radius <= 0:
        raise ValueError('Confidence must be in [0,1] and localization radius positive')
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    selections = [('overall', 'all', points)]
    for kind in ('folder', 'bodypart'):
        selections.extend((kind, str(name), subset) for name, subset in points.groupby(kind))
    for name, parts in (groups or {}).items():
        unknown = set(parts) - set(points.bodypart)
        if unknown:
            raise ValueError(f'Unknown bodyparts in group {name}: {sorted(unknown)}')
        selections.append(('group', name, points[points.bodypart.isin(parts)]))
    for kind, name, subset in selections:
        for cutoff in sorted(set([threshold] + [n / 10 for n in range(10)])):
            summaries.append(dict(scope=kind, name=name, **metrics(subset, cutoff, radius)))
    paths = [output / 'summary.csv', output / 'points.csv', output / 'report.json', output / 'confidence.png']
    pd.DataFrame(summaries).to_csv(paths[0], index=False)
    points.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(dict(coverage=coverage, provenance=provenance or {}), indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(points.loc[np.isfinite(points.likelihood), 'likelihood'], bins=20, range=(0, 1))
    ax.set(xlabel='Prediction confidence', ylabel='Points')
    fig.savefig(paths[3], bbox_inches='tight')
    plt.close(fig)
    return paths
