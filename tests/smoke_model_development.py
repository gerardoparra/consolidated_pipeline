"""Opt-in real GPU smoke workflow; never discovered by the unit-test runner."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import Setup
from pipeline.model_development import ModelDevelopment


def main():
    parser = argparse.ArgumentParser(description='Train both modes on small configured data and benchmark stock/project models')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--experiments', nargs=2, required=True, metavar=('FINETUNE', 'REPLAY'))
    parser.add_argument('--conversion-table', type=Path, required=True, help='Manually reviewed mapping')
    parser.add_argument('--video', type=Path, required=True, help='Short inspection video')
    parser.add_argument('--environment', choices=('local', 'hpc'), default='hpc')
    parser.add_argument('--adopt', action='store_true')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke test requires CUDA in the DLC PyTorch environment')
    setup = Setup.load(args.config)
    developments = [ModelDevelopment(setup.data, name, args.environment) for name in args.experiments]
    if {d.settings['mode'] for d in developments} != {'finetune', 'memory_replay'}:
        raise ValueError('Provide one experiment per training mode')
    if developments[0].project != developments[1].project:
        raise ValueError('Smoke variants must share a project and split')
    for index, dev in enumerate(developments):
        with dev.lock():
            dev.initialize(adopt=args.adopt or index > 0)
            dev.match()
            dev.prepare(dev.path(args.conversion_table))
            training = dev.train()
            pose = sorted(p for p in training.artifacts if p.suffix == '.pt' and 'detector' not in p.name)
            best = [p for p in pose if 'best' in p.name]
            pose = (best or pose)[-1]
            detector = sorted(p for p in training.artifacts if p.suffix == '.pt' and 'detector' in p.name)[-1]
            print(dev.evaluate(source='stock', scope='external'))
            print(dev.evaluate(checkpoint=pose, detector_checkpoint=detector))
            print(dev.predict(inputs=[args.video], kind='videos', checkpoint=pose,
                              detector_checkpoint=detector, overlays=True))


if __name__ == '__main__':
    main()
