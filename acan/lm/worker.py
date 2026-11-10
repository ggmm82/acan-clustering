"""Owned subprocess CLI; a signal requests a checkpointed pause, never a reset."""
import argparse
import os
import signal
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--lease-fd', type=int, required=True)
    parser.add_argument('--kind', choices=['train','generate','evaluate'], default='train')
    parser.add_argument('--generation')
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        if args.kind == 'train':
            from .store import Store
            options=Store(args.root).job(args.job)['config']
            if options.get('protocol')=='controlled-v1':
                from .research_trainer import run_training
            else:
                from .trainer import run_training
            run_training(args.root, args.job, stop)
        elif args.kind=='evaluate':
            from .evaluation import evaluate_checkpoint
            evaluate_checkpoint(args.root,args.generation,stop)
        else:
            from .generation import generate
            generate(args.root, args.generation)
    finally:
        os.close(args.lease_fd)


if __name__ == '__main__':
    main()
