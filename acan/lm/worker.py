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
    parser.add_argument('--kind', choices=['train','generate','evaluate','small-test','build'], default='train')
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
                from .resources import check_profile
                import torch
                try:
                    check_profile(options,torch.cuda.is_available())
                except ValueError as exc:
                    Store(args.root).update(args.job,status='failed',error=str(exc),pause=0)
                    raise
                from .research_trainer import run_training
            else:
                from .trainer import run_training
            run_training(args.root, args.job, stop)
            os.close(args.lease_fd)                             # free the worker before the next project of a series
            args.lease_fd = None
            from .manager import continue_series
            from .store import Store
            continue_series(Store(args.root), args.job)
        elif args.kind=='build':
            from .dialogue_data import run_in_worker
            run_in_worker(args.root, args.generation, stop)
        elif args.kind=='small-test':
            from .small_test import run_in_worker
            run_in_worker(args.root, args.generation, stop)
        elif args.kind=='evaluate':
            from .evaluation import evaluate_checkpoint
            evaluate_checkpoint(args.root,args.generation,stop)
        else:
            from .generation import generate
            generate(args.root, args.generation)
    finally:
        if args.lease_fd is not None:
            os.close(args.lease_fd)


if __name__ == '__main__':
    main()
