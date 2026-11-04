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
    parser.add_argument('--kind', choices=['train','generate'], default='train')
    parser.add_argument('--generation')
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        if args.kind == 'train':
            from .trainer import run_training
            run_training(args.root, args.job, stop)
        else:
            from .generation import generate
            generate(args.root, args.generation)
    finally:
        os.close(args.lease_fd)


if __name__ == '__main__':
    main()
