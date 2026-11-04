"""Executable paper contracts, versioned limitations and deployment audit receipt."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

import numpy as np

from acan.core.network import HAS_TORCH
from acan.generative import ConceptLanguageModel, demo_corpus
from acan.paper_contract import PAPER_SHA256, CONFORMANCE, GAPS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/paper-audit.json')
    parser.add_argument('--require-torch', action='store_true')
    args = parser.parse_args()
    paper = ROOT / 'paper/Algebraic_Concept_Attention_Paper.tex'
    if hashlib.sha256(paper.read_bytes()).hexdigest() != PAPER_SHA256:
        raise SystemExit('Paper changed: review and update the correspondence contract before deployment.')
    if args.require_torch and not HAS_TORCH:
        raise SystemExit('Required CPU PyTorch backend is missing; audit cannot pass by skipping it.')
    if HAS_TORCH:
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(1)
    suite = unittest.TestSuite()
    for pattern in ('test_paper_contract.py', 'test_acan.py', 'test_language.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun >= 13 and (not args.require_torch or not result.skipped)
    report = dict(contract_version=1, status='passed' if passed else 'failed',
                  conformance=CONFORMANCE, full_paper_conformance=False, paper_sha256=PAPER_SHA256,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
                  source_clean=not bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                  run_id=int(os.environ.get('GITHUB_RUN_ID', 0)),
                  audit_seed=42, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), torch_verified=HAS_TORCH,
                  python=platform.python_version(), numpy=np.__version__, gaps=GAPS, log=stream.getvalue())
    if HAS_TORCH:
        import torch
        report['torch'] = torch.__version__
    if passed:
        model = ConceptLanguageModel(context=4, slots=8).fit(demo_corpus(), epochs=15)
        report['language_prototype'] = model.report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in ('status', 'conformance', 'tests_run', 'skipped', 'torch_verified')}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
