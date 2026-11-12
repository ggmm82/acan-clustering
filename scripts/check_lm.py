"""Mandatory offline LM mechanics gate; does not certify billion-token claims."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/lm-audit.json')
    args=parser.parse_args()
    source=json.loads((ROOT/'references/generative/source-manifest.json').read_text())
    for name, expected in source.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise SystemExit('LM reference changed; review source correspondence: '+name)
    import torch, transformers, peft
    from acan.lm.attention import PROFILE
    from acan.lm.checkpoints import contract
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_lm*.py')
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_exclusive_cp.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_history_navigation.py'))
    from acan.lm.audit_contract import REQUIRED_TESTS,MIN_TESTS,PROTOCOL
    def cases(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from cases(item)
            else:yield item.id().rsplit('.',1)[-1]
    present=set(cases(suite))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    verified=result.wasSuccessful() and not result.skipped and REQUIRED_TESTS.issubset(present)
    passed=verified and result.testsRun>=MIN_TESTS
    report=dict(status='passed' if passed else 'failed', profile=PROFILE, contract=contract(),
                controlled_protocol=PROTOCOL,controls_verified=verified,required_controls=sorted(REQUIRED_TESTS),
                reference_sha256=source, full_paper_conformance=False, gpu_verified=False,
                pretrained_general_quality_verified=False, tests_run=result.testsRun, skipped=len(result.skipped),
                torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                commit=os.environ.get('ACAN_SOURCE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                run_id=int(os.environ.get('GITHUB_RUN_ID',0)),
                tree=os.environ.get('ACAN_SOURCE_TREE') or subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),log=stream.getvalue())
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(stream.getvalue())
    print(json.dumps({key:report[key] for key in ('status','profile','tests_run','skipped','gpu_verified')}))
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
