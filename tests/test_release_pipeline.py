import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts import release_pipeline as pipeline
from scripts.configure_github import policy

REPO = 'ggmm82/acan-clustering'
COMMIT = 'a' * 40
TREE = 'b' * 40
IMAGE = f'ghcr.io/{REPO}@sha256:' + '1' * 64


def identity():
    return dict(schema=4, repository=REPO, workflow=pipeline.WORKFLOW,
                event='pull_request', source_commit=COMMIT, tree=TREE,
                head_sha='c' * 40, head_repository=REPO, pull_request=1,
                run_id=23, run_attempt=1)


class TestReleasePipeline(unittest.TestCase):
    def test_sealed_digest_tree_and_compose_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())
            self.assertEqual(pipeline.validate(directory, REPO, TREE)['image'], IMAGE)
            with self.assertRaises(ValueError):
                pipeline.validate(directory, REPO, 'd' * 40)
            manifest = Path(directory) / 'runtime-manifest.json'
            value = json.loads(manifest.read_text())
            value['image'] = f'ghcr.io/{REPO}:latest'
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                pipeline.validate(directory, REPO)
            seal(directory, IMAGE, identity())
            compose = Path(directory) / 'compose.release.yaml'
            content = json.loads(compose.read_text())
            content['services']['acan']['privileged'] = True
            compose.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                pipeline.validate(directory, REPO)
            # Even a recomputed checksum must not permit an altered runtime.
            value = json.loads(manifest.read_text())
            value['compose_sha256'] = hashlib.sha256(compose.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                pipeline.validate(directory, REPO)

    def test_reuse_rejects_stale_foreign_or_different_runs(self):
        value = {**identity(), 'verified': True, 'image': IMAGE}
        pull = dict(number=1, merged_at='2026-11-04', head={
            'sha': 'c' * 40, 'repo': {'full_name': REPO}},
            base={'ref': 'main', 'repo': {'full_name': REPO}})
        job = dict(head_repository={'full_name': REPO}, head_sha='c' * 40,
                   conclusion='success', event='pull_request', path=pipeline.WORKFLOW,
                   id=23, run_attempt=1)
        self.assertTrue(pipeline.reusable(value, REPO, TREE, pull, job))
        for altered in ({**job, 'id': 24}, {**job, 'conclusion': 'failure'},
                        {**job, 'head_sha': 'd' * 40}, {**job, 'path': 'foreign.yml'},
                        {**job, 'head_repository': {'full_name': 'other/repo'}}):
            self.assertFalse(pipeline.reusable(value, REPO, TREE, pull, altered))
        self.assertFalse(pipeline.reusable(value, REPO, 'f' * 40, pull, job))
        fork = copy.deepcopy(pull)
        fork['head']['repo']['full_name'] = 'fork/repo'
        self.assertFalse(pipeline.reusable(value, REPO, TREE, fork, job))

    def test_semver_floor_patch_and_retry(self):
        self.assertEqual(pipeline.choose_tag([], COMMIT, '8.0.0'), 'v8.0.0')
        releases = [dict(tag_name='v8.0.3', target_commitish='d' * 40,
                         prerelease=False, draft=False)]
        self.assertEqual(pipeline.choose_tag(releases, COMMIT, '8.0.0'), 'v8.0.4')
        self.assertEqual(pipeline.choose_tag(releases, COMMIT, '8.1.0'), 'v8.1.0')
        self.assertEqual(pipeline.choose_tag(releases, 'd' * 40, '8.0.0'), 'v8.0.3')

    def test_hostinger_asset_is_sealed_without_auth_and_cannot_change_image(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())
            hostinger = Path(directory) / 'compose.hostinger.yaml'
            content = hostinger.read_text()
            self.assertIn(IMAGE, content)
            self.assertNotIn('basicauth', content)
            self.assertNotIn('ACAN_BASIC_AUTH_USERS', content)
            self.assertNotIn('ports:', content)
            self.assertNotIn('build:', content)
            hostinger.write_text(content.replace(IMAGE, 'ghcr.io/other/image:latest'))
            manifest = Path(directory) / 'runtime-manifest.json'
            value = json.loads(manifest.read_text())
            value['hostinger_compose_sha256'] = hashlib.sha256(hostinger.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Hostinger Compose'):
                pipeline.validate(directory, REPO)

    def test_stale_main_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())
            with patch.object(pipeline, 'run', return_value=TREE), patch.object(
                    pipeline, 'api', return_value={'object': {'sha': 'd' * 40}}):
                with self.assertRaisesRegex(ValueError, 'Main advanced'):
                    pipeline.publish(directory, REPO, COMMIT, '8.0.0', True)

    def test_release_uploads_assets_before_publishing_without_build(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())
            calls = []

            def command(*args):
                calls.append(args)
                if args[:2] == ('git', 'rev-parse'):
                    return TREE
                if args[:2] == ('gh', 'api'):
                    return '[[]]'
                return ''

            with patch.object(pipeline, 'run', side_effect=command), patch.object(
                    pipeline, 'api', return_value={'object': {'sha': COMMIT}}):
                self.assertEqual(pipeline.publish(directory, REPO, COMMIT, '8.0.0', True), 'v8.0.0')
            create = next(i for i, call in enumerate(calls) if call[:3] == ('gh', 'release', 'create'))
            upload = next(i for i, call in enumerate(calls) if call[:3] == ('gh', 'release', 'upload'))
            publish = next(i for i, call in enumerate(calls) if call[:3] == ('gh', 'release', 'edit'))
            self.assertLess(create, upload)
            self.assertLess(upload, publish)
            self.assertIn('--draft', calls[create])
            self.assertIn('--latest=true', calls[publish])
            self.assertFalse(any('build' in call for call in calls))

    def test_main_rules_require_pr_and_ci_without_bypasses(self):
        rules = policy(1)
        self.assertEqual(rules['bypass_actors'], [])
        types = {rule['type']: rule for rule in rules['rules']}
        self.assertEqual(types['pull_request']['parameters']['required_approving_review_count'], 1)
        self.assertEqual(types['required_status_checks']['parameters']['required_status_checks'],
                         [{'context': 'CI gate'}])

    def test_retry_keeps_published_digest_even_after_rebuild(self):
        original = f'ghcr.io/{REPO}@sha256:' + '2' * 64
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())

            def command(*args):
                calls.append(args)
                if args[0] == 'git':
                    return TREE
                if args[:2] == ('gh', 'api'):
                    return json.dumps([[dict(tag_name='v8.0.0', target_commitish=COMMIT,
                                             draft=False, prerelease=False)]])
                if args[:3] == ('gh', 'release', 'download'):
                    seal(args[args.index('--dir') + 1], original, identity())
                return ''

            with patch.object(pipeline, 'run', side_effect=command), patch.object(
                    pipeline, 'api', return_value={'object': {'sha': COMMIT}}):
                pipeline.publish(directory, REPO, COMMIT, '8.0.0', True)
            alias = next(call for call in calls if call[0] == 'docker')
            self.assertEqual(alias[-1], original)
            self.assertFalse(any(call[:3] == ('gh', 'release', 'upload') for call in calls))

    def test_conflicting_remote_tag_cannot_be_republished(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())

            def command(*args):
                if args[:2] == ('git', 'rev-parse'):
                    return TREE
                if args[:2] == ('git', 'ls-remote'):
                    return 'd' * 40 + '\trefs/tags/v8.0.0'
                return '[[]]'

            with patch.object(pipeline, 'run', side_effect=command), patch.object(
                    pipeline, 'api', return_value={'object': {'sha': COMMIT}}):
                with self.assertRaisesRegex(ValueError, 'Existing Git tag'):
                    pipeline.publish(directory, REPO, COMMIT, '8.0.0', True)

    def test_audit_and_benchmark_evidence_cannot_be_skipped_or_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            seal(directory, IMAGE, identity())
            path = Path(directory) / 'paper-audit.json'
            audit = json.loads(path.read_text())
            audit['torch_verified'] = False
            path.write_text(json.dumps(audit))
            manifest_path = Path(directory) / 'runtime-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Paper audit'):
                pipeline.validate(directory, REPO)
            seal(directory, IMAGE, identity())
            (Path(directory) / 'paper-benchmarks.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'benchmark archive'):
                pipeline.validate(directory, REPO)


def seal(directory, image, metadata):
    # Synthetic receipt is only used to exercise release-integrity validation.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / 'paper-benchmarks.zip', 'w') as archive:
        archive.writestr('fixture.txt', 'Unit-test fixture, not an executed benchmark.')
    (directory / 'paper-audit.json').write_text(json.dumps(dict(
        contract_version=2, status='passed', conformance='partial', full_paper_conformance=False,
        tree=metadata['tree'], commit=metadata['source_commit'], run_id=metadata['run_id'], source_clean=True,
        torch_verified=True, tests_run=8, failures=0, errors=0, skipped=0,
        benchmarks=dict(status='passed', archive_sha256=hashlib.sha256((directory / 'paper-benchmarks.zip').read_bytes()).hexdigest()))))
    (directory / 'lm-audit.json').write_text(json.dumps(dict(status='passed', skipped=0, tests_run=8,
        profile='acan-lm-causal-prefix-v1', gpu_verified=False, full_paper_conformance=False,
        pretrained_general_quality_verified=False, tree=metadata['tree'], commit=metadata['source_commit'], run_id=metadata['run_id'])))
    (directory / 'generative-runtime.json').write_text(json.dumps(dict(image=f'ghcr.io/{REPO}-generative@sha256:'+'2'*64,
        tree=metadata['tree'], source_commit=metadata['source_commit'], run_id=metadata['run_id'],run_attempt=metadata['run_attempt'],
        audit_sha256=hashlib.sha256((directory / 'lm-audit.json').read_bytes()).hexdigest())))
    return pipeline.seal(directory, image, metadata)
