"""Build once, seal a digest, reuse a verified PR tree and publish releases.

Inspired by ggmm82/righaven's verified-runtime flow, adapted to one ACAN image.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

WORKFLOW = '.github/workflows/ci.yml'
ASSETS = ('runtime-manifest.json', 'compose.release.yaml', 'compose.hostinger.yaml', 'paper-audit.json', 'paper-benchmarks.zip', 'generative-runtime.json', 'lm-audit.json', 'compose.generative.hostinger.yaml')


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def api(repo, path):
    return json.loads(run('gh', 'api', f'repos/{repo}/{path}'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def compose(image):
    return {'services': {'acan': {
        'image': image, 'ports': ['127.0.0.1:4000:4000'], 'init': True,
        'restart': 'unless-stopped', 'read_only': True,
        'tmpfs': ['/tmp:size=256m,mode=1777'],
        'security_opt': ['no-new-privileges:true'], 'cap_drop': ['ALL'],
    }}}


def hostinger_compose(image):
    template = Path(__file__).resolve().parents[1] / 'config/compose.hostinger.template.yaml'
    return template.read_text().replace('__ACAN_IMAGE__', image)


def generative_compose(web_image, worker_image):
    template = Path(__file__).resolve().parents[1] / 'config/compose.generative.hostinger.template.yaml'
    return template.read_text().replace('__ACAN_IMAGE__', web_image).replace('__ACAN_LM_IMAGE__', worker_image)


def validate(directory, repo, tree=None):
    directory = Path(directory)
    value = json.loads((directory / 'runtime-manifest.json').read_text())
    if (value.get('schema') != 4 or value.get('repository') != repo
            or value.get('workflow') != WORKFLOW or value.get('verified') is not True
            or value.get('head_repository') != repo
            or value.get('event') not in ('pull_request', 'push', 'workflow_dispatch')):
        raise ValueError('Invalid candidate provenance')
    for name in ('source_commit', 'tree', 'head_sha'):
        if not re.fullmatch(r'[0-9a-f]{40}', str(value.get(name, ''))):
            raise ValueError('Invalid source identity')
    for name in ('run_id', 'run_attempt'):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError('Invalid workflow run identity')
    if value['event'] == 'pull_request' and (
            type(value.get('pull_request')) is not int or value['pull_request'] < 1):
        raise ValueError('Invalid pull request identity')
    if tree is not None and value['tree'] != tree:
        raise ValueError('Candidate tree differs from release tree')
    prefix = 'ghcr.io/' + repo.lower() + '@sha256:'
    if not re.fullmatch(re.escape(prefix) + r'[0-9a-f]{64}', value.get('image', '')):
        raise ValueError('Expected an immutable image from this repository')
    content = (directory / 'compose.release.yaml').read_bytes()
    if hashlib.sha256(content).hexdigest() != value.get('compose_sha256'):
        raise ValueError('Compose checksum mismatch')
    if json.loads(content) != compose(value['image']):
        raise ValueError('Compose differs from the sealed runtime')
    hostinger = (directory / 'compose.hostinger.yaml').read_bytes()
    if hashlib.sha256(hostinger).hexdigest() != value.get('hostinger_compose_sha256'):
        raise ValueError('Hostinger Compose checksum mismatch')
    if hostinger.decode() != hostinger_compose(value['image']):
        raise ValueError('Hostinger Compose differs from the sealed runtime')
    audit_bytes = (directory / 'paper-audit.json').read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != value.get('audit_sha256'):
        raise ValueError('Paper audit checksum mismatch')
    audit = json.loads(audit_bytes)
    if (audit.get('contract_version') != 2 or audit.get('status') != 'passed'
            or audit.get('tree') != value['tree'] or audit.get('commit') != value['source_commit']
            or audit.get('run_id') != value['run_id']
            or audit.get('source_clean') is not True or audit.get('torch_verified') is not True or audit.get('shared_algebra_verified') is not True
            or audit.get('tests_run', 0) < 18 or any(audit.get(k) != 0 for k in ('failures', 'errors', 'skipped'))
            or audit.get('conformance') != 'partial' or audit.get('full_paper_conformance') is not False):
        raise ValueError('Paper audit is missing, failed, skipped, dirty or from another source/run')
    for filename in ('generative-runtime.json', 'lm-audit.json', 'compose.generative.hostinger.yaml'):
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != value.get('generative_sha256', {}).get(filename):
            raise ValueError('Generative deployment evidence checksum mismatch')
    generative = json.loads((directory / 'generative-runtime.json').read_text())
    lm_audit = json.loads((directory / 'lm-audit.json').read_text())
    worker_prefix = 'ghcr.io/' + repo.lower() + '-generative@sha256:'
    if (not re.fullmatch(re.escape(worker_prefix) + r'[0-9a-f]{64}', generative.get('image', ''))
            or generative.get('image') != value.get('generative_image')
            or any(generative.get(k) != value[k] for k in ('tree', 'source_commit', 'run_id', 'run_attempt'))
            or generative.get('audit_sha256') != hashlib.sha256((directory / 'lm-audit.json').read_bytes()).hexdigest()
            or lm_audit.get('status') != 'passed' or lm_audit.get('skipped') != 0 or lm_audit.get('tests_run',0) < 36
            or lm_audit.get('controlled_protocol') != 'controlled-v1' or lm_audit.get('controls_verified') is not True
            or lm_audit.get('profile') != 'acan-lm-causal-prefix-v1'
            or lm_audit.get('gpu_verified') is not False or lm_audit.get('full_paper_conformance') is not False
            or lm_audit.get('pretrained_general_quality_verified') is not False
            or lm_audit.get('tree') != value['tree'] or lm_audit.get('commit') != value['source_commit']
            or lm_audit.get('run_id') != value['run_id']
            or (directory / 'compose.generative.hostinger.yaml').read_text() != generative_compose(value['image'], generative['image'])):
        raise ValueError('Generative worker or LM audit is missing, incompatible, stale or altered')
    benchmark = audit.get('benchmarks', {})
    if (benchmark.get('status') != 'passed' or
            hashlib.sha256((directory / 'paper-benchmarks.zip').read_bytes()).hexdigest() != benchmark.get('archive_sha256')):
        raise ValueError('Paper benchmark archive is missing, failed or altered')
    return value


def metadata():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    pr = event.get('pull_request')
    commit = run('git', 'rev-parse', 'HEAD')
    return dict(schema=4, repository=os.environ['GITHUB_REPOSITORY'], workflow=WORKFLOW,
                event=os.environ['GITHUB_EVENT_NAME'], source_commit=commit,
                tree=run('git', 'rev-parse', 'HEAD^{tree}'),
                head_sha=pr['head']['sha'] if pr else commit,
                head_repository=pr['head']['repo']['full_name'] if pr else os.environ['GITHUB_REPOSITORY'],
                pull_request=event['number'] if pr else None,
                run_id=int(os.environ['GITHUB_RUN_ID']),
                run_attempt=int(os.environ['GITHUB_RUN_ATTEMPT']))


def seal(directory, image, identity):
    directory = Path(directory)
    save(directory / 'compose.release.yaml', compose(image))
    (directory / 'compose.hostinger.yaml').write_text(hostinger_compose(image))
    generative = json.loads((directory / 'generative-runtime.json').read_text())
    (directory / 'compose.generative.hostinger.yaml').write_text(generative_compose(image, generative['image']))
    content = (directory / 'compose.release.yaml').read_bytes()
    save(directory / 'runtime-manifest.json', {
        **identity, 'image': image, 'verified': True, 'generative_image': generative['image'],
        'generative_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('generative-runtime.json','lm-audit.json','compose.generative.hostinger.yaml')},
        'compose_sha256': hashlib.sha256(content).hexdigest(),
        'hostinger_compose_sha256': hashlib.sha256((directory / 'compose.hostinger.yaml').read_bytes()).hexdigest(),
        'audit_sha256': hashlib.sha256((directory / 'paper-audit.json').read_bytes()).hexdigest(),
    })
    return validate(directory, identity['repository'], identity['tree'])


def reusable(value, repo, tree, pull, job):
    return (value['repository'] == repo and value['tree'] == tree
            and value['event'] == 'pull_request'
            and value['pull_request'] == pull['number']
            and pull['merged_at'] is not None
            and pull['base']['ref'] == 'main'
            and pull['base']['repo']['full_name'] == repo
            and pull['head']['repo']['full_name'] == repo
            and value['head_sha'] == pull['head']['sha']
            and job['head_repository']['full_name'] == repo
            and job['head_sha'] == pull['head']['sha']
            and job['conclusion'] == 'success' and job['event'] == 'pull_request'
            and job['path'].split('@')[0] == WORKFLOW
            and job['id'] == value['run_id']
            and value['run_attempt'] <= job['run_attempt'])


def resolve(repo, commit, directory):
    tree = run('git', 'rev-parse', commit + '^{tree}')
    pulls = api(repo, f'commits/{commit}/pulls?per_page=100')
    pulls = [p for p in pulls if p.get('merged_at') and p.get('merge_commit_sha') == commit]
    if not pulls:
        return False
    jobs = api(repo, 'actions/workflows/ci.yml/runs?event=pull_request&status=completed&per_page=100')['workflow_runs']
    for pull in pulls:
        for job in jobs:
            if job.get('conclusion') != 'success' or job.get('head_sha') != pull['head']['sha']:
                continue
            with tempfile.TemporaryDirectory() as temp:
                try:
                    run('gh', 'run', 'download', str(job['id']), '--repo', repo,
                        '--name', 'verified-runtime', '--dir', temp)
                    value = validate(temp, repo, tree)
                    if not reusable(value, repo, tree, pull, job):
                        continue
                except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError):
                    continue  # Missing/expired/nonmatching artifacts require a new build.
                for name in ASSETS:
                    target = Path(directory) / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((Path(temp) / name).read_bytes())
                return True
    return False


def version_tuple(tag):
    match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', tag)
    if not match:
        raise ValueError('Expected a stable semantic version')
    return tuple(map(int, match.groups()))


def choose_tag(releases, commit, floor):
    minimum = version_tuple(floor)
    versions = []
    for release in releases:
        if release.get('prerelease'):
            continue
        try:
            version = version_tuple(release['tag_name'])
        except ValueError:
            continue
        if release.get('target_commitish') == commit:
            return release['tag_name']  # Retry preserves the assigned version.
        versions.append(version)
    latest = max(versions, default=None)
    next_version = minimum if latest is None or latest < minimum else (*latest[:2], latest[2] + 1)
    return 'v' + '.'.join(map(str, next_version))


def publish(directory, repo, commit, floor, main):
    tree = run('git', 'rev-parse', commit + '^{tree}')
    value = validate(directory, repo, tree)
    if main:
        if api(repo, 'git/ref/heads/main')['object']['sha'] != commit:
            raise ValueError('Main advanced; refusing to publish a stale stable release')
    else:
        if value['event'] != 'pull_request':
            raise ValueError('Only internal PR candidates may become previews')
        pull = api(repo, f"pulls/{value['pull_request']}")
        if (pull['state'] != 'open' or pull['head']['sha'] != value['head_sha']
                or pull['head']['repo']['full_name'] != repo):
            raise ValueError('Pull request candidate is stale or foreign')
    releases = json.loads(run('gh', 'api', '--paginate', '--slurp',
                             f'repos/{repo}/releases?per_page=100'))
    releases = [release for page in releases for release in page]
    tag = choose_tag(releases, commit, floor) if main else (
        f"v{floor}-rc.pr.{value['pull_request']}.{value['run_id']}.{value['run_attempt']}")
    # Freeze the bundle first, then expose a release only after both assets exist.
    notes = (f"Verified ACAN {'stable release' if main else 'PR candidate'} {tag}\n\n"
             f"Release commit: {commit}\nImage source commit: {value['source_commit']}\n"
             f"Source tree: {value['tree']}\nImage: {value['image']}\n\n"
             "Download compose.release.yaml and runtime-manifest.json. "
             "Run docker compose -f compose.release.yaml up -d. "
             "For Hostinger with the Generative laboratory use the standalone compose.generative.hostinger.yaml "
             "(frontend plus persistent CPU worker), configure TRAEFIK_HOST and ACAN_LM_SERVICE_TOKEN, "
             "and preserve the existing project name and data volume. For frontend-only deployment use compose.hostinger.yaml. "
             "Follow docs/HOSTINGER.md. "
             "paper-audit.json records passing software contracts and outstanding paper/code differences; "
             "it does not certify full paper conformance. "
             "The Compose pins the exact tested image digest; no rebuild is performed.\n")
    notes_path = Path(directory) / 'release-notes.md'
    notes_path.write_text(notes)
    existing = next((r for r in releases if r['tag_name'] == tag), None)
    def aliases(image):
        run('docker', 'buildx', 'imagetools', 'create',
            '--tag', f'ghcr.io/{repo.lower()}:{tag}',
            '--tag', f'ghcr.io/{repo.lower()}:latest', image)

    if existing and not existing['draft']:
        if existing['target_commitish'] != commit:
            raise ValueError('Release tag already belongs to another commit')
        if main:
            with tempfile.TemporaryDirectory() as published:
                run('gh', 'release', 'download', tag, '--repo', repo, '--dir', published,
                    '--pattern', 'runtime-manifest.json', '--pattern', 'compose.release.yaml',
                    '--pattern', 'compose.hostinger.yaml', '--pattern', 'paper-audit.json',
                    '--pattern', 'paper-benchmarks.zip', '--pattern', 'generative-runtime.json',
                    '--pattern', 'lm-audit.json', '--pattern', 'compose.generative.hostinger.yaml')
                published_value = validate(published, repo, tree)
                aliases(published_value['image'])
                aliases(published_value['generative_image'])
        print('Release already published:', tag)
        return tag
    remote = run('git', 'ls-remote', '--tags', 'origin',
                 f'refs/tags/{tag}', f'refs/tags/{tag}^{{}}')
    if remote:
        refs = dict(line.split()[::-1] for line in remote.splitlines())
        target = refs.get(f'refs/tags/{tag}^{{}}', refs.get(f'refs/tags/{tag}'))
        if target != commit:
            raise ValueError('Existing Git tag points to another commit')
    if not existing:
        args = ['gh', 'release', 'create', tag, '--repo', repo, '--target', commit,
                '--draft', '--title', tag, '--notes-file', str(notes_path)]
        if not main:
            args += ['--prerelease']
        run(*args)
    run('gh', 'release', 'upload', tag, '--repo', repo, '--clobber',
        *(str(Path(directory) / name) for name in ASSETS))
    run('gh', 'release', 'edit', tag, '--repo', repo, '--draft=false',
        '--latest=' + ('true' if main else 'false'))
    if main:
        # latest is only a convenience alias. Deployments use the sealed digest.
        aliases(value['image'])
        aliases(value['generative_image'])
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['resolve', 'seal', 'validate', 'publish'])
    parser.add_argument('--directory', default='reports/runtime')
    parser.add_argument('--image')
    args = parser.parse_args()
    repo = os.environ['GITHUB_REPOSITORY']
    commit = run('git', 'rev-parse', 'HEAD')
    if args.command == 'resolve':
        reused = resolve(repo, commit, args.directory)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'reuse={str(reused).lower()}\n')
    elif args.command == 'seal':
        seal(args.directory, args.image, metadata())
    elif args.command == 'validate':
        print(validate(args.directory, repo, run('git', 'rev-parse', 'HEAD^{tree}'))['image'])
    else:
        # Read through stdlib TOML on the Python 3.12 release runner.
        import tomllib
        floor = tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
        publish(args.directory, repo, commit, floor, main=os.environ['GITHUB_EVENT_NAME'] != 'pull_request')


if __name__ == '__main__':
    main()
