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


def validate(directory, repo, tree=None):
    directory = Path(directory)
    value = json.loads((directory / 'runtime-manifest.json').read_text())
    if (value.get('schema') != 1 or value.get('repository') != repo
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
    return value


def metadata():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    pr = event.get('pull_request')
    commit = run('git', 'rev-parse', 'HEAD')
    return dict(schema=1, repository=os.environ['GITHUB_REPOSITORY'], workflow=WORKFLOW,
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
    content = (directory / 'compose.release.yaml').read_bytes()
    save(directory / 'runtime-manifest.json', {
        **identity, 'image': image, 'verified': True,
        'compose_sha256': hashlib.sha256(content).hexdigest(),
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
                for name in ('runtime-manifest.json', 'compose.release.yaml'):
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
                    '--pattern', 'runtime-manifest.json', '--pattern', 'compose.release.yaml')
                aliases(validate(published, repo, tree)['image'])
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
        str(Path(directory) / 'runtime-manifest.json'),
        str(Path(directory) / 'compose.release.yaml'))
    run('gh', 'release', 'edit', tag, '--repo', repo, '--draft=false',
        '--latest=' + ('true' if main else 'false'))
    if main:
        # latest is only a convenience alias. Deployments use the sealed digest.
        aliases(value['image'])
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
