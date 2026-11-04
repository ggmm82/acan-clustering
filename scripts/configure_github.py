"""Add a focused main ruleset without replacing unrelated repository policy."""
import argparse
import json
import subprocess
import tempfile


def request(repo, path, method='GET', value=None):
    args = ['gh', 'api', f'repos/{repo}/{path}', '--method', method]
    if value is None:
        return json.loads(subprocess.check_output(args, text=True))
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as body:
        json.dump(value, body)
        body.flush()
        return json.loads(subprocess.check_output(args + ['--input', body.name], text=True))


def policy(approvals):
    return dict(name='ACAN controlled main', target='branch', enforcement='active',
                bypass_actors=[],
                conditions={'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                rules=[{'type': 'deletion'}, {'type': 'non_fast_forward'},
                       {'type': 'pull_request', 'parameters': {
                           'required_approving_review_count': approvals,
                           'dismiss_stale_reviews_on_push': True,
                           'require_code_owner_review': False,
                           'require_last_push_approval': approvals > 0,
                           'required_review_thread_resolution': True}},
                       {'type': 'required_status_checks', 'parameters': {
                           'required_status_checks': [{'context': 'CI gate'}],
                           'strict_required_status_checks_policy': True}}])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='ggmm82/acan-clustering')
    parser.add_argument('--approvals', type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args()
    desired = policy(args.approvals)
    existing = request(args.repo, 'rulesets?per_page=100')
    matches = [rule for rule in existing if rule['name'] == desired['name']]
    if matches:
        current = request(args.repo, f"rulesets/{matches[0]['id']}")
        if all(current.get(key) == value for key, value in desired.items()):
            print('Ruleset already configured.')
            return
        raise SystemExit('A ruleset with this name already differs. Review it in GitHub settings; nothing was overwritten.')
    created = request(args.repo, 'rulesets', 'POST', desired)
    print('Created ruleset:', created['id'])
    print('Configure release environment reviewers and branch restrictions in GitHub settings.')


if __name__ == '__main__':
    main()
