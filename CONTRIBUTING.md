# Contributing

Changes use a feature branch and a pull request into `main`. Do not push ordinary
changes directly to `main`. Require the **CI gate** check and a review before
merging; repository settings must enforce this policy (see `docs/RELEASING.md`).

```sh
git switch -c feature/my-change
python -m pip install -e '.[dev,ui]'
python -m unittest discover -s tests -v
docker build -t acan-candidate:local .
git push -u origin feature/my-change
gh pr create --base main
```

Use squash merge to keep a clear history. A release is based on the exact source
tree verified by CI. Resolve conflicts and rerun CI when main changes. Never
manually move a published version tag or replace the contents of a stable release.
After merge and successful CI, a maintainer publishes a stable release using
**Run workflow** on `main`; a push alone never publishes a stable release.

The Docker dependency lock is generated from `pyproject.toml` using:

```sh
uv pip compile pyproject.toml --extra ui --extra docker --python-version 3.12 \
  --generate-hashes --output-file requirements/docker.lock
```

Commit the manifest and lock changes together, and run the container checks.
No GPU backend is included in this CPU image.
