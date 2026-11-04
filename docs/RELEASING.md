# Controlled Docker releases

This pipeline adapts the verified-runtime approach in
[ggmm82/righaven](https://github.com/ggmm82/righaven): build a candidate, verify the
real runtime, seal its immutable digest, and promote that same image.

## Lifecycle

1. Open a PR into `main`. Python 3.10, 3.11 and 3.12 tests and package builds run.
   A dedicated CPU PyTorch job audits numerical gradients/backend parity and
   reruns the historical synthetic benchmark suite. See [Paper audit](PAPER-AUDIT.md).
2. Build the CPU UI container using a digest-pinned official Python base and a
   hash-verified dependency lock. Test its installed package as a non-root user.
3. Start the image with a read-only filesystem and exercise model reload,
   prediction and completion through the actual Dash HTTP callbacks.
4. **CI gate** succeeds only if resolution, all tests and Docker checks succeed.
5. Internal PRs publish a candidate to GHCR and a prerelease named
   `v8.1.0-rc.pr.NUMBER.RUN.ATTEMPT`. Fork PRs run the same tests and local image
   checks without logging into GHCR, pushing images or publishing releases.
6. After merge, CI looks for the successful PR's `verified-runtime` artifact.
   Reuse requires the exact same Git tree, current PR head, repository, workflow,
   run identity and immutable image digest. The reused image is tested again,
   without rebuilding. If the artifact expired, the tree differs, or there is no
   eligible PR artifact, main builds and tests a new candidate.
7. Stable publication requires a maintainer to select **Run workflow** on `main`
   in Actions after reviewing the merged changes. Pushes to main verify the
   candidate but do not automatically publish a stable release. The `release`
   environment can require an additional reviewer when configured. The publisher checks that
   the commit is still main's head, creates a draft, attaches both deployment
   assets, and only then publishes. Releases are serialized. A stale main run
   is refused. PR previews never become the latest stable release.

`pyproject.toml` supplies the minimum stable version (initially `8.0.0`). The
first release uses that floor; following releases increment patch, unless the
floor is raised for a minor/major release. Retrying a commit keeps its version.
The Python distribution retains the version declared in `pyproject.toml`;
the GitHub deployment release has its own incrementing version. Both identities
and the image source commit are available in the manifest/release notes.

## Deploy or roll back

Download `compose.release.yaml` and `runtime-manifest.json` from one verified
release. The Compose is JSON (a valid YAML subset) and pins a GHCR digest.

```sh
docker compose -f compose.release.yaml up -d
docker compose -f compose.release.yaml ps
```

Open the application on port 4000 of the local machine. The default binding is
loopback; use an authenticated TLS reverse proxy for shared hosting. To roll
back, run the Compose from the previous release. Export model JSON and prediction
CSV before replacing a container: training jobs live in memory and do not survive
restart. Run one server process, as required by the current session job registry.

Hostinger uses the additional **compose.hostinger.yaml** asset, with the same
sealed image digest and Traefik HTTPS routing, without authentication. See
[Hostinger deployment](HOSTINGER.md) for setup and environment variables.

The GHCR `latest` alias is a convenience for stable images; sealed Compose files
always use the tested digest. GHCR packages can initially be private: grant read
access or change package visibility before expecting anonymous pulls.

Releases also attach `paper-audit.json` and `paper-benchmarks.zip`. The runtime
manifest binds their checksums and rejects missing, skipped, failed or mismatched
evidence. These receipts explicitly record **partial** paper correspondence and
synthetic benchmark provenance; they are not a full scientific certification.

For a local source build use `docker compose up --build -d`. The production
command uses Gunicorn with one worker and threads, not Dash's development server.

## Required GitHub settings

Workflow files cannot enforce repository policy by themselves. A maintainer must
apply these settings before calling the process protected:

- Protect `main` with required PRs, one approval, dismissal of stale approvals,
  resolved conversations, **CI gate**, and an up-to-date branch. Block force
  pushes/deletion and enforce rules for administrators too. A sole maintainer
  needs another reviewer; choose zero required approvals explicitly if that is
  your intended governance policy.
- Optionally create a `release` environment with required reviewers and prevent self-review.
  Restrict deployment branches to `main`. The workflow's environment declaration
  alone does not create required reviewers. Manual workflow dispatch remains
  mandatory even without an environment reviewer. Leave `preview` without a manual
  gate if automatic internal-PR prereleases are desired.
- Keep Actions enabled and permit the workflow's scoped `contents: write` and
  `packages: write` permissions. No personal token is required in Actions;
  `GITHUB_TOKEN` is used. Fork PRs never receive publication privileges.
- Set the package visibility/read permissions appropriate for deployment users.

`python scripts/configure_github.py` adds an active main ruleset using the API
without replacing existing rulesets. Use `--approvals 0` only when deliberately
allowing single-maintainer merges. It leaves environment reviewers to the UI,
where the intended people can be selected without guessing their identities.

The first pipeline change must be bootstrapped by merging its reviewed PR.
Existing workflows on main cannot provide checks that do not exist there yet.
No initial release or tag is created by merely adding these files.

## Managed cloud builds

Use the inherited Docker proxy configuration. If the build needs the managed
session CA, mount the combined trust bundle without embedding it in the image:

```sh
docker build --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -t acan-candidate:local .
```

API access to `api.github.com` is needed for creating PRs, rulesets, releases and
resolving prior Actions artifacts. Git clone/push access alone is not API access.


The generative extension uses runtime schema 4. In addition to the web image,
`generative-runtime.json` records the tested CPU worker digest and source/run;
`lm-audit.json` records the causal/resume contracts. The optional sealed
`compose.generative.hostinger.yaml` standalone stack pins both the web image and
that worker, without optional profiles. Hostinger validates this file by itself. Missing, stale,
skipped or altered LM evidence blocks release. Neither the overlay nor opening
the GUI starts training. CUDA builds/hardware are explicitly separate from the
CPU mechanics certification; see [GENERATIVE.md](GENERATIVE.md).
