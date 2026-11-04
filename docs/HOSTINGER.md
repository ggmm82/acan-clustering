# Hostinger deployment

Use **compose.hostinger.yaml attached to a verified release**, not the source
template. It contains the exact ACAN image digest tested by CI and needs no build
on the VPS. Both prereleases and stable releases attach this file.

## Prerequisites

Use the Hostinger Docker Manager/Traefik setup already used by RigHaven. It must
have the `websecure` entrypoint, `letsencrypt` certificate resolver, and access to
the Compose project's Docker network. This Compose reuses that reverse proxy;
it does not create another Traefik or publish port 4000 on the VPS.

This deployment has no authentication. Each page receives a distinct session
identifier; training jobs are keyed by both the browser session and that page
identifier. Dataset, model and prediction stores are local to the page, so two
tabs can train on different datasets independently.

## Install

1. Download `compose.hostinger.yaml` and `runtime-manifest.json` from the same
   successful release. Import the Compose into Hostinger Docker Manager, or
   upload it to a dedicated directory on the VPS.
2. Set `ACAN_PROJECT_NAME=acan` and `TRAEFIK_HOST` to the same Hostinger server
   hostname suffix used by RigHaven, for example `srv-123456.hstgr.cloud`.
   With that example the application URL is `https://acan.srv-123456.hstgr.cloud`.
   DNS must route that hostname to the VPS and ports 80/443 must reach Traefik.
3. Set these values in Hostinger's environment editor, or copy
   `config/hostinger.env.example` to `.env` next to the downloaded Compose for
   CLI deployment. No username, password or authentication hash is required.
4. Make the GHCR package readable by the VPS (public package or an appropriate
   registry login configured on the VPS/Hostinger). A GitHub release being public
   does not automatically make its GHCR package public.
5. Deploy via Docker Manager, or use:

   ```sh
   docker compose -f compose.hostinger.yaml config --quiet
   docker compose -f compose.hostinger.yaml up -d
   docker compose -f compose.hostinger.yaml ps
   docker compose -f compose.hostinger.yaml logs --tail 50
   ```

Open the HTTPS URL and use Data → Model → Inference.
The image healthcheck calls the internal `/healthz` endpoint. The browser session
cookie is marked Secure in this deployment, so use HTTPS.

## Operations

- The default limit is 1 GiB RAM and two CPU cores, with a 256 MiB temporary
  filesystem. Start with small datasets. Increase resource limits only after
  checking the VPS capacity alongside RigHaven and other applications.
- No Docker socket, privileged container, Redis or separate database is needed
  by ACAN. A single Gunicorn worker owns the in-memory training jobs.
- Export model JSON and results before upgrades. Jobs, browser-held data and
  in-memory models are not a persistent model registry; restarting the container
  does not resume training. The current application has no persistent model/data
  directory to put on a named volume.
- Upgrade by deploying the sealed Compose of a newer release while preserving
  the project name and environment values. Roll back with the older release's
  Compose. Do not edit its image to a mutable `latest` tag.
- If Traefik is not connected to the application's project network, connect it
  using the same Hostinger network arrangement as RigHaven. A custom shared
  network must also be explicitly attached to both services.

CI checks the Compose interpolation, sealed digest and checksum. The actual VPS,
DNS, certificate issuance and Hostinger routing must be validated after deployment;
repository CI does not have access to that infrastructure.


For the optional Generative tab, retain the normal frontend deployment and
configure a durable worker. For the bundled CPU worker, deploy the single sealed
`compose.generative.hostinger.yaml` instead of the normal frontend-only Compose.
It includes both images and the persistent worker volume, with no Docker profile.
Set the private `ACAN_LM_SERVICE_TOKEN` environment variable in Hostinger and
retain the same project name so the existing data volume is reused. Download
the raw YAML asset; rich-text copy/paste can introduce invalid non-breaking
spaces in the indentation. No user authentication is added;
for long training connect a private HTTPS GPU worker using `ACAN_LM_URL` and
`ACAN_LM_SERVICE_TOKEN`. The frontend does not store optimizer state. Keep the
worker's named data volume and exact image pinned, and do not remove volumes
on deploy. See [Generative setup and recovery](GENERATIVE.md).
