# Hostinger deployment

Use **compose.hostinger.yaml attached to a verified release**, not the source
template. It contains the exact ACAN image digest tested by CI and needs no build
on the VPS. Both prereleases and stable releases attach this file.

## Prerequisites

Use the Hostinger Docker Manager/Traefik setup already used by RigHaven. It must
have the `websecure` entrypoint, `letsencrypt` certificate resolver, and access to
the Compose project's Docker network. This Compose reuses that reverse proxy;
it does not create another Traefik or publish port 4000 on the VPS.

ACAN has no native login. This deployment adds mandatory Traefik BasicAuth in
front of the application. Browser sessions isolate training jobs, not accounts.

## Install

1. Download `compose.hostinger.yaml` and `runtime-manifest.json` from the same
   successful release. Import the Compose into Hostinger Docker Manager, or
   upload it to a dedicated directory on the VPS.
2. Set `ACAN_PROJECT_NAME=acan` and `TRAEFIK_HOST` to the same Hostinger server
   hostname suffix used by RigHaven, for example `srv-123456.hstgr.cloud`.
   With that example the application URL is `https://acan.srv-123456.hstgr.cloud`.
   DNS must route that hostname to the VPS and ports 80/443 must reach Traefik.
3. Generate your login entry interactively on your own machine/VPS:

   ```sh
   docker run --rm -it httpd:2.4-alpine htpasswd -nB acan
   ```

   Put the complete `username:bcrypt-hash` output into `ACAN_BASIC_AUTH_USERS`
   using Hostinger's environment editor. Never put the password into Compose or
   commit a real hash. For a CLI deployment, copy `config/hostinger.env.example`
   to `.env` next to the downloaded Compose and put the hash in **single quotes**
   to preserve dollar signs. Do not double dollar signs in the environment value.
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

Open the HTTPS URL, enter the BasicAuth login, and use Data → Model → Inference.
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
