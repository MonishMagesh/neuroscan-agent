#!/usr/bin/env bash
# Adds ONE new site block to the existing Caddyfile. Existing blocks are never edited.
# Backs up first, validates, reloads gracefully, and restores the backup if validation fails.
set -euo pipefail
HOST="${1:?usage: bash deploy/add_caddy_site.sh yourname.duckdns.org}"
CF=/etc/caddy/Caddyfile
BAK="$CF.bak.$(date +%s)"

if grep -qF "$HOST" "$CF"; then echo "$HOST is already in $CF, nothing to do."; exit 0; fi
sudo cp "$CF" "$BAK"
sudo tee -a "$CF" >/dev/null <<BLOCK

$HOST {
    reverse_proxy 127.0.0.1:8766
}
BLOCK

if sudo caddy validate --config "$CF" --adapter caddyfile >/dev/null 2>&1; then
  sudo systemctl reload caddy
  echo "Caddy reloaded. Open https://$HOST (the first load can take ~30s while the certificate is issued)."
else
  echo "New config failed validation. Restoring the backup. Nothing was changed."
  sudo cp "$BAK" "$CF"
  exit 1
fi
