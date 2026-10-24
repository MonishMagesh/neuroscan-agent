#!/usr/bin/env bash
# Run from inside the cloned repo:  bash deploy/setup_vm.sh
# Touches only: swap file, Python 3.12 (alongside system Python), this app's venv, and one new systemd unit.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYV="${PYTHON:-python3.12}"

echo "[1/5] Swap (safety net for a 1 GB VM)"
if ! swapon --show | grep -q .; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "    swap already present, skipping"
fi

echo "[2/5] Python 3.12 (installed alongside, system python3 is not changed)"
if ! command -v "$PYV" >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y python3.12 python3.12-venv
fi

echo "[3/5] Virtualenv + dependencies (takes a few minutes)"
"$PYV" -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --no-cache-dir -q -U pip
"$DIR/venv/bin/pip" install --no-cache-dir -q -r "$DIR/requirements-deploy.txt"
[ -f "$DIR/.env" ] || cp "$DIR/deploy.env.example" "$DIR/.env"

echo "[4/5] systemd service on 127.0.0.1:8766"
sed -e "s|__DIR__|$DIR|g" -e "s|__USER__|$(id -un)|g" "$DIR/deploy/neuroscan.service" \
  | sudo tee /etc/systemd/system/neuroscan.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now neuroscan
sleep 5

echo "[5/5] Health check"
curl -fsS http://127.0.0.1:8766/health && echo
echo "Done. Next: bash deploy/add_caddy_site.sh YOURNAME.duckdns.org"
