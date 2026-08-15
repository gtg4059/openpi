#!/usr/bin/env bash
# Enable host settings that help diagnose unexpected reboots after the fact.
# Safe to re-run. Requires sudo for journald/sysctl changes.
set -euo pipefail

echo "==> Persistent journal (survives reboot)"
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal >/dev/null 2>&1 || true

# Drop-in: keep more history + flush more aggressively (helps hard-reset forensics).
sudo mkdir -p /etc/systemd/journald.conf.d
sudo tee /etc/systemd/journald.conf.d/99-openpi-forensics.conf >/dev/null <<'EOF'
[Journal]
Storage=persistent
SystemMaxUse=1G
RuntimeMaxUse=200M
SyncIntervalSec=30s
EOF
sudo systemctl restart systemd-journald
echo "    journald restarted (Storage=persistent, SyncIntervalSec=30s)"

echo "==> Kernel printk / oops visibility"
# Keep console printk verbosity reasonable; ensure oops are printed.
sudo tee /etc/sysctl.d/99-openpi-forensics.conf >/dev/null <<'EOF'
kernel.panic_on_oops = 0
kernel.softlockup_panic = 0
kernel.print-fatal-signals = 1
EOF
sudo sysctl --system >/dev/null 2>&1 || sudo sysctl -p /etc/sysctl.d/99-openpi-forensics.conf
echo "    sysctl applied"

echo "==> NVIDIA persistence (already often enabled)"
if systemctl list-unit-files | grep -q nvidia-persistenced; then
  sudo systemctl enable --now nvidia-persistenced 2>/dev/null || true
  systemctl is-active nvidia-persistenced || true
fi

echo "==> Done"
echo
echo "During training, also run:"
echo "  uv run python scripts/monitor_train_health.py --interval-sec 5"
echo "  # or: uv run --group rlds scripts/train_pi05_droid_cri.py --monitor --resume ..."
echo
echo "After an unexpected reboot:"
echo "  uv run python scripts/diagnose_reboot.py"
