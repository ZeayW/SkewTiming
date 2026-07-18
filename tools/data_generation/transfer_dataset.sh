#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_DIR:?set SOURCE_DIR}"
: "${DESTINATION_DIR:?set DESTINATION_DIR}"
: "${DESTINATION_HOST:=zywang@projgw}"
: "${DESTINATION_PORT:=2349}"

ssh_options=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/tmp/nua_timer_known_hosts
  -p "$DESTINATION_PORT"
)

ssh "${ssh_options[@]}" "$DESTINATION_HOST" mkdir -p "$DESTINATION_DIR"
rsync -a --partial --checksum --stats \
  -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/nua_timer_known_hosts -p $DESTINATION_PORT" \
  "$SOURCE_DIR/" "$DESTINATION_HOST:$DESTINATION_DIR/"
