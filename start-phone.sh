#!/usr/bin/env bash
# QuantLab for your phone: starts QuantLab so other devices on your Wi-Fi can open it,
# and shows the address to type on your phone. No password: home network only.
exec "$(dirname "$0")/start.sh" --phone "$@"
