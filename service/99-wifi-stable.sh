#!/bin/bash
# NetworkManager dispatcher script: override bgscan on every WiFi connection.
#
# NM computes aggressive bgscan parameters (e.g. "simple:30:-65:300") for
# some connection types.  Combined with broken CQM signal monitoring on the
# BCM43455 (brcmfmac), this triggers blind 30 s scans that cause a constant
# roaming / disconnect loop between 2.4 GHz and 5 GHz bands.
#
# This script overrides bgscan to the conservative values used by the
# known-good fleet: -70 dBm threshold, 86400 s long interval (effectively
# never triggers a background roam).
#
# WiFi power-save is handled separately via the NM config file
# /etc/NetworkManager/conf.d/wifi-stable.conf (wifi.powersave=2).
#
# Installed to /etc/NetworkManager/dispatcher.d/99-wifi-stable by install.sh.

IFACE="$1"
ACTION="$2"

[ "$IFACE" = 'wlan0' ] || exit 0
[ "$ACTION" = 'up' ] || exit 0

/usr/sbin/wpa_cli -i wlan0 set_network 0 bgscan '"simple:30:-70:86400"' 2>/dev/null

exit 0
