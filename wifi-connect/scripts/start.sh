#!/usr/bin/env bash
# Modified by OMGrab from the original balena-os/wifi-connect start.sh:
# - Rewritten for on-demand captive portal (triggered by button long-press)
# - Added iptables rules for Android captive portal detection
# - Added DNS redirect via NetworkManager dnsmasq
# - Added WiFi disconnect before hotspot creation

export DBUS_SYSTEM_BUS_ADDRESS=unix:path=/host/run/dbus/system_bus_socket

GATEWAY="192.168.42.1"
IFACE="wlan0"

# Ensure NetworkManager's dnsmasq redirects DNS for captive portal detection
CAPTIVE_PORTAL_CONF="/etc/NetworkManager/dnsmasq-shared.d/captive-portal.conf"
if [ ! -f "$CAPTIVE_PORTAL_CONF" ]; then
    echo "Setting up captive portal DNS redirect..."
    mkdir -p /etc/NetworkManager/dnsmasq-shared.d
    echo "address=/#/$GATEWAY" > "$CAPTIVE_PORTAL_CONF"
    echo "Captive portal config created at $CAPTIVE_PORTAL_CONF"
fi

# --- iptables rules for Android captive portal detection ---
# Android uses both HTTP and HTTPS probes to detect captive portals.
# Without these rules, the HTTPS probe to port 443 gets "connection refused"
# which causes many Android versions to skip the portal popup entirely.
# We also force all DNS through our local dnsmasq to handle devices that
# cache DNS or use DNS-over-TLS (Private DNS, default on Android 9+).
setup_iptables() {
    echo "Setting up iptables captive portal rules..."

    # Flush any stale rules from a previous run
    cleanup_iptables 2>/dev/null

    # Force all DNS through local dnsmasq (handles cached DNS, DoT fallback)
    iptables -t nat -A PREROUTING -i "$IFACE" -p udp --dport 53 -j DNAT --to-destination "$GATEWAY"
    iptables -t nat -A PREROUTING -i "$IFACE" -p tcp --dport 53 -j DNAT --to-destination "$GATEWAY"

    # Redirect all HTTP to the captive portal server
    iptables -t nat -A PREROUTING -i "$IFACE" -p tcp --dport 80 -j DNAT --to-destination "$GATEWAY:80"

    # Redirect all HTTPS to the captive portal server on port 80.
    # The TLS handshake will fail (server speaks HTTP), but Android interprets
    # "TCP connected then TLS failed" as traffic interception, which together
    # with the HTTP 302 from the HTTP probe triggers the portal popup.
    iptables -t nat -A PREROUTING -i "$IFACE" -p tcp --dport 443 -j DNAT --to-destination "$GATEWAY:80"

    echo "iptables captive portal rules applied"
}

cleanup_iptables() {
    iptables -t nat -D PREROUTING -i "$IFACE" -p udp --dport 53 -j DNAT --to-destination "$GATEWAY" 2>/dev/null
    iptables -t nat -D PREROUTING -i "$IFACE" -p tcp --dport 53 -j DNAT --to-destination "$GATEWAY" 2>/dev/null
    iptables -t nat -D PREROUTING -i "$IFACE" -p tcp --dport 80 -j DNAT --to-destination "$GATEWAY:80" 2>/dev/null
    iptables -t nat -D PREROUTING -i "$IFACE" -p tcp --dport 443 -j DNAT --to-destination "$GATEWAY:80" 2>/dev/null
}

# Clean up iptables rules on exit (normal exit, error, or signal)
trap cleanup_iptables EXIT

# Disconnect any active WiFi connections first to speed up hotspot creation
# This saves time since NetworkManager doesn't have to disconnect during create_hotspot
echo "Disconnecting any active WiFi connections..."
dbus-send --system --print-reply --dest=org.freedesktop.NetworkManager \
    /org/freedesktop/NetworkManager \
    org.freedesktop.NetworkManager.DeactivateConnection \
    objpath:/org/freedesktop/NetworkManager/ActiveConnection/1 2>/dev/null || true

# Alternative: deactivate all wireless connections via device
WIFI_DEVICE=$(dbus-send --system --print-reply --dest=org.freedesktop.NetworkManager \
    /org/freedesktop/NetworkManager \
    org.freedesktop.NetworkManager.GetDeviceByIpIface \
    string:wlan0 2>/dev/null | grep "object path" | cut -d'"' -f2)

if [ -n "$WIFI_DEVICE" ]; then
    echo "Disconnecting WiFi device: $WIFI_DEVICE"
    dbus-send --system --print-reply --dest=org.freedesktop.NetworkManager \
        "$WIFI_DEVICE" \
        org.freedesktop.NetworkManager.Device.Disconnect 2>/dev/null || true
fi

# Small delay to ensure disconnect completes
sleep 0.5

# Set up iptables rules before starting the portal
setup_iptables

# Always start the portal - this script is only called on-demand via long-press
printf 'Starting WiFi Connect portal for configuration\n'
./wifi-connect
printf 'WiFi configuration complete. Exiting.\n'
exit 0
