#!/bin/bash

set -e

echo "=================================="
echo "OMGrab SDK Uninstallation"
echo "=================================="
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

SERVICE_FILE="omgrab.service"
INSTALL_DIR="/opt/omgrab"

echo "Step 1: Stopping and disabling service..."
systemctl stop "$SERVICE_FILE" 2>/dev/null || true
systemctl disable "$SERVICE_FILE" 2>/dev/null || true

echo "Step 2: Removing systemd service files..."
rm -f "/etc/systemd/system/$SERVICE_FILE"
systemctl stop early-led.service 2>/dev/null || true
systemctl disable early-led.service 2>/dev/null || true
rm -f "/etc/systemd/system/early-led.service"
# Clean up legacy service from older installs
rm -f "/etc/systemd/system/pi-agent.service"
systemctl daemon-reload

echo "Step 3: Stopping and removing Docker containers..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    docker rm -f omgrab-runtime wifi-connect 2>/dev/null || true
    docker compose down 2>/dev/null || true
fi

echo "Step 4: Removing Docker image..."
docker rmi omgrab-runtime:latest 2>/dev/null || true

echo "Step 5: Removing Samba share..."
rm -f /etc/samba/smb.conf.d/omgrab.conf
if command -v smbd &> /dev/null; then
    systemctl restart smbd 2>/dev/null || true
fi

echo ""
echo "Do you want to remove the installation directory? (y/n)"
echo "WARNING: This will delete all data in $INSTALL_DIR (recordings, config)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Removing installation directory..."
    rm -rf "$INSTALL_DIR"
    echo "Installation directory removed."
else
    echo "Keeping installation directory: $INSTALL_DIR"
    echo "Note: Data preserved in $INSTALL_DIR/data"
fi

echo ""
echo "=================================="
echo "Uninstallation Complete!"
echo "=================================="
