#!/usr/bin/env bash
#
# Hardware (BCM numbering):
#   Green LED:  GPIO 17  (physical pin 11)
#   Red LED:    GPIO 23  (physical pin 16)
#   Buzzer:     GPIO 12  (physical pin 32)
#   Button:     GPIO 24  (physical pin 18) with pull-up
#   OLED (SSD1306): I2C bus 1, address 0x3c
#       SDA: GPIO 2 (pin 3)  SCL: GPIO 3 (pin 5)
#       VCC: 3.3V (pin 17)   GND: pin 25

set -euo pipefail

GREEN_LED=17
RED_LED=23
BUZZER=12
BUTTON=24

pass_count=0
fail_count=0

pass() { echo "  PASS: $1"; pass_count=$((pass_count + 1)); }
fail() { echo "  FAIL: $1"; fail_count=$((fail_count + 1)); }

cleanup() {
    echo ""
    echo "Cleaning up GPIO..."
    pinctrl set $GREEN_LED ip pn 2>/dev/null || true
    pinctrl set $RED_LED ip pn 2>/dev/null || true
    pinctrl set $BUZZER ip pn 2>/dev/null || true
    pinctrl set $BUTTON ip pn 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo ""
echo "=== GPIO Quick Test (Raspberry Pi 5) ==="
echo ""

# ── Root check ──────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    echo "This script needs root for direct GPIO access."
    echo "Re-run with:  sudo $0"
    exit 1
fi

# ── Pre-check: required tools ──────────────────────────────────────
echo "[Pre-check] Required tools"
if command -v pinctrl &>/dev/null; then
    pass "pinctrl found ($(pinctrl -v 2>/dev/null || echo 'version unknown'))"
else
    fail "pinctrl not found"
    echo "  pinctrl should be pre-installed on Raspberry Pi OS Bookworm."
    echo "  If missing: sudo apt install raspi-utils"
    exit 1
fi
echo ""

# ── Test 1: Green LED ──────────────────────────────────────────────
echo "[Test 1] Green LED (GPIO $GREEN_LED) — 2s ON"
if pinctrl set $GREEN_LED op dh 2>/dev/null; then
    sleep 2
    pinctrl set $GREEN_LED op dl
    pass "Green LED toggled"
else
    fail "Green LED (cannot access GPIO — is this a Pi 5?)"
fi

# ── Test 2: Red LED ────────────────────────────────────────────────
echo "[Test 2] Red LED (GPIO $RED_LED) — 2s ON"
if pinctrl set $RED_LED op dh 2>/dev/null; then
    sleep 2
    pinctrl set $RED_LED op dl
    pass "Red LED toggled"
else
    fail "Red LED"
fi

# ── Test 3: Buzzer ─────────────────────────────────────────────────
echo "[Test 3] Buzzer (GPIO $BUZZER) — 3 short pulses"
buzzer_ok=true
for _ in 1 2 3; do
    if ! pinctrl set $BUZZER op dh 2>/dev/null; then
        buzzer_ok=false
        break
    fi
    sleep 0.15
    pinctrl set $BUZZER op dl
    sleep 0.1
done
if $buzzer_ok; then
    pass "Buzzer pulsed 3x"
else
    fail "Buzzer"
fi

# ── Test 4: Button (read current state) ────────────────────────────
echo "[Test 4] Button (GPIO $BUTTON) — reading state"
pinctrl set $BUTTON ip pu 2>/dev/null || true
sleep 0.3  # let pull-up stabilize before first read
level=$(pinctrl lev $BUTTON 2>/dev/null) && {
    if [ "$level" = "0" ]; then
        echo "  (Button is currently PRESSED)"
    else
        echo "  (Button is currently released — expected)"
    fi
    pass "Button read succeeded (level=$level)"
} || {
    fail "Button read"
}

# ── Test 5: Button (interactive press) ─────────────────────────────
echo ""
echo "[Test 5] Button — press within 5 seconds..."
pinctrl set $BUTTON ip pu 2>/dev/null || true
sleep 0.3  # let pull-up stabilize
btn_pressed=false
for _ in $(seq 1 50); do
    level=$(pinctrl lev $BUTTON 2>/dev/null) || true
    if [ "$level" = "0" ]; then
        btn_pressed=true
        break
    fi
    sleep 0.1
done
if $btn_pressed; then
    pass "Button press detected!"
    # Confirmation beep.
    pinctrl set $BUZZER op dh 2>/dev/null || true
    sleep 0.1
    pinctrl set $BUZZER op dl 2>/dev/null || true
else
    echo "  (No press detected — skipping, not a failure)"
fi

# ── Test 6: OLED display (I2C scan) ────────────────────────────────
echo ""
echo "[Test 6] OLED display — scanning I2C bus 1 for 0x3c"

OLED_ADDR=0x3c
I2C_BUS=1
OLED_FOUND=false

if [ ! -e /dev/i2c-1 ]; then
    fail "I2C bus 1 not available (/dev/i2c-1 missing)"
    echo "  Enable I2C with: sudo raspi-config nonint do_i2c 0 && sudo modprobe i2c-dev"
else
    if ! command -v i2cdetect &>/dev/null || ! command -v i2ctransfer &>/dev/null; then
        echo "  i2c-tools not found — installing..."
        apt-get install -y -qq i2c-tools 2>/dev/null || true
    fi

    if command -v i2cdetect &>/dev/null; then
        i2c_out=$(i2cdetect -y $I2C_BUS 2>/dev/null) || true
        if echo "$i2c_out" | grep -q "3c"; then
            pass "SSD1306 detected at $OLED_ADDR on I2C bus $I2C_BUS"
            OLED_FOUND=true
        else
            fail "No device at $OLED_ADDR (check wiring: SDA=pin3, SCL=pin5, VCC=3.3V, GND=pin25)"
            echo "  i2cdetect output:"
            echo "$i2c_out" | sed 's/^/    /'
        fi
    else
        fail "Could not install i2c-tools (run: sudo apt install i2c-tools)"
    fi
fi

# ── Test 7: OLED display (init + full white screen) ────────────────
if $OLED_FOUND && command -v i2ctransfer &>/dev/null; then
    echo ""
    echo "[Test 7] OLED display — init SSD1306 and fill white"

    # Helper: send one or more command bytes to the SSD1306.
    oled_cmd() { i2ctransfer -y $I2C_BUS w$(($# + 1))@$OLED_ADDR 0x00 "$@"; }

    # SSD1306 init sequence (128x64, internal charge pump).
    oled_cmd 0xAE              # display off
    oled_cmd 0xD5 0x80         # clock divide ratio
    oled_cmd 0xA8 0x3F         # multiplex ratio = 63 (64 rows)
    oled_cmd 0xD3 0x00         # display offset = 0
    oled_cmd 0x40              # start line = 0
    oled_cmd 0x8D 0x14         # charge pump ON
    oled_cmd 0x20 0x00         # horizontal addressing mode
    oled_cmd 0xA1              # segment re-map (col 127 → SEG0)
    oled_cmd 0xC8              # COM scan direction remapped
    oled_cmd 0xDA 0x12         # COM pins config
    oled_cmd 0x81 0xCF         # contrast
    oled_cmd 0xD9 0xF1         # pre-charge period
    oled_cmd 0xDB 0x40         # VCOMH deselect level
    oled_cmd 0xA4              # display follows RAM
    oled_cmd 0xA6              # normal (not inverted)

    # Reset column and page pointers to cover the full 128x64 area.
    oled_cmd 0x21 0x00 0x7F    # column range 0–127
    oled_cmd 0x22 0x00 0x07    # page range 0–7

    oled_cmd 0xAF              # display ON

    # Fill the entire framebuffer with 0xFF (white).
    # 128 cols × 8 pages = 1024 bytes, sent in 8 transfers of 128 bytes.
    white_row=$(printf '0xff %.0s' $(seq 1 128))
    for _ in $(seq 0 7); do
        # w129 = 1 control byte (0x40 = data mode) + 128 data bytes.
        i2ctransfer -y $I2C_BUS w129@$OLED_ADDR 0x40 $white_row
    done

    pass "OLED showing full white screen"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Results: $pass_count passed, $fail_count failed ==="
if [ "$fail_count" -gt 0 ]; then
    exit 1
fi
