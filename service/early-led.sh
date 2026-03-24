#!/bin/bash
# Early boot LED indicator - blinks green LED until omgrab takes over

GPIO_PIN=17

# Configure pin as output using pinctrl (Pi 5)
pinctrl set $GPIO_PIN op

# Blink fast (~5Hz) until killed
while true; do
    pinctrl set $GPIO_PIN dh  # high
    sleep 0.1
    pinctrl set $GPIO_PIN dl  # low
    sleep 0.1
done
