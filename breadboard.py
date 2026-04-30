#!/usr/bin/env python3
from gpiozero import PWMLED, Button, RotaryEncoder
from picamera2 import Picamera2
from datetime import datetime
from signal import pause
import os
import time

# --- GPIO Pins (Verified for your hardware) ---
PIN_PWM = 13
PIN_CLK = 4
PIN_DT = 17
PIN_SW = 27  # Corrected from 22 to 27 based on your working test

# --- Components ---
# We use frequency=1000 to prevent LED flickering
led = PWMLED(PIN_PWM, frequency=1000)

# wrap=False prevents the value from jumping from 1.0 to -1.0 abruptly
encoder = RotaryEncoder(PIN_CLK, PIN_DT, wrap=False)

# --- Handlers ---
def update_brightness():
    """
    RotaryEncoder.value ranges from -1 to 1.
    We convert this to a 0.0 to 1.0 scale for the PWMLED.
    """
    # Normalize encoder (-1 to 1) to LED (0 to 1)
    new_val = (encoder.value + 1) / 2
    # Clamp the value just in case of precision errors
    led.value = max(0.0, min(1.0, new_val))
    print(f"Brightness: {int(led.value * 100)}%")


# --- Linking Events ---
encoder.when_rotated = update_brightness


# --- Main Loop ---
if __name__ == "__main__":
    print("--- System Active ---")
    print(f"LED Pin: {PIN_PWM} | Encoder Pins: {PIN_CLK},{PIN_DT}")
    print("Rotate knob to dim. Press knob to snap photo.")
    
    # Set initial brightness
    led.value = 0.5
    
    try:
        pause() # Keeps the script running to listen for interrupts
    except KeyboardInterrupt:
        print("\nCleaning up and shutting down...")
        camera.stop()
