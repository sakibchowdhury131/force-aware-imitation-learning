#!/usr/bin/env python3

import os
import ctypes
import numpy as np

from calibrate_firmware_gravity import (
    load_api,
    connect,
    GRAVITY_PARAM_SIZE,
    GRAVITY_OPTIMAL,
)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(PIPELINE_DIR, "data", "gravity_params.npy")


def main():

    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(
            f"Could not find gravity parameter file:\n{PARAMS_PATH}"
        )

    params = np.load(PARAMS_PATH).astype(np.float32)

    if len(params) != GRAVITY_PARAM_SIZE:
        raise RuntimeError(
            f"Expected {GRAVITY_PARAM_SIZE} parameters, got {len(params)}"
        )

    api = load_api()

    try:
        connect(api, control=True)

        buf = (ctypes.c_float * GRAVITY_PARAM_SIZE)(*params)

        print("Applying gravity parameters...")

        r1 = api.SetGravityOptimalZParam(buf)
        print(f"SetGravityOptimalZParam -> {r1}")

        print("Switching gravity mode to OPTIMAL...")

        r2 = api.SetGravityType(GRAVITY_OPTIMAL)
        print(f"SetGravityType -> {r2}")

        if (r1 in (1, 2005)) and (r2 == 1):
            print("\nSUCCESS")
            print("Firmware gravity compensation is now using the saved parameters.")
        else:
            print("\nFAILED")
            print(f"SetGravityOptimalZParam returned {r1}")
            print(f"SetGravityType returned {r2}")

    finally:
        api.CloseAPI()
        print("\nAPI closed.")


if __name__ == "__main__":
    main()
