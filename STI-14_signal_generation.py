import json
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acoustics.generator as generator
from tqdm import tqdm
from os import makedirs
from scipy import signal
from scipy.signal import find_peaks
from scipy.io.wavfile import write

# ---------- Variables prescribed by the standard ----------

k_vals = {
    0 : {"f_c":125, "L_k": -2.5},
    1 : {"f_c":250, "L_k": 0.5},
    2 : {"f_c":500, "L_k": 0},
    3 : {"f_c":1000, "L_k": -6},
    4 : {"f_c":2000, "L_k": -12},
    5 : {"f_c":4000, "L_k": -18},
    6 : {"f_c":8000, "L_k": -24}
}

mod_vals = [0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5]

alpha_k = [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173]

beta_k = [0.085, 0.078, 0.065, 0.011, 0.047, 0.095]

# ---------- System Variables ----------

with open('STI-14_config.json', 'r') as file:
    config = json.load(file)

# ---------- Signal Generation ----------

def signal_generation():
    """
    start of signal is silence, calibration, silence
    """

    def am_modulation(sign:np.ndarray, f_m:float, time:np.ndarray):
        """
        sign:np.ndarray -> 1d signal array
        f_m:float -> amplitude modulation frequency
        time:np.ndarray -> 1d time array
        """
        amod = np.sqrt(0.5 * (1 + np.cos(2 * np.pi * f_m * time)))

        return sign * amod

    def G_k(sign:np.ndarray, k:int):
        return sign * 10**(k_vals[k]["L_k"] / 20)

    time = np.arange(0, config["sample_time"], 1 / config["srate"])

    silence = config["dead_time"] * config["srate"]
    sample_time_size = config["sample_time"] * config["srate"]
    peak_width = int(config["srate"]/config["cal_freq"])

    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_freq"] * t/config["srate"]) for t in range(peak_width)])

    length_of_signal = 98 * sample_time_size + 100 * silence + peak_width

    full_signal = np.zeros(length_of_signal)
    full_signal[silence : silence + peak_width] = calibration

    i = 0

    for k in tqdm(k_vals):
        low_f:float = k_vals[k]["f_c"] / np.sqrt(2)
        high_f:float = np.sqrt(2) * k_vals[k]["f_c"]
        sos_band = signal.butter(20, (low_f, high_f), "band", fs = config["srate"], analog = False, output = "sos")
        for m in tqdm(mod_vals):
            noise = generator.pink(config["srate"] * config["sample_time"])
            noise = signal.sosfilt(sos_band, noise)
            sign = am_modulation(noise, m, time) # type: ignore
            sign = G_k(sign, k)
            
            low_index = 2 * silence + peak_width + silence * i + sample_time_size * i
            high_index = 2 * silence + peak_width + silence * i + sample_time_size * (i+1)
            full_signal[low_index:high_index] = sign
            i += 1

    full_signal = full_signal - np.mean(full_signal)
    full_signal = full_signal / np.max(np.abs(full_signal))
    return full_signal

# ---------- Main ----------

def main():
    sign = signal_generation()

    path = r"C:\Programmieren\Praktikum\GPII\Data"

    newpath = path + rf"\Messung_{0}"
    counter = 1

    # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
    # Posted by mcandre, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    while os.path.exists(newpath):
        newpath = path + rf"\Messung_{counter}"
        counter += 1
    makedirs(newpath)

    match config["bitrate"]:
        case 16:
            bitrate = np.int16
        case 32:
            bitrate = np.int32
        case _:
            raise ValueError

    print(max(sign), min(sign))
    sign = sign * 2**(config["bitrate"]-1)
    write(filename = newpath + r"\Input.wav", rate = config["srate"], data = sign.astype(bitrate))
    
if __name__ == "__main__":
    main()