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

# ---------- System Variables ----------

with open('resonance_config.json', 'r') as file:
    config = json.load(file)

# ---------- Signal Generation ----------

def signal_generation():
    """
    start of signal is silence, calibration, silence
    """

    # create a time array that is as long as one of the samples
    time = np.arange(0, config["chirp_time"], 1 / config["srate"])

    # define the space the different samples take up in the np array (for indexing)
    silence = config["dead_time"] * config["srate"]
    sample_time_size = config["chirp_time"] * config["srate"]
    peak_width = int(config["srate"]/(2 * config["cal_amp_freq"]))

    # create the calibration peak
    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_amp_freq"] * t/config["srate"]) * np.sin(2 * np.pi * config["cal_freq"] * t/config["srate"]) for t in range(peak_width)])

    # preallocate an empty np array to which the different signals are added to
    length_of_signal = sample_time_size + 2 * silence + peak_width
    full_signal = np.zeros(length_of_signal)
    
    # add calibration peak
    full_signal[silence : silence + peak_width] = calibration

    full_signal[2 * silence + peak_width : 2 * silence + peak_width + sample_time_size] = signal.chirp(time, config["f0"], time[-1], config["f1"], method = "logarithmic", phi = 90)

    # normalize signal to maximum amplitude of 1
    full_signal = full_signal - np.mean(full_signal)
    full_signal = full_signal / np.max(np.abs(full_signal))
    return full_signal

#---------- Main ----------

def main():
    sign = signal_generation()

    path = r"C:\Programmieren\Praktikum\GPII\Data\Res"

    newpath = path + rf"\Messung_{0}"
    counter = 1

    # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
    # Posted by mcandre, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    # checks which measurement row this is
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

    # writes generated signal to PCM wav file of specified bitrate and samplerate
    sign = sign * 2**(config["bitrate"]-1)
    write(filename = newpath + r"\Input.wav", rate = config["srate"], data = sign.astype(bitrate))
    
if __name__ == "__main__":
    main()