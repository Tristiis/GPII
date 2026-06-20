import json
import numba
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acoustics.generator as generator
from tqdm import tqdm
from os import makedirs
from scipy import signal
from scipy import interpolate
from scipy.io.wavfile import read

# ---------- System Variables ----------

with open('resonance_config.json', 'r') as file:
    config = json.load(file)

calibration_overwrite = False
test_phase = False

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray):
    silence = config["dead_time"] * config["srate"]
    sample_time_size = config["chirp_time"] * config["srate"]
    peak_width = int(config["srate"]/(2 * config["cal_amp_freq"]))

    mean = np.int32(np.mean(sign))
    sign -= mean

    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_amp_freq"] * t/config["srate"]) * np.sin(2 * np.pi * config["cal_freq"] * t/config["srate"]) for t in range(peak_width)])

    corr = signal.correlate(sign[:len(sign)//16], calibration)
    lags = signal.correlation_lags(len(sign[:len(sign)//16]), len(calibration))

    peaks, props = signal.find_peaks(corr/max(corr), prominence = 1)

    lag = lags[peaks[np.where(props["prominences"] == max(props["prominences"]))]][0] #+ peak_width//2

    if calibration_overwrite == False:
        plt.plot(sign, rasterized = True)
        plt.axvline(lag)

        print("Please check if the calibration peak was correctly identified in the following plot.")
        plt.show()

        logical = input("Was the Calibration Peak found correctly? (y/n) ")
        if logical in ["n", "N"]:
            print("Please name the index of the calibration peak.")
            plt.plot(sign, rasterized = True)
            plt.show()
            peak_index = input("Peak Index: ")
            
            def test_for_int(peak_index):
                try:
                    int(peak_index)
                except ValueError:
                    return False
                else:
                    return True

            while test_for_int(peak_index) == False:
                print("Please only input integers.")
                peak_index = input("Peak Index: ")

            sign = sign[int(peak_index) + peak_width + silence:]
        else:
            sign = sign[lag + peak_width + silence:]
    else:
        sign = sign[lag + peak_width + silence:]

    return sign

# ---------- Resonance Computation ----------

def res_comp(sign, newpath:str):
    sos_low = signal.butter(10, 80, 'low', fs = config["srate"], output = "sos")

    def envelope_detection(sign:np.ndarray):
        sign *= sign
        sign = signal.sosfiltfilt(sos_low, sign)
        return sign
    
    sign = envelope_detection(sign)

    freq = np.geomspace(config["f0"], config["f1"], len(sign)) #config["srate"] * config["chirp_time"])

    #peaks, props = signal.find_peaks(sign)
    #freq = [freq[i] for i in peaks]
    #res = [np.abs(sign[i]) for i in peaks]

    return freq, sign

def monte_carlo(sliced_signals, N):
    pass

def plt_sav_results(freq, res, newpath):
    fig, axs = plt.subplots()

    axs.plot(freq, res)

    axs.set_title("Frequency Response")
    axs.set_xlabel("Frequencies [Hz]")
    axs.set_ylabel("Sound Pressure Level [dB]")

    fig.tight_layout()
    
    if test_phase == False:
        fig.savefig(fname = newpath + r"\Freq_Res_plot.pdf", format = "pdf")

    plt.show()
    plt.close("all")

def main(num):
    path = r"C:\Programmieren\Praktikum\GPII\Data\Res"
    newpath = path + rf"\Messung_{num}"
    """
    with open(newpath + r"\Config.json", mode = "r") as fl:
        js = json.load(fl)
    """
    num = 2
    newpath = path + rf"\Messung_{num}"

    srate, sign = read(newpath + r"\Mes.wav")

    sign = signal_slicing(sign)

    freq, res = res_comp(sign, newpath)

    #u_sti, u_ti = monte_carlo(sliced_signs, config["N"])

    plt_sav_results(freq, res, newpath)


if __name__ == "__main__":
    path = r"C:\Programmieren\Praktikum\GPII\Data\STI"
    for i in tqdm(range(len(os.listdir(path))), colour= "#20C20E"):
        main(i)