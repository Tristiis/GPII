import json
import numba
import pyaudio
import uncertainties
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

#from Praktikumsmodul import *

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

def signal_generation(config:dict):
    """
    start of signal is silence, calibration, silence
    """
    @numba.njit(fastmath = True, parallel = True)
    
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
    return full_signal

# ---------- Measurement Pipeline ----------

def measurement(sign): # needs to be implemented
    return [sign, sign] # first ref then mes signal

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray, config:dict):
    peaks, props = find_peaks(sign)

    silence = config["dead_time"] * config["srate"]
    sample_time_size = config["sample_time"] * config["srate"]
    peak_width = int(3/4 * config["srate"]/config["cal_freq"])
    anti_transient = int(config["srate"] * config["a_transient"])

    plt.plot(sign, rasterized = True)
    plt.axvline(peaks[0])

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

        sign = sign[int(peak_index):]
    else:
        sign = sign[peaks[0]:]

    fig_slic, axs_slic = plt.subplots()

    axs_slic.plot(sign, rasterized = True)

    sliced_signals = []
    for i in range(len(k_vals) * len(mod_vals)):
        low_index = silence + peak_width + anti_transient + silence * i + sample_time_size * i
        high_index = silence + peak_width + silence * i + sample_time_size * (i+1)
        axs_slic.axvline(low_index)
        axs_slic.axvline(high_index)
        sliced_signals.append(sign[low_index : high_index]) 

    plt.show()

    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    counter = 0
    for k in range(len(k_vals)):
        for f_m in range(len(mod_vals)):
            arr[k, f_m] = sliced_signals[counter]
            counter += 1
    return arr

# ---------- STI Computation ----------

def sti_comp(signs, config:dict, newpath:str):
    sos_low = signal.butter(20, 100, 'low', fs = config["srate"], output = "sos")

    def envelope_detection(sign:np.ndarray):
        arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
        for i_k in range(len(k_vals)):
            low_f:float = k_vals[i_k]["f_c"] / np.sqrt(2)
            high_f:float = np.sqrt(2) * k_vals[i_k]["f_c"]
            sos_band = signal.butter(20, (low_f, high_f), "band", fs = config["srate"], analog = False, output = "sos")
            for j_f_m in range(len(mod_vals)):
                arr[i_k, j_f_m] = signal.sosfilt(sos_band, sign[i_k, j_f_m])
                arr[i_k, j_f_m] *= arr[i_k, j_f_m]
                y = signal.sosfilt(sos_low, arr[i_k, j_f_m])
                arr[i_k, j_f_m] = y
        return arr

    def modulation_depths(I:np.ndarray, time:np.ndarray):
        arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
        for i_k in range(len(k_vals)):
            for j_f_m in range(len(mod_vals)):
                sin_sum = (np.sum(I[i_k, j_f_m] * np.sin(2 * np.pi * mod_vals[j_f_m] * time)))**2
                cos_sum = (np.sum(I[i_k, j_f_m] * np.cos(2 * np.pi * mod_vals[j_f_m] * time)))**2
                denom_sum = np.sum(I[i_k, j_f_m])
                arr[i_k, j_f_m] = 2 * np.sqrt(sin_sum + cos_sum) / denom_sum
        return arr
    
    def limit_mod_ratio(m):
        return min([m, 1])
    
    def auditory_effects(m):
        pass

    def snr_comp(m):
        if m == 1:
            return 15
        snr = 10 * np.log10(m / (1 - m))
        match snr:
            case _ if snr < -15:
                return -15
            case _ if snr > 15:
                return 15
            case _:
                return snr
    
    def transmission_index(snr):
        return (snr + 15) / 30
    
    def modulation_transfer_index(ti):
        mti = np.empty(len(k_vals))
        for k in range(len(k_vals)):
            mti[k] = np.mean(ti[k])
        return mti

    def sti_last_step(mti):
        first_term = np.sum([alpha_k[k] * mti[k] for k in range(len(k_vals))])
        second_term = np.sum([beta_k[k] * np.sqrt(mti[k] * mti[k+1]) for k in range(len(k_vals)-1)])
        return first_term - second_term

    params = {
        "sign": [],
        "I_k_m": [],
        "mod_dep": []
    }

    time = np.arange(0, config["sample_time"] - config["a_transient"], 1 / config["srate"])

    for sign, i in zip(signs, range(len(signs))):
        params["sign"].append(signal_slicing(sign, config))
        params["I_k_m"].append(envelope_detection(params["sign"][i]))
        params["mod_dep"].append(modulation_depths(params["I_k_m"][i], time))

    m_k_fm = params["mod_dep"][1] / params["mod_dep"][0]

    limit_mod_ratio_vec = np.vectorize(limit_mod_ratio)
    m_k_fm = limit_mod_ratio_vec(m_k_fm)

    # steps 5 and 6 are still missing because we first need to understand what value to use for I_k

    snr_comp_vec = np.vectorize(snr_comp)
    snr_k_fm = snr_comp_vec(m_k_fm)

    transmission_index_vec = np.vectorize(transmission_index)
    ti = transmission_index_vec(snr_k_fm)

    mti = modulation_transfer_index(ti)

    sti = sti_last_step(mti)
    print(f"STI Value: {sti}")

    k = [k_vals[i]["f_c"] for i in k_vals]
    k = [f"{i/1000}k" if i >= 1000 else i for i in k]

    fig_ti, axs_ti = plt.subplots()

    im = axs_ti.imshow(ti)

    axs_ti.set_title("Transfer Index TI")
    axs_ti.set_xticks(range(len(mod_vals)), labels=mod_vals,rotation=45, ha="right", rotation_mode="anchor")
    axs_ti.set_yticks(range(len(k_vals)), labels=k)
    axs_ti.set_xlabel("Modulation Frequencies [Hz]")
    axs_ti.set_ylabel("Center frequency [Hz]")

    fig_ti.colorbar(im, ax=axs_ti, orientation='horizontal', fraction=.1)
    fig_ti.tight_layout()
    
    fig_ti.savefig(fname = newpath + r"TI_plot.pdf", format = "pdf")

    plt.show()

    return sti, ti

# ---------- Main ----------

def main():
    sign = signal_generation(config)

    path = r"\Data"

    newpath = path + rf"\Messung_{0}"
    counter = 1

    logical = input("Proceed with saving all signals? (y/n) ")
    # check wether or not the path to the raw measurement data exists
    if logical in ["y", "Y"]:
        # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
        # Posted by mcandre, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-12-15, License - CC BY-SA 3.0
        while os.path.exists(newpath):
            newpath = path + rf"\Messung_{counter}"
            counter += 1
        makedirs(newpath)

        write(filename = newpath + r"\Input.wav", rate = config["srate"], data = sign)

    signs = measurement(sign)

    # ---------- Saving the Data in a csv file ----------
    
    sti, ti = sti_comp(signs, config, newpath)

    logical = input("Proceed with saving all signals? (y/n) ")
    # check wether or not the path to the raw measurement data exists
    if logical in ["y", "Y"]:
        print("Saving data...")
        pd.DataFrame({"ref": signs[0], "mes": signs[1]}).to_csv(newpath + r"\Messdaten.csv", sep = ";", index = False)
        
if __name__ == "__main__":
    main()