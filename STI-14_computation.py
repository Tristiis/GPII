import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acoustics.generator as generator
from tqdm import tqdm
from os import makedirs
from scipy import signal
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.io.wavfile import read

# ---------- Variables prescribed by the standard ----------

k_vals = {
    0 : {"f_c":125,     "L_k": -2.5,    "sigma": 2},
    1 : {"f_c":250,     "L_k": 0.5,     "sigma": 2},
    2 : {"f_c":500,     "L_k": 0,       "sigma": 2},
    3 : {"f_c":1000,    "L_k": -6,      "sigma": 2},
    4 : {"f_c":2000,    "L_k": -12,     "sigma": 2},
    5 : {"f_c":4000,    "L_k": -18,     "sigma": 2},
    6 : {"f_c":8000,    "L_k": -24,     "sigma": 2}
}

mod_vals = [0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5]

alpha_k = [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173]

beta_k = [0.085, 0.078, 0.065, 0.011, 0.047, 0.095]

# ---------- System Variables ----------

with open('STI-14_config.json', 'r') as file:
    config = json.load(file)

calib_csvs = [pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Calibration_files\Mic_old.csv", header = 0), pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Calibration_files\Mic_old.csv", header = 0)]

calibration_overwrite = False
test_phase = False
ref_signal = False

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray, calibration_intervention: bool, peak_index = None):
    silence = config["dead_time"] * config["srate"]
    sample_time_size = config["sample_time"] * config["srate"]
    peak_width = int(config["srate"]/(2 * config["cal_amp_freq"]))
    anti_transient = int(config["srate"] * config["a_transient"])

    mean = np.int32(np.mean(sign))
    sign -= mean

    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_amp_freq"] * t/config["srate"]) * np.sin(2 * np.pi * config["cal_freq"] * t/config["srate"]) for t in range(peak_width)])

    corr = signal.correlate(sign[:len(sign)//16], calibration)
    lags = signal.correlation_lags(len(sign[:len(sign)//16]), len(calibration))

    peaks, props = signal.find_peaks(corr/max(corr), prominence = 1)

    lag = lags[peaks[np.where(props["prominences"] == max(props["prominences"]))]][0] + peak_width//2

    if calibration_overwrite == True and peak_index != None:
        sign = sign[peak_index:]
    elif calibration_overwrite == False and peak_index == None:
        plt.plot(sign[:config["srate"]*60], rasterized = True)
        plt.axvline(lag)

        print("Please check if the calibration peak was correctly identified in the following plot.")
        plt.show()

        logical = input("Was the Calibration Peak found correctly? (y/n) ")
        if logical in ["n", "N"]:
            print("Please name the index of the calibration peak.")
            plt.plot(sign[:config["srate"]*60], rasterized = True)
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
            sign = sign[lag:]
            calibration_intervention = False
    elif peak_index != None:
        sign = sign[int(peak_index):]
    else:
        sign = sign[lag:]
        calibration_intervention = False

    sliced_signals = []
    for i in range(len(k_vals) * len(mod_vals)):
        low_index = silence + peak_width//2 + anti_transient + silence * i + sample_time_size * i
        high_index = silence + peak_width//2 + silence * i + sample_time_size * (i+1)
        sliced_signals.append(sign[low_index : high_index]) 
    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    counter = 0
    for k in range(len(k_vals)):
        for f_m in range(len(mod_vals)):
            arr[k, f_m] = sliced_signals[counter]
            counter += 1
    if calibration_intervention == True:
        # Source - https://stackoverflow.com/a/13747443
        # Posted by Bitwise
        # Retrieved 2026-07-15, License - CC BY-SA 3.0
        # d=np.array(dataPoints.tolist())
        return np.array(arr.tolist()), peak_index # type: ignore
    return np.array(arr.tolist()), None

# ---------- Equalizer ----------
def equalize(sign, calib_csv):
    fft_sign = np.fft.rfft(sign)
    fq = np.fft.rfftfreq(len(sign), d = 1/config["srate"])
    spl = interpolate.interp1d(calib_csv.x, calib_csv.y, fill_value = "extrapolate") # type: ignore
    x = np.linspace(0, max(fq), len(fft_sign))
    calib = spl(x)

    fft_sign /= calib

    sign = np.fft.irfft(fft_sign)

    return sign

# ---------- STI Computation ----------

# low pass for envelope detection
sos_low = signal.butter(20, 100, 'low', fs = config["srate"], output = "sos")

# band filters for sti comp
band_filters = []
for i_k in range(len(k_vals)):
    low_f:float = k_vals[i_k]["f_c"] / np.sqrt(2)
    high_f:float = np.sqrt(2) * k_vals[i_k]["f_c"]
    sos_band = signal.butter(20, (low_f, high_f), "band", fs = config["srate"], analog = False, output = "sos")
    band_filters.append([sos_band for _ in range(14)])
band_filters = np.array(band_filters)


# constants and functions for sti comp
anti_transient = int(config["srate"] * config["a_transient"])

def envelope_detection(sign):
    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    for i_k in range(len(k_vals)):
        for j_f_m in range(len(mod_vals)):
            arr[i_k, j_f_m] = signal.sosfiltfilt(band_filters[i_k,j_f_m], sign[i_k, j_f_m])
            arr[i_k, j_f_m] *= arr[i_k, j_f_m]
            y = signal.sosfiltfilt(sos_low, arr[i_k, j_f_m])
            arr[i_k, j_f_m] = y[anti_transient:]
    return arr
    """arr = signal.sosfiltfilt(band_filters, sign)
    arr *= arr
    y = signal.sosfiltfilt(sos_low, arr)
    arr = y[:, :, anti_transient:]
    return arr"""

def modulation_depths(I:np.ndarray, time:np.ndarray):
    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    tmp_mod_vals = [mod_vals for _ in range(len(k_vals))]
    def tmp(I, mods):
        sin_sum = (np.sum(I * np.sin(2 * np.pi * mods * time)))**2
        cos_sum = (np.sum(I * np.cos(2 * np.pi * mods * time)))**2
        denom_sum = np.sum(I)
        return 2 * np.sqrt(sin_sum + cos_sum) / denom_sum
    tmp = np.vectorize(tmp)
    return tmp(I, tmp_mod_vals)

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

limit_mod_ratio_vec = np.vectorize(limit_mod_ratio)
snr_comp_vec = np.vectorize(snr_comp)
transmission_index_vec = np.vectorize(transmission_index)


def sti_comp(sliced_signs):
    params = {
        "sign": sliced_signs,
        "I_k_m": [],
        "mod_dep": []
    }

    time = np.arange(0, config["sample_time"] - 2 * config["a_transient"], 1 / config["srate"])

    if ref_signal == True:
        for i in range(2):
            params["I_k_m"].append(envelope_detection(params["sign"][i]))
            params["mod_dep"].append(modulation_depths(params["I_k_m"][i], time))

        m_k_fm = params["mod_dep"][0] / params["mod_dep"][1]
    else:
        params["I_k_m"].append(envelope_detection(params["sign"][0]))
        params["mod_dep"].append(modulation_depths(params["I_k_m"][0], time))
        m_k_fm = params["mod_dep"][0] / 1

    m_k_fm = limit_mod_ratio_vec(m_k_fm)

    # steps 5 and 6 are still missing because we first need to understand what value to use for I_k

    snr_k_fm = snr_comp_vec(m_k_fm)
    
    ti = transmission_index_vec(snr_k_fm)

    mti = modulation_transfer_index(ti)

    sti = sti_last_step(mti)

    return sti, ti

# monte carlo constants
k_fq = np.array([k_vals[i]["f_c"] for i in k_vals]).reshape(7,1).repeat(14, -1).reshape(7,14,1)
def randomise(sign, k_fq):
    fft_sign = np.fft.rfft(sign, n = sign.shape[-1], axis = -1)
    fq = np.fft.rfftfreq(sign.shape[-1], d = 1/config["srate"])
    k_fq = k_fq.repeat(len(fq), -1)
    boa = k_fq / np.sqrt(2) < fq
    bob = fq < np.sqrt(2) * k_fq
    # Source - https://stackoverflow.com/a/2357785
    # Posted by Steve Tjoa
    # Retrieved 2026-07-15, License - CC BY-SA 2.5
    # import scipy
    # X = scipy.rand(9,4,1)
    # Y = X.repeat(4096,2)
    rand = np.random.normal(loc = 1, scale = config["std_mc"], size = (7,14,1)).repeat(len(fq), -1)
    multiplier = np.where(boa & bob, rand, 1)
    fft_sign *= multiplier
    return np.fft.irfft(fft_sign, n = sign.shape[-1], axis = -1)

def monte_carlo(sliced_signs, newpath):
    sti_li = np.empty(config["N"], dtype = np.float64)
    ti_li = np.empty(shape = (config["N"], 7, 14), dtype = np.float64)

    for i in tqdm(range(config["N"]), colour = "#FF13F0"):
        randomised_signs = []
        for sign in sliced_signs:
            randomised_signs.append(randomise(sign, k_fq))

        sti_li[i], ti_li[i] = sti_comp(randomised_signs)

    u_sti = np.std(sti_li)
    u_ti = np.std(ti_li)

    print(f"STI: {np.mean(sti_li)}")
    print(f"Unc.Sti: {u_sti}")

    fig, axs = plt.subplots()

    axs.hist(sti_li)
    fig.savefig(fname = newpath + r"\Unc_hist.pdf", format = "pdf")
    plt.close("all")
    #plt.show()

    return u_sti, u_ti

def plt_sav_results(sti, ti, newpath):
    k = [k_vals[i]["f_c"] for i in k_vals]
    k = [f"{i/1000}k" if i >= 1000 else i for i in k]

    fig_ti, axs_ti = plt.subplots()

    im = axs_ti.imshow(ti, vmin=0, vmax=1)

    axs_ti.set_title("Transfer Index TI")
    axs_ti.set_xticks(range(len(mod_vals)), labels=mod_vals,rotation=45, ha="right", rotation_mode="anchor")
    axs_ti.set_yticks(range(len(k_vals)), labels=k)
    axs_ti.set_xlabel("Modulation Frequencies [Hz]")
    axs_ti.set_ylabel("Center frequency [Hz]")

    fig_ti.colorbar(im, ax=axs_ti, orientation='horizontal', fraction=.1)
    fig_ti.tight_layout()
    
    if test_phase == False: #  and not os.path.exists(newpath + r"\TI_plot.pdf")
        fig_ti.savefig(fname = newpath + r"\TI_plot_wo_ref.pdf", format = "pdf") # _wo_ref
        df = pd.DataFrame(ti, index = k)
        df.to_csv(newpath + r"\TI_Daten_wo_ref.csv", sep = ";", header = mod_vals) # _wo_ref

    #print(f"STI Value: {sti}")
    #plt.show()

def main(num):
    global calibration_overwrite
    newpath = path + rf"\Messung_{num}"

    calibration_intervention = True

    with open(newpath + r"\Config.json", mode = "r") as fl:
        js = json.load(fl)

    if "calibration_intervention" in js.keys() and js["calibration_intervention"] == 1:
        calibration_overwrite = True
    else:
        calibration_overwrite = False

    signs = []
    if ref_signal == True:
        for i in [r"\Mes.wav", r"\Ref.wav"]: # , r"\Ref.wav"
            srate, data = read(path + rf"\Messung_{num}" + i)
            signs.append(data)
    else:
        srate, data = read(path + rf"\Messung_{num}" + r"\Mes.wav")
        signs.append(data)
    
    sliced_signs = []
    peak_index = []

    for sign, i in zip(signs, range(len(signs))):
        plt.close("all")
        if i == 0 and "peak_index_mes" in js.keys():
            slices, peak = signal_slicing(sign, calibration_intervention, js["peak_index_mes"])
        elif i == 1:
            slices, peak = signal_slicing(sign, calibration_intervention, 56350)
        else:
            slices, peak = signal_slicing(sign, calibration_intervention)
        peak_index.append(peak)
        sliced_signs.append(slices)

    for sign in range(len(sliced_signs)):
        for i_k in range(len(sliced_signs[sign])):
            for j_fm in range(len(sliced_signs[sign][i_k])):
                sliced_signs[sign][i_k][j_fm] = equalize(sliced_signs[sign][i_k][j_fm], calib_csvs[sign])

    sti, ti = sti_comp(sliced_signs)

    u_sti, u_ti = monte_carlo(sliced_signs, newpath)

    plt_sav_results(sti, ti, newpath)

    js["STI_wo_ref"] = sti
    js["u_STI_wo_ref"] = u_sti
    js["calibration_intervention"] = 0 if calibration_intervention == False else 1
    if calibration_intervention == True and type(peak_index) != list[None]:
        js["peak_index_ref"] = peak_index[0]
    elif ref_signal == True:
        js["peak_index_mes"] = peak_index[1]
    
    js = json.dumps(js, indent = 4)

    with open(newpath + r"\Config.json", mode = "w") as fl:
        fl.write(js)

    plt.close("all")

    del js, ti, sliced_signs, slices, data # type:ignore

if __name__ == "__main__":
    path = r"C:\Programmieren\Praktikum\GPII\Data\STI"
    while True:
        for i in tqdm(range(len(os.listdir(path))-2), colour= "#20C20E"):
            main(i+1)