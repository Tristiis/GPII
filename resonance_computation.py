import json
import scienceplots
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from scipy import fftpack
from scipy import interpolate
from scipy import integrate
from scipy.io.wavfile import read, write

# Source - https://stackoverflow.com/a/3900167
# Posted by Herman Schaaf, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-24, License - CC BY-SA 4.0

# ---------- System Variables ----------

with open('resonance_config.json', 'r') as file:
    config = json.load(file)

calibration_overwrite = True
test_phase = 1

calib_csvs = [[pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Calibration_files\Res_data_mes.csv", sep = ";")],[pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Calibration_files\Res_data_ref.csv", sep = ";")]]

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray, srate:int, index:int):
    silence = config["dead_time"] * srate
    sample_time_size = config["chirp_time"] * srate
    peak_width = int(srate/(2 * config["cal_amp_freq"]))

    mean = np.int32(np.mean(sign))
    sign -= mean

    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_amp_freq"] * t/srate) * np.sin(2 * np.pi * config["cal_freq"] * t/srate) for t in range(peak_width)])

    corr = signal.correlate(sign[:len(sign)], calibration)
    lags = signal.correlation_lags(len(sign[:len(sign)]), len(calibration))

    peaks, props = signal.find_peaks(corr/max(corr), prominence = 1)

    lag = lags[peaks[np.where(props["prominences"] == max(props["prominences"]))]][0] + peak_width//2

    if index == 1:
        return sign[2 * silence + peak_width : 2 * silence + peak_width + sample_time_size]
    elif calibration_overwrite == False:
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

            sign = sign[int(peak_index) + peak_width + silence:int(peak_index) + peak_width + silence + sample_time_size]
        else:
            sign = sign[lag + peak_width//2 + silence: lag + peak_width//2 + silence + sample_time_size]
    else:
        sign = sign[lag + peak_width//2 + silence:lag + peak_width//2 + silence + sample_time_size]
    plt.close("all")
    return sign

# ---------- Resonance Computation ----------

def res_comp(sign, newpath:str, srate, index):
    freq = np.geomspace(config["f0"], config["f1"], len(sign)) #config["srate"] * config["chirp_time"])

    hilbert = signal.hilbert(sign)
    envelope = np.absolute(hilbert)
    phase_data = np.unwrap(np.angle(hilbert)) #type:ignore

    for _ in range(2):
        envelope = signal.savgol_filter(envelope, 600, 1)

    for calib_csv in calib_csvs[index]:
        spl = interpolate.interp1d(calib_csv.freq, calib_csv.res, fill_value = "extrapolate") # type: ignore
        calib = spl(freq)
    
        envelope /= calib

    return [freq[4000:], envelope[4000:]]

def fwhm(data):
    peaks, props = signal.find_peaks(data[1], prominence = 0.2)
    width, width_heights, left_ips, right_ips = signal.peak_widths(data[1], peaks, rel_height = 0.5)
    freqs = data[0, peaks]
    ress = data[1, peaks]
    left_pos = data[0, left_ips.astype(int)]
    right_pos = data[0, right_ips.astype(int)]
    return np.array([freqs, ress]), np.array([width_heights, left_pos, right_pos])

def plt_sav_results(data, newpath, index=2, dat = np.empty(1), peaks = np.empty(1)):
    with plt.style.context("science"):
        fig, axs = plt.subplots(figsize = (12,8))

        if index == 2:
            for i in data:
                axs.semilogx(i[0], i[1])
        else:
            axs.semilogx(data[0], data[1])
            #axs.scatter(dat[0], dat[1], marker = "x", color = "C2")
            #for i in range(len(peaks)):
            #    axs.hlines(y = peaks[0,i], xmin = peaks[1,i], xmax = peaks[2,i], color = "C2")

        axs.set_title("Frequency Response")
        axs.set_xlabel("Frequencies [Hz]")
        axs.set_ylabel("Arb. Units")
        axs.grid()

        fig.tight_layout()
        
        if test_phase == False:
            match index:
                case 0:
                    pd.DataFrame({"freq":data[0], "res": data[1]}).to_csv(newpath + r"\Res_data_mes.csv", sep = ";")
                    fig.savefig(fname = newpath + r"\Freq_Res_mes_plot.pdf", format = "pdf")
                case 1:
                    pd.DataFrame({"freq":data[0], "res": data[1]}).to_csv(newpath + r"\Res_data_ref.csv", sep = ";")
                    fig.savefig(fname = newpath + r"\Freq_Res_ref_plot.pdf", format = "pdf")
                case 2:
                    fig.savefig(fname = newpath + r"\Freq_Res_plot.pdf", format = "pdf")
        plt.show()
        plt.close("all")

def main(num, index_counter):
    path = r"C:\Programmieren\Praktikum\GPII\Data\Res"
    newpath = path + rf"\Messung_{num}"

    names = ["Mes", "Ref"]

    with open(newpath + r"\Config.json") as fl:
        config_local = json.load(fl)

    db_li = []
    database = []

    for index in range(2):
        srate, sign = read(newpath + rf"\{names[index]}.wav")
        sign = signal_slicing(sign, srate, index) / 2**32
        
        data = np.array(list(res_comp(sign, newpath, srate, index)))
        
        database.append(data)
        dat, peaks = fwhm(data)

        for i in range(len(dat[0])):
            row = config_local.copy()
            row["signal"] = names[index]
            row["peak_pos"] = dat[0,i]
            row["peak_height"] = dat[1,i]
            row["peak_fwhm_height"] = peaks[0,i]
            row["peak_fwhm_left_pos"] = peaks[1,i]
            row["peak_fwhm_right_pos"] = peaks[2,i]
            row["peak_fwhm_width"] = peaks[2,i] - peaks[1,i]
            db_li.append(pd.DataFrame(row, pd.Index([index_counter])))
            index_counter += 1

        plt_sav_results(data, newpath, index, dat, peaks)
    database = np.array(database)
    #print((database[0,1]**2).mean(), (database[1,1]**2).mean())
    plt_sav_results(database, newpath)
    return db_li, index_counter

if __name__ == "__main__":
    path = r"C:\Programmieren\Praktikum\GPII\Data\Res"

    main(32,0)

    js_files = []
    index_counter = 0
    for i in tqdm(range(22, 31), colour= "#20C20E"):
        db, index_counter = main(i, index_counter)
        js_files.extend(db)
    
    #main(22)

    df = pd.concat(js_files)
    #df.to_csv(path + r"\Res_Datensatz.csv")