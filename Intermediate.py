import os
import json
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    path = r"C:\Programmieren\Praktikum\GPII\Data\STI"

    js_files = []
    for i in tqdm(range(len(os.listdir(path))-1), colour= "#20C20E"):
        newpath = path + rf"\Messung_{i}"
        with open(newpath + r"\Config.json") as fl:
            data = json.load(fl)
        js_files.append(pd.DataFrame(data, index = pd.Index([i])))
    df = pd.concat(js_files)
    density = []

    for d, material in zip(df["Durchmesser"],df["Material"]):
        match d, material:
            case 0.5, "Nylon":
                density.append(0.235)
            case 0.99, "Nylon":
                density.append(0.605)
            case 0.35, "Nylon":
                density.append(0.118)
            case 0.2, "Kupfer":
                density.append(0.278)
            case 0.4, "Kupfer":
                density.append(0)
            case 0.8, "Kupfer":
                density.append(4.487)
    df["Dichte"] = density

    force = []
    for f, mu, l in zip(df["Frequenz"], df["Dichte"], df["Abstand"]):
        force.append((f * 2 * l)**2 * mu)

    df["Kraft"] = force
    df["Approx_dist"] = [round(i) for i in df["Abstand"]]

    df["Material_Boden"] = df["Material_Boden"].fillna("Laborbuch_Papier")
    df["Bodendicke"] = df["Bodendicke"].fillna(0.1)
    df["u_Bodendicke"] = df["u_Bodendicke"].fillna(0.002405)
    df["[Bodendicke]"] = df["[Bodendicke]"].fillna("mm")

    df.to_csv(path + r"\STI_Datensatz.csv", sep = ";")

if __name__ == "__main__":
    main()