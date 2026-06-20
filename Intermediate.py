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
    print(df)
    df.to_csv(path + r"\STI_Datensatz.csv", sep = ";")

if __name__ == "__main__":
    main()