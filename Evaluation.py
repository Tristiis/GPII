import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

df = pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Data\STI\STI_Datensatz.csv", sep = ";")

# Source - https://stackoverflow.com/a/28299215
# Posted by cel, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-23, License - CC BY-SA 3.0

df.sort_values(by = ["Material_Boden", "Durchmesser", "Abstand"], ascending = True, inplace=True)

"""
fig, ax = plt.subplots(figsize=(8,6))
for label, dfl in df.groupby("Material_Boden"):
    #dfl.plot("Durchmesser", "STI_wo_ref", ax=ax, label=label)
plt.legend()
"""


fig, axs = plt.subplots(layout='constrained')

df_bar = df.sort_values(["Approx_dist", "Material", "Durchmesser"], inplace=False)
df_bar.reset_index(drop=True, inplace=True)
df_bar = df_bar[df_bar["Material_Boden"] == "Laborbuch_Papier"]
group_bar = {}
distances = list(df_bar["Approx_dist"].value_counts().values)
for dist, i in zip(set(df_bar["Approx_dist"]), range(len(set(df_bar["Approx_dist"])))):
    group_bar[f"{dist} m"] = list(df_bar["STI_wo_ref"].round(2)[distances[i] * i:distances[i] * (i+1)])

res = axs.grouped_bar(group_bar, group_spacing=1, colors=["#abe83e", "#1074c3", "#c0379f"], alpha = 0.5) # , alpha = 0.5
#for container in res.bar_containers:
#    axs.bar_label(container, padding=3)

counter = 0
for label, i in df_bar.groupby("Material"):
    axs.plot(np.arange(0,3,1) + counter * 3, i.groupby("Durchmesser")["STI_wo_ref"].agg("mean"))
    counter += 1

#axs.plot([df_bar["STI_wo_ref"][3*i:3*(i+1)].agg("mean") for i in range(6)])

# Add some text for labels, title, etc.
axs.set_xticklabels([""] + [rf"{list(df_bar["Durchmesser"])[i]}, {list(df_bar["Material"])[i]}" for i in range(6)], rotation = 45, rotation_mode = "anchor", ha="right")
axs.set_ylabel('STI []')
axs.set_ylim(0,1)
axs.set_xlabel("Durchmesser [mm], Material")
axs.legend(title = "Approx. Abstand")

#df.groupby("Durchmesser").plot.scatter("Abstand", "STI_wo_ref",kind = "kde", ax = axs)
#df.groupby("Durchmesser").plot.scatter("Kraft", "STI_wo_ref")
#plot("Dichte", "STI_wo_ref") # type: ignore


fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Alles_2.0.pdf", format = "pdf")
plt.show()