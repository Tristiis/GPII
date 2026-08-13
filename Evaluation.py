import locale
import scienceplots
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("science")

#plt.rcParams.update({'font.size': 13})

# enable latex in plots
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams.update(mpl.rcParamsDefault)

#locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

df = pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Data\STI\STI_Datensatz.csv", sep = ";")
"""
sns.regplot(data = df, x = "Abstand", y = "STI_wo_ref")
sns.lmplot(data = df, x = "Abstand", y = "STI_wo_ref", hue = "Durchmesser")
sns.lmplot(data = df, x = "Abstand", y = "STI_wo_ref")
df.groupby("Durchmesser").plot(kind = "scatter", x = "Abstand", y = "STI_wo_ref", xerr = "u_Abstand", yerr = "u_STI_wo_ref", ax = axs)
"""

def colour_background(fig, axs, lim):
    vals = [0, 0.3, 0.45, 0.6, 0.75, 1]
    labels = ["Bad", "Poor", "Fair", "Good", "Excellent"]

    # following code snippet was created by AI
    colors = ["red", "yellow", "lime"]
    cmap = LinearSegmentedColormap.from_list("red_orange_green", colors)
    colours = cmap(np.linspace(0, 1, len(vals)))

    ticks = [i for i in vals if lim[0] < i < lim[1]]
    if lim[0] not in ticks:
        ticks.insert(0, lim[0])
    if lim[1] not in ticks:
            ticks.append(lim[1])
    norm = mcolors.BoundaryNorm(ticks, cmap.N)

    divider = make_axes_locatable(axs)
    cax = divider.append_axes("left", size="5%", pad = 0)
    sc_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sc_map.set_clim(lim)
    cbar = fig.colorbar(sc_map, cax=cax, orientation='vertical', spacing = "proportional")

    # Source - https://stackoverflow.com/a/76595252
    # Posted by AlefiyaAbbas, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-13, License - CC BY-SA 4.0
    cbar.set_ticklabels([])
    for i in range(len(ticks)-1):
        tick = (ticks[i] + ticks[i+1]) / 2
        if vals[0] <= tick <= vals[1]:
            label = labels[0]
        elif vals[1] <= tick <= vals[2]:
            label = labels[1]
        elif vals[2] <= tick <= vals[3]:
            label = labels[2]
        elif vals[3] <= tick <= vals[4]:
            label = labels[3]
        else:
            label = labels[4]
        cbar.ax.text(0.5, tick, s = label, ha='center', va='center', rotation = "vertical")

    #for i in range(len(vals) - 1):
    #    axs.axhspan(vals[i], vals[i + 1], color = colours[i], alpha = 0.7)


# ---------------- Copper better than Nylon ----------------
with plt.style.context("science"):
    fig = plt.figure(figsize = (6,4))
    axs = sns.boxplot(data = df.loc[df['Material_Boden'] == "Laborbuch_Papier"], x = "Approx_dist", y = "STI_wo_ref", hue = "Material")
    #axs = sns.stripplot(data = df.loc[df['Material_Boden'] == "Laborbuch_Papier"], x = "Approx_dist", y = "STI_wo_ref", color = "orange", jitter=0.2, size=4)
    lim = axs.get_ylim()
    colour_background(fig, axs, list(lim))
    axs.set_ylim(lim)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Abstand")

fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Copper_good.pdf", format = "pdf")
plt.show()
exit()

axs = sns.lmplot(data = df, x = "Abstand", y = "STI_wo_ref")

plt.show()

# ---------------- Bar plots ----------------

df.sort_values(by = ["Material_Boden", "Durchmesser", "Abstand"], ascending = True, inplace=True)

fig, axs = plt.subplots(layout='constrained')

df_bar = df.loc[(df["Abstand"] != 3.182) & (df["Frequenz"] != 0) & (df["Material_Boden"] == "Laborbuch_Papier") | (df["Frequenz"] == 589)].sort_values(["Approx_dist", "Material", "Durchmesser"], inplace=False)
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


#fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Alles_2.0.pdf", format = "pdf")
plt.show()