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
from scipy import stats

test_mode = True

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
copper_bool = (df["Material"] == "Kupfer") & (df["Material_Boden"] == "Laborbuch_Papier")
nylon_bool = (df["Abstand"] != 3.182) & (df["Frequenz"] != 0) & (df["Material_Boden"] == "Laborbuch_Papier") & (df["Material"] == "Nylon") 
nylon_05 = (df["Frequenz"] == 589)
df = df.loc[copper_bool | nylon_bool | nylon_05].sort_values(["Approx_dist", "Material", "Durchmesser"])

def weighted_mean(values:list[float], uncertainties:list[float]) -> float:
    if len(values) != len(uncertainties):
        raise ArithmeticError("Miss match in val and unc length")
    upper = lower = 0
    for i in range(len(values)):
        upper += values[i] / np.square(uncertainties[i])
        lower += 1 / np.square(uncertainties[i])

    return upper / lower

def unc_sum(uncertainties:list[float]) -> float:
    lower = 0
    for i in uncertainties:
        lower += 1 / np.square(i)
    return lower

def internal_unc_type_a(uncertainties:list[float]) -> float:
    lower = unc_sum(uncertainties)
    n = len(uncertainties)
    return k[n - 1] * np.sqrt(1 / lower) / np.sqrt(n)

def external_unc_type_a(values:list[float], uncertainties:list[float], weighted_mean:float) -> float:
    n = len(uncertainties)
    if len(values) != n:
        raise ArithmeticError("Miss match in val and unc length")
    upper = 0
    lower = unc_sum(uncertainties)
    for i in range(len(values)):
        upper += np.square(values[i] - weighted_mean) / uncertainties[i]
    return k[n - 1] / np.sqrt(n) * np.sqrt(upper / ((len(values) - 1) * lower))

def weigted_type_a_unc(values:list[float], uncertainties:list[float]) -> list[float]:
    mean = weighted_mean(values, uncertainties)
    internal = internal_unc_type_a(uncertainties)
    external = external_unc_type_a(values, uncertainties, mean)
    return [mean, max(internal, external)]

def colour_background(fig, axs):
    vals = [0, 0.3, 0.45, 0.6, 0.75, 1]
    labels = ["Bad", "Poor", "Fair", "Good", "Excellent"]

    lim = axs.get_ylim()

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

# Dependence on the radius
with plt.style.context("science"):
    fig = plt.figure(figsize = (6,4))
    axs = sns.boxplot(data = df, x = "Durchmesser", y = "STI_wo_ref", hue = "Material") #, width = 0.4)
    #axs = sns.violinplot(data = df, x = "Durchmesser", y = "STI_wo_ref", alpha = 0.5, fill = False, inner = None, legend = False)
    sns.stripplot(data = df, x = "Durchmesser", y = "STI_wo_ref", legend = False, color = "orange", jitter=True, size=5)
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Abstand")
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Copper_good_radius.pdf", format = "pdf")
    #plt.show()
    #exit()

# Dependence on the Distance

table = stats.ttest_ind()
print(table)

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))
    sns.boxplot(data = df, x = "Approx_dist", y = "STI_wo_ref", hue = "Material", ax = axs)#, boxprops = dict(alpha = 0.6), patch_artist = True, positions = [2,3,4])
    sns.stripplot(data = df, x = "Approx_dist", y = "STI_wo_ref", hue = "Material", size=5, legend = False, ax = axs)
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Abstand")
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Copper_good_dist.pdf", format = "pdf")
    #plt.show()

with plt.style.context("science"):
    lm = sns.lmplot(data = df, x = "Abstand", y = "STI_wo_ref", hue="Material")
    lm.despine(top = False, right = False)
    lm.legend.remove() # type:ignore
    axs = lm.axes[0,0]
    fig = lm.figure
    fig.set_figwidth(6)
    fig.set_figheight(4)
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Abstand")
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\dist.pdf", format = "pdf")
    #plt.show()

with plt.style.context("science"):
    joint = sns.jointplot(kind = "kde", data = df, x = "Frequenz", y = "STI_wo_ref", hue = "Material")
    axs = joint.ax_joint
    fig = joint.figure
    sns.despine(ax = axs, top = False, right = False)
    fig.set_figwidth(6)
    fig.set_figheight(4)    
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Abstand")
    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\force.pdf", format = "pdf")
    #plt.show()

plt.show()

exit()
# ---------------- Bar plots ----------------

df.sort_values(by = ["Material_Boden", "Durchmesser", "Abstand"], ascending = True, inplace=True)

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4), layout='constrained')

    df_bar = df.sort_values(["Approx_dist", "Material", "Durchmesser"], inplace=False)
    df_bar.reset_index(drop=True, inplace=True)
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
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI []', rotation = 270)
    axs.yaxis.set_label_position("right")
    axs.set_xlabel("Durchmesser [mm], Material")
    axs.legend(title = "Approx. Abstand")

    #df.groupby("Durchmesser").plot.scatter("Abstand", "STI_wo_ref",kind = "kde", ax = axs)
    #df.groupby("Durchmesser").plot.scatter("Kraft", "STI_wo_ref")
    #plot("Dichte", "STI_wo_ref") # type: ignore

    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Barplots.pdf", format = "pdf")
plt.show()