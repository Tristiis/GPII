import locale
import scienceplots
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

test_mode = False
mc_tests = True

N = int(1e04)

plt.style.use("science")

cmap = ["#E1812C", "#3274A1"]

colours = ["#c235a1", "#7f3da9", "#4fd2d2"]

#plt.rcParams.update({'font.size': 13})

# enable latex in plots
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams.update(mpl.rcParamsDefault)

#locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

df_raw = pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Data\STI\STI_Datensatz.csv", sep = ";")

copper_bool = (df_raw["Material"] == "Kupfer") & (df_raw["Material_Boden"] == "Laborbuch_Papier")
nylon_bool = (df_raw["Abstand"] != 3.182) & (df_raw["Frequenz"] != 0) & (df_raw["Material_Boden"] == "Laborbuch_Papier") & (df_raw["Material"] == "Nylon") 
nylon_05 = (df_raw["Frequenz"] == 589)
df = df_raw.loc[copper_bool | nylon_bool | nylon_05].sort_values(["Approx_dist", "Material", "Durchmesser"])

def colour_background(fig, axs):
    vals = [0, 0.3, 0.45, 0.6, 0.75, 1]
    labels = ["Bad", "Poor", "Fair", "Good", "Excellent"]
    colours = ["#fe0000", "#ff7e00", "#fefe00", "#80ff00", "#01ff01"]

    lim = axs.get_ylim()

    ticks = [i for i in vals if lim[0] < i < lim[1]]
    if lim[0] not in ticks:
        ticks.insert(0, lim[0])
    if lim[1] not in ticks:
            ticks.append(lim[1])
    if len(ticks) != len(vals):
        counter = 0
        while lim[0] >= vals[counter]:
            counter += 1
        cut_colours = colours[counter - 1:]
    else:
        cut_colours = colours

    # following code snippet was created by AI
    cmap = LinearSegmentedColormap.from_list("red_orange_green", cut_colours)

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
def diameter(statistics = False, xlim = None, ylim = None):
    # Dependence on the radius
    with plt.style.context("science"):
        fig, axs = plt.subplots(figsize = (6,4))
        for mat, colour, fmt in zip(np.unique(df.Material), cmap, ["s", "v"]):
            df_tmp = df.loc[df.Material == mat].sort_values(["Durchmesser", "STI_wo_ref"])
            df_tmp = df_tmp.reset_index()
            
            for i, pad in zip(range(df_tmp.shape[0]), np.linspace(0.01, -0.01, df_tmp.shape[0])):
                x = df_tmp.loc[df_tmp.index == i].Durchmesser + pad
                y = df_tmp.loc[df_tmp.index == i].STI_wo_ref
                yerr = df_tmp.loc[df_tmp.index == i].u_STI_wo_ref
                axs.errorbar(x = x, y = y, yerr = yerr, fmt = fmt, c = colour, capsize = 2)

        if statistics == True:
            for i in [[0.2, 0.35], [0.4, 0.35], [0.4, 0.5], [0.8, 0.99]]:
                a_expc = df.loc[(df.Durchmesser == i[0]) & (df.Material == "Kupfer")]
                b_expc = df.loc[(df.Durchmesser == i[1]) & (df.Material == "Nylon")]

                p_li = np.empty(N, dtype = float)
                
                for j in tqdm(range(N), colour = "#20C20E"):
                    a = np.random.normal(a_expc.STI_wo_ref, scale = a_expc.u_STI_wo_ref)
                    b = np.random.normal(b_expc.STI_wo_ref, scale = b_expc.u_STI_wo_ref)
                    p_li[j] = stats.ttest_ind(a, b, equal_var = False)[1]

                # the following code was partially created with AI
                bracket_height = 0.02
                y_max = max(max(df.loc[df.Durchmesser == i[0]].STI_wo_ref + df.loc[df.Durchmesser == i[0]].u_STI_wo_ref), max(df.loc[df.Durchmesser == i[1]].STI_wo_ref + df.loc[df.Durchmesser == i[1]].u_STI_wo_ref))
                if i == [0.4, 0.5]:
                    y_max += 0.055
                axs.plot([i[0], i[0], i[1], i[1]], [y_max, y_max+bracket_height, y_max+bracket_height, y_max], lw=1.2, color='black')
                # the following code was partially created with AI
                axs.annotate(f'Med(p) = {round(np.median(p_li), 3)}\n IQR(p) = {round(stats.iqr(p_li), 3)}', xy=((i[0] + i[1])/2, y_max + bracket_height), 
                    xytext=(0, 3), # Offset a few points above the bracket
                    textcoords="offset points", 
                    ha='center', va='bottom', color='black')
                
                fig_hist, axs_hist = plt.subplots(figsize = (6,4))
                sns.kdeplot(data = p_li, ax = axs_hist, color = colours[0])#axs.hist(p_li, bins = "auto")
                axs_hist.axvline(np.median(p_li), label = "Med(p)", c = colours[1])
                axs_hist.axvspan(np.percentile(p_li, 25), np.percentile(p_li, 75), label = "IQR", color = colours[2], alpha = 0.3)
                axs_hist.set_title(f"MC Verteilung für $d$ = ({i[0]}, {i[1]})mm")
                axs_hist.legend()
                axs_hist.grid()
                axs_hist.set_xlabel("p(T-Test) [1]")
                fig_hist.tight_layout()
                fig_hist.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\MC_{i}.pdf", format = "pdf", dpi = 900)
                fig_hist.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\MC_{i}.png", format = "png", dpi = 900)
        if xlim != None:
            axs.set_xlim(xlim)
            axs.set_ylim(ylim)
        else:
            lim = axs.get_ylim()
            axs.set_ylim(lim[0], lim[1] + 0.04)
        colour_background(fig, axs)
        axs.yaxis.tick_right()
        axs.set_xlabel("Durchmesser [mm]")
        axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
        axs.yaxis.set_label_position("right")

        # Source - https://stackoverflow.com/a/39500357
        # Posted by gabra, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-25, License - CC BY-SA 4.0

        cu = mpatches.Patch(color=cmap[0], label='Kupfer')
        ny = mpatches.Patch(color=cmap[1], label='Nylon')

        axs.legend(handles=[cu, ny], title = "Material", frameon = True)
        axs.grid()
        fig.tight_layout()
        if test_mode == False and statistics == True:
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Durchmesser.pdf", format = "pdf", dpi = 900)
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Durchmesser.png", format = "png", dpi = 900)
        elif test_mode == False:
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Durchmesser_wo_stat.pdf", format = "pdf", dpi = 900)
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Durchmesser_wo_stat.png", format = "png", dpi = 900)
    return axs.get_xlim(), axs.get_ylim()

def distance(statistics = False):
    with plt.style.context("science"):
        fig, axs = plt.subplots(figsize = (6,4))
        for mat, colour, fmt in zip(np.unique(df.Material), cmap, ["s", "v"]):
            df_tmp = df.loc[df.Material == mat]
            df_tmp = df_tmp.reset_index()
            for i in range(df_tmp.shape[0]):
                x = df_tmp.loc[df_tmp.index == i].Abstand# + np.random.uniform(-0.01, 0.01)
                y = df_tmp.loc[df_tmp.index == i].STI_wo_ref
                yerr = df_tmp.loc[df_tmp.index == i].u_STI_wo_ref
                axs.errorbar(x = x, y = y, yerr = yerr, fmt = fmt, c = colour, capsize = 2)

        if statistics == True:
            for i in [2,3,4]:
                a_expc = df.loc[(df.Approx_dist == i) & (df.Material == "Kupfer")]
                b_expc = df.loc[(df.Approx_dist == i) & (df.Material == "Nylon")]

                p_li = np.empty(N, dtype = float)
                
                for j in tqdm(range(N), colour = "#20C20E"):
                    a = np.random.normal(a_expc.STI_wo_ref, scale = a_expc.u_STI_wo_ref)
                    b = np.random.normal(b_expc.STI_wo_ref, scale = b_expc.u_STI_wo_ref)
                    p_li[j] = stats.ttest_ind(a, b, equal_var = False)[1]

                fig_hist, axs_hist = plt.subplots(figsize = (6,4))
                sns.kdeplot(data = p_li, ax = axs_hist, color = colours[0])#axs.hist(p_li, bins = "auto")
                axs_hist.axvline(np.median(p_li), label = "Med(p)", c = colours[1])
                axs_hist.axvspan(np.percentile(p_li, 25), np.percentile(p_li, 75), label = "IQR", color = colours[2], alpha = 0.3)
                axs_hist.set_title(f"MC Verteilung für Abst. = {i}m")
                axs_hist.legend()
                axs_hist.grid()
                axs_hist.set_xlabel("p(T-Test) [1]")
                fig_hist.tight_layout()
                fig_hist.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\MC_{i}.pdf", format = "pdf", dpi = 900)
                fig_hist.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\MC_{i}.png", format = "png", dpi = 900)

                
                # the following code was partially created with AI
                shift = 0.2
                bracket_height = 0.02
                y_min = min(df.loc[df.Approx_dist == i].STI_wo_ref - df.loc[df.Approx_dist == i].u_STI_wo_ref)
                y_max = max(df.loc[df.Approx_dist == i].STI_wo_ref + df.loc[df.Approx_dist == i].u_STI_wo_ref)
                if i != 4:
                    axs.plot([i + shift, i + bracket_height + shift, i + bracket_height + shift, i + shift], [y_min, y_min, y_max, y_max], lw=1.2, color='black')

                    # the following code was partially created with AI
                    axs.annotate(f'Med(p) = {round(np.median(p_li), 3)}\n IQR(p) = {round(stats.iqr(p_li), 3)}', xy=(i+ bracket_height + shift + 0.3, (y_min + y_max) / 2), 
                        xytext=(0, 3), # Offset a few points above the bracket
                        textcoords="offset points", 
                        ha='center', va='bottom', color='black')
                else:
                    shift = 0.1
                    axs.plot([i - shift, i - bracket_height - shift, i - bracket_height - shift, i - shift], [y_min, y_min, y_max, y_max], lw=1.2, color='black')

                    # the following code was partially created with AI
                    axs.annotate(f'Med(p) = {round(np.median(p_li), 3)}\n IQR(p) = {round(stats.iqr(p_li), 3)}', xy=(i - bracket_height - shift - 0.3, (y_min + y_max) / 2 + 0.1), 
                        xytext=(0, 3), # Offset a few points above the bracket
                        textcoords="offset points", 
                        ha='center', va='bottom', color='black')
        lim = axs.get_ylim()
        axs.set_ylim(lim[0], lim[1] + 0.04)
        colour_background(fig, axs)
        axs.yaxis.tick_right()
        axs.set_xlabel("Länge [m]")
        axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
        axs.yaxis.set_label_position("right")

        # Source - https://stackoverflow.com/a/39500357
        # Posted by gabra, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-25, License - CC BY-SA 4.0

        cu = mpatches.Patch(color=cmap[0], label='Kupfer')
        ny = mpatches.Patch(color=cmap[1], label='Nylon')

        axs.legend(handles=[cu, ny], title = "Material", frameon = True)
        axs.grid()
        fig.tight_layout()
        if test_mode == False and statistics == True:
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abstand.pdf", format = "pdf", dpi = 900)
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abstand.png", format = "png", dpi = 900)
        elif test_mode == False:
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abstand_wo_stat.pdf", format = "pdf", dpi = 900)
            fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abstand_wo_stat.png", format = "png", dpi = 900)

if mc_tests == True:
    xlim = ylim = None
    for i in [True, False]:
        xlim, ylim = diameter(i, xlim, ylim)
        distance(i)

"""
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
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")
    axs.legend(title = "Approx. Länge")
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\dist.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\dist.png", format = "png", dpi = 900)
    #plt.show()
"""
"""
with plt.style.context("science"):
    joint = sns.jointplot(kind = "kde", data = df, x = "Frequenz", y = "STI_wo_ref", hue = "Material", legend = True)
    axs = joint.ax_joint
    fig = joint.figure
    sns.despine(ax = axs, top = False, right = False)
    fig.set_figwidth(6)
    fig.set_figheight(4)    
    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")
    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\force.pdf", format = "pdf")
    #plt.show()
"""
"""
df_abn = pd.read_csv(r"C:\Programmieren\Praktikum\GPII\Data\STI\STI_Abnutzung_Datensatz.csv", sep =";")

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))
    axs.plot(df_abn.index, df_abn.STI_wo_ref)

    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_xlabel("Anzahl der Messungen")
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")

    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abnutzung.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abnutzung.png", format = "png", dpi = 900)
"""
"""
with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))
    axs.errorbar(df_abn.index, df_abn.STI_wo_ref, yerr = df_abn.u_STI_wo_ref, capsize = 2, fmt = "o")

    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_xlabel("Anzahl der Messungen")
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")

    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abnutzung.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Abnutzung.png", format = "png", dpi = 900)
"""
"""
df_for = df_raw.loc[df_raw.Abstand == 3.182]

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))
    axs.errorbar(df_for.Frequenz, df_for.STI_wo_ref, yerr = df_for.u_STI_wo_ref, capsize = 2, fmt = "o")

    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_xlabel("Frequenz [Hz]")
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")

    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Spannung.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Spannung.png", format = "png", dpi = 900)
"""
df_mem = df_raw.loc[(df_raw.Durchmesser == 0.2) & (df_raw.Approx_dist == 4)]
df_mem = df_mem.sort_values("STI_wo_ref")
membranes = list(df_mem.Material_Boden)
thicknesses = list(df_mem.Bodendicke)
membranes = [f"{i}\n$d={j}" + r"~\text{mm}$" if i != "Laborbuch_Papier" else r"Papier $90~\frac{\text{g}}{\text{m}^2}$" for i, j in zip(membranes, thicknesses)]

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))
    axs.errorbar(membranes, df_mem.STI_wo_ref, yerr = df_mem.u_STI_wo_ref, capsize = 2, fmt = "o")

    colour_background(fig, axs)
    axs.yaxis.tick_right()
    axs.set_xlabel("Membran")
    axs.set_ylabel('STI [1]', rotation = 270, labelpad = 15)
    axs.yaxis.set_label_position("right")

    axs.grid()
    fig.tight_layout()
    if test_mode == False:
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Membran.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Membran.png", format = "png", dpi = 900)


data = []
path = r"C:\Programmieren\Praktikum\GPII\Data\Res"

for i in range(3):
    df = pd.read_csv(path + rf"\Messung_{i + 22}\Res_data_mes.csv", sep = ";")
    data.append([df.freq, df.res])

df = pd.read_csv(r"Data/Res/Res_Trans_Datensatz.csv")
df = df.loc[df.Durchmesser == 0.35]

force_keys = ["Hoch", "Mittel", "Gering"]

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))

    for i, label, colour in zip(data, force_keys, colours):
        axs.semilogx(i[0], i[1], label = label, c = colour)
        axs.fill_between(i[0], i[1], 0, alpha = 0.5, color = colour)

    axs.set_xlabel("Frequenzen [Hz]")
    axs.set_ylabel("Amplitude [AE]")
    axs.legend(loc = "upper right", title = "Wirkende Spannkraft", frameon = True)
    axs.grid()

    # Source - https://stackoverflow.com/a/38557425
    # Posted by pms, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-25, License - CC BY-SA 4.0

    inset_axs = inset_axes(axs, width="30%", height=1., loc = "center right", borderpad=1)

    inset_axs.scatter(force_keys, df.trans)
    inset_axs.set_xlabel("Spannkraft")
    inset_axs.set_ylabel("Energie [AE]")
    inset_axs.set_title("Transmittierte Energie $\\bar{A^2}$")
    inset_axs.grid()

    fig.tight_layout()
    
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_force.pdf", format = "pdf", dpi = 900)
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_force.png", format = "png", dpi = 900)
data = []
path = r"C:\Programmieren\Praktikum\GPII\Data\Res"

for i in range(3):
    df = pd.read_csv(path + rf"\Messung_{i + 22}\Res_data_mes.csv", sep = ";")
    data.append([df.freq, df.res])

df = pd.read_csv(r"Data/Res/Res_Trans_Datensatz.csv")
df = df.loc[df.Durchmesser == 0.35]

force_keys = ["Hoch", "Mittel", "Gering"]

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))

    for i, label, colour in zip(data, force_keys, colours):
        """
        # Source - https://stackoverflow.com/a/2566508
        # Posted by unutbu, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-28, License - CC BY-SA 4.0
        idx = (np.abs(i[0] - 2e03)).argmin()

        fit_freq = i[0][:idx]
        fit_res = i[1][:idx]

        # Source - https://stackoverflow.com/a/38431524
        # Posted by strpeter, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-28, License - CC BY-SA 3.0
        mean = sum(fit_freq * fit_res) / sum(fit_freq)
        sigma = np.sqrt(sum(fit_freq * (fit_res - mean)**2) / sum(fit_freq))

        def Gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        popt,pcov = curve_fit(Gauss, fit_freq, fit_res, p0=[max(fit_res), mu, sigma])

        plt.plot(fit_freq, Gauss(fit_freq, *popt), 'r-', label='fit')
        """
        axs.semilogx(i[0], i[1], label = label, c = colour)
        axs.fill_between(i[0], i[1], 0, alpha = 0.5, color = colour)

    axs.set_xlabel("Frequenzen [Hz]")
    axs.set_ylabel("Amplitude [AE]")
    axs.legend(loc = "upper right", title = "Wirkende Spannkraft", frameon = True)
    axs.grid()

    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    handels, labels = axs.get_legend_handles_labels()
    fig.tight_layout()
    
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_force_wo_inset.pdf", format = "pdf", dpi = 900)
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_force_wo_inset.png", format = "png", dpi = 900)

with plt.style.context("science"):
    for i, label, colour in zip(data, force_keys, colours):
        fig, axs = plt.subplots(figsize = (6,4))
        """
        # Source - https://stackoverflow.com/a/2566508
        # Posted by unutbu, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-28, License - CC BY-SA 4.0
        idx = (np.abs(i[0] - 2e03)).argmin()

        fit_freq = i[0][:idx]
        fit_res = i[1][:idx]

        # Source - https://stackoverflow.com/a/38431524
        # Posted by strpeter, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-08-28, License - CC BY-SA 3.0
        mean = sum(fit_freq * fit_res) / sum(fit_freq)
        sigma = np.sqrt(sum(fit_freq * (fit_res - mean)**2) / sum(fit_freq))

        def Gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        popt,pcov = curve_fit(Gauss, fit_freq, fit_res, p0=[max(fit_res), mu, sigma])

        plt.plot(fit_freq, Gauss(fit_freq, *popt), 'r-', label='fit')
        """
        axs.semilogx(i[0], i[1], label = label, c = colour)
        axs.fill_between(i[0], i[1], 0, alpha = 0.5, color = colour)

        axs.set_xlabel("Frequenzen [Hz]")
        axs.set_ylabel("Amplitude [AE]")
        axs.legend(handels, labels, loc = "upper right", title = "Wirkende Spannkraft", frameon = True)
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid()

        fig.tight_layout()

        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\Res_force_wo_inset_{label}.pdf", format = "pdf", dpi = 900)
        fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + rf"\Res_force_wo_inset_{label}.png", format = "png", dpi = 900)


"""
data = []
path = r"C:\Programmieren\Praktikum\GPII\Data\Res"

for i in range(3):
    df = pd.read_csv(path + rf"\Messung_{i + 33}\Res_data_mes.csv", sep = ";")
    data.append([df.freq, df.res])

df = pd.read_csv(r"Data/Res/Res_Trans_Abn_Datensatz.csv")

with plt.style.context("science"):
    fig, axs = plt.subplots(figsize = (6,4))

    for i, j, colour in zip(data, range(len(data)), colours):
        axs.semilogx(i[0], i[1], label = f"{j}ter Durchlauf", c = colour)
        axs.fill_between(i[0], i[1], 0, alpha = 0.5, color = colour)

    axs.set_xlabel("Frequenzen [Hz]")
    axs.set_ylabel("Amplitude [AE]")
    axs.legend(loc = "upper right")
    axs.grid()

    # Source - https://stackoverflow.com/a/38557425
    # Posted by pms, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-25, License - CC BY-SA 4.0

    inset_axs = inset_axes(axs, width="30%", height=1., loc = "center right")

    inset_axs.plot(range(3), df.trans)
    inset_axs.set_xlabel("Durchlauf]")
    inset_axs.set_ylabel("Transmission [AE]")
    inset_axs.set_title("Transmittierte Energie $\\bar{A^2}$")
    inset_axs.grid()

    # Source - https://stackoverflow.com/a/11579834
    # Posted by Chris
    # Retrieved 2026-08-26, License - CC BY-SA 3.0
    inset_axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.tight_layout()
    
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_Abnutzung.pdf", format = "pdf", dpi = 900)
    fig.savefig(fname = r"C:\Programmieren\Praktikum\GPII\Data" + r"\Res_Abnutzung.png", format = "png", dpi = 900)
"""
plt.show()