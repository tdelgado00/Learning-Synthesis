import matplotlib.pyplot as plt
import seaborn as sns
import os
from util import *

import jpype

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

grey = "#d0e1d4"
red = "#ed6a5a"
blue = "#008bf8"
green = "#08a045"
green1 = "#adc178"
green2 = "#045c27"
white = "#ffffff"
yellow = "#ffff00"


def plot_line(value, label, color, ax, fontsize):
    ax.text(5, value, label, {"fontsize": fontsize})
    ax.axhline(y=value, color=color, linestyle="-")


def add_rounded_idx(df, base):
    df["rounded idx"] = df["idx"].apply(lambda idx: idx // base * base if idx < 100 else 100 - base)
    df = df.loc[df["idx"] <= (df["rounded idx"].max() - 1) * base]
    return df


def add_convolution(df, window_size):
    df["mean transitions"] = list(
        np.convolve(list(df["expanded transitions"]), np.ones(window_size), mode='valid') / window_size) + \
                             [np.nan for _ in range(window_size - 1)]
    df["mean transitions / total"] = df["mean transitions"] / df["total transitions"]
    return df


def fillna(df):
    df["expanded transitions"] = df["expanded transitions"].fillna(df["expanded transitions"].max() + 10)
    return df


def onlyifsolvedlast(res):
    for n in range(1, 16):
        for k in range(1, 16):
            if (n > 1 and res[n - 1][k] == float("inf")) or (k > 1 and res[n][k - 1] == float("inf")):
                res[n][k] = float("inf")
    return res


def get_pivot(df, metric="expanded transitions", require_solved_last=True):
    df = fill_df(df, 15)
    df = df.pivot("n", "k", metric)
    df = df.fillna(float("inf"))
    if require_solved_last:
        df = onlyifsolvedlast(df)
    return df


def plot_ra_and_random_trans(data, problem, n, k, ax, fontsize=15):
    ra = get_trans(data["ra 10m"][problem], n, k)
    random_min = min(data["random small"][(problem, n, k)])
    random_mean = np.mean(data["random small"][(problem, n, k)])
    plot_line(-ra, "RA", "red", ax, fontsize)
    plot_line(-random_min, "Random max", "green", ax, fontsize)
    plot_line(-random_mean, "Random mean", "green", ax, fontsize)


def train_transitions_min(used_problems, used_files, data, save_dir):
    print("Saving training transitions")
    with open(save_dir + "train trans.txt", "w+") as f:
        for problem in data["train"].keys():
            f.write(problem + "\n")
            for file in data["train"][problem].keys():
                f.write(file + " " + str(data["train"][problem][file]["expanded transitions"].min()) + "\n")


def get_trans(df, n, k):
    loc = df.loc[(df["n"] == n) & (df["k"] == k)]["expanded transitions"]
    if len(loc) == 0:
        return np.nan
    else:
        assert len(loc) == 1
        return loc.iat[0]


def plot_2_2_3_3(used_problems, used_files, data, path, metric="expanded transitions"):
    used_problems = [p for p in used_problems if p != "CM"]
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))

    for i in range(len(used_problems)):
        problem = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs

        assert len({g for f, g in used_files}) == 1
        dfs = [data["all"][problem][file] for file, group in used_files]
        dfs = [df.loc[(df["n"] == 2) & (df["k"] == 2)].copy() for df in dfs] + \
              [df.loc[(df["n"] == 3) & (df["k"] == 3)].copy() for df in dfs]
        if metric == "mean transitions":
            dfs = [add_convolution(df, 10) for df in dfs]

        df = pd.concat(dfs, ignore_index=True)

        df["reward"] = -df[metric + " / total"]
        sns.lineplot(data=df, x="idx", y="reward", ax=ax, ci="sd", hue="n", palette=["red", "blue"])

        mono_2_2 = data["mono"]["expanded transitions", problem][2][2]
        mono_3_3 = data["mono"]["expanded transitions", problem][3][3]
        ra_2_2 = get_trans(data["ra 10m"][problem], 2, 2)
        plot_line(-ra_2_2 / mono_2_2, "RA 2 2", "red", ax, 15)
        ra_3_3 = get_trans(data["ra 10m"][problem], 3, 3)
        plot_line(-ra_3_3 / mono_3_3, "RA 3 3", "blue", ax, 15)

        ax.set_title(problem, fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    if metric == "mean transitions":
        plt.savefig(path + "mean 2 2 3 3" + ".jpg")
    else:
        plt.savefig(path + "2 2 3 3" + ".jpg")


def plot_test_transitions(used_problems, used_files, data, path, n=2, k=2, metric="expanded transitions", units=False):
    print("Plotting test transitions")
    if n >= 3 or k >= 3:
        used_problems = [p for p in used_problems if p != "CM"]
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))
    fout = open(path + "test trans " + str(n) + " " + str(k) + ".txt", "w+")

    for i in range(len(used_problems)):
        problem = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs

        dfs = [data["all"][problem][file] for file, group in used_files]
        dfs = [df.loc[(df["n"] == n) & (df["k"] == k)].copy() for df in dfs]
        if metric == "mean transitions":
            dfs = [add_convolution(df, 10) for df in dfs]
        df = pd.concat(dfs, ignore_index=True)

        df["reward"] = -df[metric]
        if (units):
            sns.lineplot(data=df, x="idx", y="reward", ax=ax, hue="group", estimator=None, units=True)
        else:
            sns.lineplot(data=df, x="idx", y="reward", ax=ax, ci="sd", hue="group")

        ra = get_trans(data["ra 10m"][problem], n, k)
        plot_line(-ra, "RA", "red", ax, 15)

        fout.write(problem + "\n")
        fout.write("RA " + str(ra) + "\n")
        for file, dfg in df.groupby("file"):
            m = dfg["expanded transitions"].min()
            fout.write(file + " " + str(m) + " " + str(dfg.loc[dfg["expanded transitions"] == m]["idx"].iat[0]) + "\n")

        if (n, k) == (2, 2):
            random_min = min(data["random small"][(problems[i], n, k)])
            random_mean = np.mean(data["random small"][(problems[i], n, k)])
            plot_line(-random_min, "Random max", "green", ax, 15)
            plot_line(-random_mean, "Random mean", "green", ax, 15)

        ax.set_title(" ".join([problem, str(n), str(k)]), fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    u = "units" if units else ""
    if metric == "mean transitions":
        plt.savefig(path + " ".join(["mean", str(n), str(k)]) + u + ".jpg")
    else:
        plt.savefig(path + " ".join([str(n), str(k)]) + u + ".jpg")


# Tabla: Cada approach es una fila, cada instancia es una columna. Una tabla por cada problema.
def transitions_table(used_problems, used_files, data, path):
    print("Saving transitions table")
    print(path)
    instances = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4)]
    fout = open(path + "transitions_table.txt", "w+")
    dfs = []
    groups = list({g for f, g in used_files})
    for i in range(len(used_problems)):
        df = []
        p = used_problems[i]
        for key in data["agent 10m"].keys():
            if key != "all":
                groups += list({g + " " + key for f, g in used_files})

        agent_results = {g: {(n, k): [] for n, k in instances} for g in groups}
        for f, g in used_files:
            results = data["agent 10m"]["all"][p][f].pivot("n", "k", "expanded transitions")
            for n, k in instances:
                agent_results[g][(n, k)].append(results[k][n])
            for key in data["agent 10m"].keys():
                if key != "all":
                    results = data["agent 10m"][key][p][f].pivot("n", "k", "expanded transitions")
                    for n, k in instances:
                        agent_results[g + " " + key][(n, k)].append(results[k][n])
        for g, results in agent_results.items():
            agent_row = {(n, k): results[(n, k)] for n, k in instances}
            agent_row["approach"] = g
            df.append(agent_row)

        ra_row = {(n, k): [get_trans(data["ra 10m"][p], n, k)] for n, k in instances}
        random_row = {(n, k): [get_trans(data["random 10m"][p][i], n, k) for i in range(len(data["random 10m"][p]))]
                      for n, k in instances}
        ra_row["approach"], random_row["approach"] = "RA", "random"

        toNan = lambda x: np.nan if x == float("inf") else x
        mono_row = {(n, k): [toNan(data["mono"]["expanded transitions", p][k][n])] for n, k in instances}
        mono_row["approach"] = "total"

        df += [ra_row, random_row, mono_row]
        df = pd.DataFrame(df)
        df["problem"] = p
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[instances]
    indexes = [(p, ap) for p in used_problems for ap in groups + ["RA", "Random", "Total"]]
    df.index = pd.MultiIndex.from_tuples(indexes)

    # s = df.style.highlight_max(
    #    props='cellcolor:[HTML]{FFFF00}; color:{red};'
    #      'textit:--rwrap; textbf:--rwrap;'
    # )
    def format(x):
        if np.any(np.isnan(x)):
            return "nan"
        if len(x) == 1:
            return "$" + str(x[0]) + "$"
        return "$" + str(np.round(np.mean(x), 2)) + " \pm " + str(np.round(np.std(x), 2)) + "$"

    s = df.style.format(formatter=format)

    fout.write(s.to_latex(position="h", position_float="centering",
                          hrules=True, label="tbl:generalization", caption="Styled LaTeX Table",
                          multirow_align="c", multicol_align="r", clines="skip-last;data") + "\n")

    fout.close()

    df.to_csv(path + "transitions_table.csv")


def plot_loss(used_problems, used_files, data, path):
    print("Plotting loss")
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))

    for i in range(len(used_problems)):
        p = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs
        df = pd.concat(list(data["train"][p].values()), ignore_index=True)
        df.dropna(subset=["loss"])
        sns.lineplot(data=df, x="training steps", y="loss", ax=ax, hue="group", ci="sd", alpha=0.6)

        ax.get_legend().remove()
        # if i != 0:
        #    ax.get_yaxis().set_visible(False)
        ax.set_title(used_problems[i], fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    plt.savefig(path + "/loss.jpg")


def process_training_data(data, used_problems, used_files, base=10000, window_size=10):
    print("Processing training data")
    normalized = False
    for problem in used_problems:
        data["bucket train"][problem] = {}
        for file, group in used_files:
            if problem not in data["train"].keys() or not file in data["train"][problem].keys():
                continue
            big_df = data["train"][problem][file]

            big_df["training steps"] = big_df["training steps"] // base * base

            grouped_df = big_df.groupby("training steps")
            df = pd.DataFrame({"training steps": grouped_df["training steps"].first()})
            df["expanded transitions"] = grouped_df["expanded transitions"].mean()

            # in each group problem, n and k should be constant
            df["n"] = grouped_df["n"].first()
            df["k"] = grouped_df["k"].first()
            df["problem"] = grouped_df["problem"].first()

            df["total transitions"] = df.apply(
                lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]],
                axis=1)

            df["min transitions"] = df["expanded transitions"].cummin()
            df["group"] = big_df["group"].iat[0]
            df["file"] = big_df["file"].iat[0]

            if normalized:
                df['norm transitions'] = df.groupby('total transitions')["expanded transitions"].apply(
                    lambda x: (x - x.mean()) / x.std())
                df["mean transitions"] = list(
                    np.convolve(list(df["norm transitions"]), np.ones(window_size), mode='valid') / window_size) + [
                                             np.nan for _ in range(window_size - 1)]
            else:
                df["mean transitions"] = list(
                    np.convolve(list(df["expanded transitions"]), np.ones(window_size), mode='valid') / window_size) + \
                                         [np.nan for _ in range(window_size - 1)]
            data["bucket train"][problem][file] = df


def plot_training_transitions(used_problems, used_files, data, path):
    print("Plotting training transitions")
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))

    for i in range(len(used_problems)):
        p = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs
        max_step = min([data["bucket train"][p][f]["training steps"].max() for f, g in used_files])
        df = pd.concat(list(data["bucket train"][p].values()), ignore_index=True)

        # df["reward"] = -df["min transitions"]
        df["reward"] = -df["mean transitions"]

        print("Plotting", p)

        # plotting units
        sns.lineplot(data=df, x="training steps", y="reward", ax=ax, hue="group", ci=None, alpha=0.5, estimator=None,
                     units="file", linewidth=1.0, legend=False, color="blue")

        # plotting average
        sns.lineplot(data=df.loc[df["training steps"] <= max_step], x="training steps", y="reward", ax=ax, hue="group",
                     ci=None, linewidth=2, alpha=1.0, color="blue")

        # sns.lineplot(data=df, x="training steps", y="reward", ax=ax, hue="file", alpha=0.7, ci=None)

        plot_ra_and_random_trans(data, p, 2, 2, ax, fontsize=16)

        if i > 0:
            axs[i].set_ylabel("")
        else:
            axs[i].set_ylabel("expanded transitions", fontdict={"size": 18})
        axs[i].set_xlabel("training steps", fontdict={"size":17})
        ax.get_legend().remove()
        ax.set_title(used_problems[i], fontdict={"fontsize": 20})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 16}, loc="lower right")
    plt.tight_layout()
    plt.savefig(path + "/training.jpg")

def plot_scatter_generalization(used_problems, used_files, data, path):
    # quiero un dataframe que tenga una columna de instancia, otra de tamaño, otra de expansiones resueltas y otra de heurística
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))
    for i in range(len(used_problems)):
        p = used_problems[i]
        print(p)
        df = pd.concat(data["agent 10m"]["all"][p].values(), ignore_index=True)
        df["heuristic"] = "RL"
        data["ra 10m"][p]["heuristic"] = "RA"
        df = pd.concat([df, data["ra 10m"][p]], ignore_index=True)
        for dfr in data["random 10m"][p]:
            dfr["heuristic"] = "Random"
        df = pd.concat([df]+data["random 10m"][p], ignore_index=True)
        
        to_append = []
        for h in ["Random", "RL", "RA"]:
            for n in range(1, 16):
                for k in range(1, 16):
                    if len(df.loc[(df["n"] == n) & (df["k"] == k) & (df["heuristic"] == h)]) == 0:
                        to_append.append({"problem": p, "n": n, "k": k, "expanded transitions": float("inf"), "heuristic": h})
        df = pd.concat([df, pd.DataFrame(to_append)], ignore_index=True)
        
        df["total transitions"] = df.apply(
                    lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]],
                    axis=1)
        
        #df = df.loc[(df["n"] > 1) & (df["k"] > 1)]
        
        df = df.loc[df["total transitions"] != float("inf")]
        #df = df.loc[df["synthesis time(ms)"] < 5000]
        total_trans = sorted(list(df["total transitions"].unique()))
        #np.random.shuffle(total_trans)
        df["instance"] = df.apply(lambda r: total_trans.index(r["total transitions"]), axis=1)
        #print(p, sorted(list(df["instance"].unique())))
        
        df["expanded transitions / total"] = df["expanded transitions"] / df["total transitions"]
        df["log rel trans"] = np.log(df["expanded transitions"])
        metric = "expanded transitions / total"
        metric = "log rel trans"
        metric = "expanded transitions"
        
        #print(np.max(df.loc[df[metric]<float("inf")][metric]))
        df[metric] = df.apply(lambda r: r[metric] if (not np.isnan(r[metric]) and r[metric] != float("inf")) else np.max(df.loc[df[metric]<float("inf")][metric])*5, axis=1)
        #print(df.loc[df["instance"] == 7][metric])
        axs[i].set_title(p, fontdict={"fontsize": 20})
        sns.scatterplot(data=df, x="total transitions", y=metric, hue="heuristic", 
                        style="heuristic", markers=["o", "P", "X"], alpha=0.7, ax=axs[i], s=100)
        axs[i].get_legend().remove()
        handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
        axs[0].legend(handles, labels, prop={'size': 16}, loc="upper left")
        axs[i].set_yscale("log")
        axs[i].set_xscale("log")
        if i > 0:
            axs[i].set_ylabel("")
        else:
            axs[i].set_ylabel("expanded transitions", fontdict={"size": 18})
        axs[i].set_xlabel("instance size", fontdict={"size":17})
        #axs[i].set_yticks(10**axs[i].get_yticks())
        #x = df.loc[df["expanded transitions"] == float("inf")]
        #print(x)
        #plt.plot(, [4], marker="*", ls="none", ms=20)
        df["inst-n-k"] = df.apply(lambda r: (r["instance"], r["n"], r["k"]), axis=1)
        #for x in sorted(list(df["inst-n-k"].unique())):
        #    print(x)
        print(df["expanded transitions"].max())
        
        
    plt.tight_layout()
    plt.savefig(path + "scatter.jpg", dpi=500)


def plot_solved_training(used_problems, used_files, data, path):
    print("Plot solved training")

    def get_df_solved(problem, file, group):
        df = data["all"][problem][file]

        df_solved = []
        for x, cant in dict(df["idx"].value_counts()).items():
            df_solved.append({"idx": x, "solved": cant})
        df_solved = pd.DataFrame(df_solved)
        df_solved.sort_values(by="idx", inplace=True)

        df_solved["file"] = file
        df_solved["group"] = group

        window_size = 10
        df_solved["max solved"] = df_solved["solved"].cummax()
        df_solved["mean solved"] = list(
            np.convolve(list(df_solved["solved"]), np.ones(window_size), mode='valid') / window_size) + \
                                   [np.nan for _ in range(window_size - 1)]
        return df_solved

    df_solved = {p: pd.concat([get_df_solved(p, f, g) for f, g in used_files], ignore_index=True) for p in
                 used_problems}
    solved_ra = {p: len(list(data["ra 5s"][p]["expanded transitions"].dropna())) for p in used_problems}
    solved_random = {p: len(list(data["random 5s"][p]["expanded transitions"].dropna())) / 20 for p in used_problems}

    fout = open(path + "/solved max.txt", "w+")
    for problem in used_problems:
        fout.write(problem + "\n")

        fout.write("RA " + str(solved_ra[problem]) + "\n")
        for file, dfg in df_solved[problem].groupby("file"):
            fout.write(file + " " + str(dfg["solved"].max()) + " at idx: " + str(dfg["solved"].argmax()) + "\n")
    fout.close()

    for metric in ["solved", "mean solved"]:
        f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))
        for i in range(len(used_problems)):
            problem = used_problems[i]
            ax = axs[i] if len(used_problems) > 1 else axs

            plot_line(solved_ra[problem], "RA", "red", ax, 15)
            plot_line(solved_random[problem], "Random", "green", ax, 15)

            # sns.lineplot(data=df_solved[problem], x="idx", y="solved", ax=ax, estimator=None, units="file", ci="sd",
            #             hue="group")
            df = df_solved[problem]
            
            # plotting units
            sns.lineplot(data=df, x="idx", y=metric, ax=ax, hue="group", ci="None", alpha=0.5, estimator=None,
                        units="file", linewidth=1.0, legend=False)
            # sns.lineplot(data=df, x="idx", y=metric, ax=ax, ci="sd", hue="group")

            # plotting average
            #sns.lineplot(data=df, x="idx", y=metric, ax=ax, hue="group",
            #            ci=None, linewidth=2, alpha=1.0, color="blue")

            ax.set_title(problem)

        plt.tight_layout()
        plt.savefig(path + metric + " training.jpg", dpi=500)


def plot_15_15(used_problems, path, files1, file2, figure_name, name, require_solved_last=True):
    print("Plotting 15 15", files1, file2)
    vmin = -3
    vmax = 3

    def get_solved_value(a, r):
        # Possibilities:
        # 0. not solved
        # 1. not solved by the agent and solved by RA
        # 2. solved sometimes by the agent and not solved by RA
        # 3. solved sometimes by the agent and solved by RA
        # 4. solved always by the agent and not solved by RA
        # 5. solved always by the agent and solved by RA

        solved_ra = r != float("inf")
        solved_agent = [ai != float("inf") for ai in a]
        if not np.any(solved_agent):
            return 0 if not solved_ra else 1
        elif np.all(solved_agent):
            return 4 if not solved_ra else 5
        else:
            return 2 if not solved_ra else 3

    def get_trans_rel_solved(a, r):
        if r == float("inf") or np.any([ai == float("inf") for ai in a]):
            return np.nan
        else:
            return np.mean([np.log2(ai / r) for ai in a])

    def get_df(dfs1, df2, func):
        m = np.zeros(shape=(15, 15))
        for n in range(15):
            for k in range(15):
                r = df2[k + 1][n + 1]
                a = [df[k + 1][n + 1] for df in dfs1]
                # if (n, k) == (13, 13):
                #     print(r, a)
                # if (n, k) == (14, 14):
                #     print(r, a)
                m[n, k] = func(a, r)
        return m

    trans_df = {}
    solved_df = {}
    for i in range(len(used_problems)):
        p = used_problems[i]
        dfs1 = [get_pivot(data["agent 10m"][name][p][f], require_solved_last=require_solved_last) for f in files1]
        if file2 == "ra":
            df2 = get_pivot(data["ra 10m"][p], require_solved_last=require_solved_last)
        elif file2 == "random":
            df2 = get_pivot(data["random 10m"][p][0], require_solved_last=require_solved_last)
        elif file2 == "mono":
            df2 = data["mono"]["expanded transitions", p]
        else:
            df2 = get_pivot(data["agent 10m"][name][p][file2], require_solved_last=require_solved_last)

        trans_df[p] = get_df(dfs1, df2, get_trans_rel_solved)
        solved_df[p] = get_df(dfs1, df2, get_solved_value)

    for trans in [False, True]:
        f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))
        for i in range(len(used_problems)):
            p = used_problems[i]
            ax = axs[i] if len(used_problems) > 1 else axs

            # annot_df = pd.DataFrame(solved_df[p])
            cmap = "coolwarm" if trans else [white, red, green2, yellow, green, grey]

            sns.heatmap(data=trans_df[p] if trans else solved_df[p],
                        cmap=cmap,
                        annot=trans, annot_kws={"size": 6},
                        xticklabels=list(range(1, 16)),
                        yticklabels=list(range(1, 16)),
                        ax=ax, cbar=True, vmin=vmin if trans else 0, vmax=vmax if trans else 5)

            # for problem, n, k in train_instances(p):
            #                 ax.text(k - 0.5, n - 0.5, "X",
            #                         horizontalalignment='center',
            #                         verticalalignment='center',
            #                         )

            ax.invert_yaxis()
            ax.set_title(p, fontsize=30)

        plt.tight_layout()
        plt.savefig(path + "15 15 " + figure_name + (" trans" if trans else " solved") + ".jpg", dpi=200)


def solved_table(used_problems, used_files, data, path, add_mono=False, long_timeout=False):
    print("Writing solved table")

    def solved_metric(df, factor=1):
        return len(df["expanded transitions"].dropna()) / factor

    agent_data = "agent 10m" if not long_timeout else "agent 30m"
    ra_data = "ra 10m" if not long_timeout else "ra 30m"

    metric = solved_metric

    groups = list({g for f, g in used_files})

    for key in data[agent_data].keys():
        if key != "all":
            groups += list({g + " " + key for f, g in used_files})

    groups += ["random", "ra"]
    if add_mono:
        groups += ["mono"]

    problem_columns = used_problems + ["all"]
    if "multiple" in groups:
        problem_columns.append("all (AT, BW, DP, TA)")
    results = {p: {ap: [] for ap in groups} for p in problem_columns}

    for i in range(len(used_problems)):
        p = problems[i]
        for file, group in used_files:
            if group == "multiple" and (p not in ["AT", "TA", "BW", "DP"]):
                continue
            df = data[agent_data]["all"][p][file]
            results[p][group].append(metric(df))

            for key in data[agent_data].keys():
                if key != "all":
                    df = data[agent_data][key][p][file]
                    results[p][group + " " + key].append(metric(df))

        results[p]["random"] += [metric(data["random 10m"][p][i]) for i in range(len(data["random 10m"][p]))]
        results[p]["ra"].append(metric(data[ra_data][p]))
        if add_mono:
            assert metric == solved_metric
            mono_solved = np.sum(
                [data["mono"]["expanded transitions", p][k][n] != float("inf") for n in range(1, 16) for k in
                 range(1, 16)])
            results[p]["mono"].append(mono_solved)

    for group in groups:
        n = len(results[used_problems[0]][group])
        if group != "multiple":
            results["all"][group] = [0 for j in range(n)]
            for j in range(n):
                for problem in problems:
                    results["all"][group][j] += results[problem][group][j]

        if "multiple" in groups:
            results["all (AT, BW, DP, TA)"][group] = [0 for j in range(n)]
            for j in range(n):
                for problem in ["AT", "BW", "DP", "TA"]:
                    results["all (AT, BW, DP, TA)"][group][j] += results[problem][group][j]

    # with open(path + "ttest_results.txt", "w+") as f:
    #     for problem in problems + ["all"]:
    #         f.write(problem + "\n")
    #         random = results[problem]["random"][0]
    #         ra = results[problem]["ra"][0] / random
    #         agent = np.array(results[problem]["RL"]) / random
    #         f.write(str(np.round(np.mean(agent), 2)) + " ")
    #         f.write(str(np.round(ra, 2)) + " ")
    #         f.write(str(ttest_1samp(agent - ra, 0, alternative="two-sided").pvalue) + "\n")

    rows = [{"approach": ap} for ap in groups]
    for problem in problem_columns:
        for j in range(len(groups)):
            r = results[problem][groups[j]]
            # print(problem, groups[j], r)
            if len(r) == 0:
                continue
            # r = np.array(r) / results[problem]["random"][0]
            mean = np.round(np.mean(r), 2)
            std = np.round(np.std(r), 2)
            if std > 0.0001:
                rows[j][problem] = str(mean) + " ± " + str(std)
            else:
                rows[j][problem] = str(mean)

    dft = pd.DataFrame(rows)

    dft["approach"] = dft["approach"].apply(lambda a: " ".join(a.split("_")))

    with open(path + "latex_table.txt", "w+") as f:

        f.write(dft.style.hide(axis="index").to_latex(
            column_format="llllllll", position="h", position_float="centering",
            hrules=True, label="tbl:generalization", caption="Styled LaTeX Table",
            multirow_align="t", multicol_align="r") + "\n")
        for problem in problem_columns:
            f.write(problem + "\n")
            for group in groups:
                f.write(group + " " + str(results[problem][group]) + "\n")


def pipeline_plot_15_15(files1, file2, figure_name, name, require_solved_last):
    return lambda used_problems, used_files, data, path: plot_15_15(used_problems, path, files1, file2, figure_name,
                                                                    name, require_solved_last)


if __name__ == "__main__":
    path = "experiments/figures/paper/"
    if not os.path.exists(path):
        print("Creating dir", path)
        os.makedirs(path)

    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]

    focused_files = ["focused_1", "focused_2", "focused_3", "focused_4", "focused_5", "focused_6", "focused_7"]
    epsdec = ["pytorch_epsdec", "epsdec_2", "epsdec_3", "epsdec_4"]
    ffiles = ["boolean", "boolean_2", "boolean_3", "boolean_4", "boolean_5"]
    # at33 = ["boolean33", "boolean33 2", "boolean33 3", "boolean33 4"]

    files = []
    files += [(f, "RL") for f in ffiles]# + [(f, "33") for f in at33]
    # files += [(f, "epsdec") for f in epsdec]
    # files += [(f, "focused") for f in focused_files]

    data = {}
    data["mono"] = read_monolithic()
    read_ra_and_random(problems, data)
    read_test(data, problems, files)
    read_training(data, problems, files)
    process_training_data(data, problems, files, base=5000, window_size=10)
    data["random small"] = read_random_small()

    data["agent 10m"], data["agent 30m"] = {}, {}
    data["agent 10m"]["all"] = read_agents_10m(problems, files)
    #data["agent 10m"]["best22"] = read_agents_10m(problems, files, "all_best22")
    # data["agent 30m"]["all"] = read_agents_10m(problems, files, "all30m")

    under_test = "boolean33"

    pipeline = [
        #lambda p, f, d, pth: solved_table(p, f, d, pth, add_mono=False),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=2, k=2, metric="expanded transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=2, k=2, metric="mean transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=3, k=3, metric="expanded transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=3, k=3, metric="mean transitions"),
        # plot_2_2_3_3,
        # lambda p, f, d, pth: plot_2_2_3_3(p, f, d, pth, metric="mean transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=4, k=4, metric="expanded transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=4, k=4, metric="mean transitions"),
        #train_transitions_min,
        plot_scatter_generalization,
        #plot_training_transitions,
        #transitions_table,
        #lambda p, f, d, pth: solved_table(p, f, d, pth, long_timeout=False),
        #plot_solved_training,
        #plot_loss,
        # pipeline_plot_15_15(ffiles, "ra", "RL vs RA 30m", "all30m", require_solved_last=False),
        # pipeline_plot_15_15(ffiles, "random", "RL vs Random mean"),
        # pipeline_plot_15_15(ffiles, "mono", "RL vs mono"),
        #pipeline_plot_15_15(ffiles, under_test, "baseline vs under test", "all", require_solved_last=True),
        #pipeline_plot_15_15([under_test], "ra", "under test vs ra", "all", require_solved_last=True),
        # pipeline_plot_15_15([under_test], "random", "under test vs random")
    ]

    for e in pipeline:
        e(problems, files, data, path)
