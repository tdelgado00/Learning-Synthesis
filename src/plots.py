import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import pickle
from util import *

import jpype

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

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


def fill_df(df, m):
    added = []
    for n in range(1, m + 1):
        for k in range(1, m + 1):
            if len(df.loc[(df["n"] == n) & (df["k"] == k)]) == 0:
                added.append({"n": n, "k": k, "expanded transitions": float("inf"), "synthesis time(ms)": float("inf")})
    df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return df


def onlyifsolvedlast(res):
    for n in range(1, 16):
        for k in range(1, 16):
            if (n > 1 and res[n - 1][k] == float("inf")) or (k > 1 and res[n][k - 1] == float("inf")):
                res[n][k] = float("inf")
    return res


def get_pivot(df, metric="expanded transitions"):
    df = fill_df(df, 15)
    df = df.pivot("n", "k", metric)
    df = df.fillna(float("inf"))
    df = onlyifsolvedlast(df)
    return df


def plot_ra_and_random_trans(data, problem, n, k, ax):
    ra = get_trans(data["ra 10m"][problem], n, k)
    random_min = min(data["random small"][(problem, n, k)])
    random_mean = np.mean(data["random small"][(problem, n, k)])
    plot_line(-ra, "RA", "red", ax, 15)
    plot_line(-random_min, "Random max", "green", ax, 15)
    plot_line(-random_mean, "Random mean", "green", ax, 15)


def read_test(data, problems, files):
    print("Reading test files")

    data["all"] = {}

    old_files = ["base_features_2h", "ra_feature2opt_2h", "5mill_RA", "5mill_L"]

    for p in problems:
        data["all"][p] = {}
        for file, group in files:
            if file in old_files:
                path = results_path(p) + file + "/" + filename([p, 2, 2]) + ".csv"
            else:
                path = results_path(p) + file + "/generalization_all.csv"
            try:
                df = pd.read_csv(path)
            except:
                print("File not found", path)
                continue
            df = df.dropna(subset=["expanded transitions"])
            df["instance"] = df.apply((lambda r: (r["problem"], r["n"], r["k"])), axis=1)
            df["total transitions"] = df.apply(
                lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]], axis=1)
            df["group"] = group
            df["file"] = file
            df["training steps"] = df["idx"] * 50000
            df["expanded transitions / total"] = df["expanded transitions"] / df["total transitions"]
            df["min transitions"] = df.groupby("instance")["expanded transitions"].cummin()
            data["all"][p][file] = df


def read_ra_and_random(used_problems, data):
    data.update({name: {} for name in ["ra 5s", "random 5s", "ra 10m", "random 10m"]})
    for p in used_problems:
        data["ra 5s"][p] = pd.read_csv(results_path(p) + "/RA_5s_15.csv")
        data["random 5s"][p] = pd.read_csv(results_path(p) + "/random_5s.csv")

        data["ra 10m"][p] = pd.read_csv(results_path(p) + "/all_ra_afterfix_15.csv")
        data["random 10m"][p] = pd.read_csv(results_path(p) + "/all_random.csv")

        for df in [data[name][p] for name in ["ra 5s", "random 5s", "ra 10m", "random 10m"]]:
            df["instance"] = df.apply(lambda r: (r["problem"], r["n"], r["k"]), axis=1)
            df["total transitions"] = df.apply(
                lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]], axis=1)
            df.dropna(subset=["expanded transitions"])


def read_training_for_file(problem, file, multiple):
    if file == "5mill_JE_NORA":  # we didn't save training data
        return None
    with open(results_path(problem, file=file) + "/training_data.pkl", "rb") as f:
        training_data, agent_params, env_params = pickle.load(f)
    df = pd.DataFrame(training_data)
    df["idx"] = df["training steps"] // 10000
    df["file"] = file
    df["problem"] = problem
    df["multiple"] = multiple
    if multiple:
        instances = train_instances(problem)
        df["n"] = df.apply(lambda r: instances[int(r["idx"]) % len(instances)][1], axis=1)
        df["k"] = df.apply(lambda r: instances[int(r["idx"]) % len(instances)][2], axis=1)
    else:
        df["n"] = 2
        df["k"] = 2
    return df


def read_training(data, used_problems, used_files):
    print("Reading training")
    data.update({"train": {}, "bucket train": {}})
    for problem in used_problems:
        data["train"][problem] = {}
        for file, group in used_files:
            multiple = "RR" in group
            df = read_training_for_file(problem, file, multiple)
            if df is None:
                continue
            df["group"] = group
            data["train"][problem][file] = df

    normalized = False
    for problem in used_problems:
        data["bucket train"][problem] = {}
        for file, group in used_files:
            if problem not in data["train"].keys() or not file in data["train"][problem].keys():
                continue
            big_df = data["train"][problem][file]

            base = 10000
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
            window_size = 10

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


def read_random_small():
    random_results_small = {}
    for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
        for n, k in [(2, 2), (3, 3)]:
            df = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/random.csv")
            random_results_small[(problem, n, k)] = list(df["expanded transitions"])
    return random_results_small


def read_monolithic():
    monolithic_results = {}
    for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
        df = pd.read_csv("experiments/results/ResultsPaper/" + problem + ".csv")
        df = df.loc[df["controllerType"] == "mono"]
        df["n"] = df["testcase"].apply(lambda t: int(t.split("-")[1]))
        df["k"] = df["testcase"].apply(lambda t: int(t.split("-")[2]))
        df = fill_df(df, 15)
        monolithic_results["expanded transitions", problem] = df.pivot("n", "k", "expandedTransitions")
        monolithic_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesisTimeMs")
    return monolithic_results


def train_transitions_min(used_problems, used_files, data, save_dir):
    print("Saving training transitions")
    with open(save_dir + "train trans.txt", "w+") as f:
        for problem in data["train"].keys():
            f.write(problem + "\n")
            for file in data["train"][problem].keys():
                f.write(file + " " + str(data["train"][problem][file]["expanded transitions"].min()) + "\n")


def read_agents_10m(problems, files, name="all"):
    agents_10m = {}
    for p in problems:
        agents_10m[p] = {}
        for file, g in files:
            agents_10m[p][file] = pd.read_csv(results_path(p, file=file) + "/"+name+".csv")
    return agents_10m


def get_trans(df, n, k):
    loc = df.loc[(df["n"] == n) & (df["k"] == k)]["expanded transitions"]
    if len(loc) == 0:
        return np.nan
    else:
        assert len(loc) == 1
        return loc.iat[0]


def plot_test_transitions(used_problems, used_files, data, path, n=2, k=2, metric="expanded transitions"):
    print("Plotting test transitions")
    if n >= 3 or k >= 3:
        used_problems = [p for p in used_problems if p != "CM"]
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))
    fout = open(path + "test trans.txt", "w+")

    for i in range(len(used_problems)):
        problem = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs

        dfs = [data["all"][problem][file] for file, group in used_files]
        dfs = [df.loc[(df["n"] == n) & (df["k"] == k)].copy() for df in dfs]
        if metric == "mean transitions":
            dfs = [add_convolution(df, 10) for df in dfs]
        df = pd.concat(dfs, ignore_index=True)

        df["reward"] = -df[metric]

        sns.lineplot(data=df, x="training steps", y="reward", ax=ax, ci="sd", hue="group")

        ra = get_trans(data["ra 10m"][problem], n, k)
        plot_line(-ra, "RA", "red", ax, 15)

        fout.write(problem + "\n")
        fout.write("RA " + str(ra) + "\n")
        for file, dfg in df.groupby("file"):
            fout.write(file + " " + str(dfg["expanded transitions"].min()) + "\n")

        if (n, k) in [(2, 2), (3, 3)]:
            random_min = min(data["random small"][(problems[i], n, k)])
            random_mean = np.mean(data["random small"][(problems[i], n, k)])
            plot_line(-random_min, "Random max", "green", ax, 15)
            plot_line(-random_mean, "Random mean", "green", ax, 15)

        ax.set_title(" ".join([problem, str(n), str(k)]), fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    if metric == "mean transitions":
        plt.savefig(path + " ".join(["mean", str(n), str(k)]) + ".jpg")
    else:
        plt.savefig(path + " ".join([str(n), str(k)]) + ".jpg")


# Tabla: Cada approach es una fila, cada instancia es una columna. Una tabla por cada problema.
def save_transitions_table(used_problems, used_files, data, path):
    print("Saving transitions table")
    print(path)
    instances = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4)]
    fout = open(path+"transitions_table.txt", "w+")
    for i in range(len(used_problems)):
        df = []
        p = used_problems[i]
        for f, g in used_files:
            results = data["agent 10m"][p][f].pivot("n", "k", "expanded transitions")
            agent_row = {(n, k): results[k][n] for n, k in instances}
            agent_row["approach"] = g
            df.append(agent_row)
                
        ra_row = {(n, k): get_trans(data["ra 10m"][p], n, k) for n, k in instances}
        random_row = {(n, k): get_trans(data["random 10m"][p], n, k) for n, k in instances} # todo: esto debería estar ejecutado muchas veces
        ra_row["approach"], random_row["approach"] = "RA", "random"
        df += [ra_row, random_row]
        df = pd.DataFrame(df)
        df = df.groupby("approach").mean()
        fout.write(p + "\n")
        fout.write(df.style.format(precision=2).to_latex() + "\n")
    fout.close()


def plot_loss(used_problems, used_files, data, path):
    print("Plotting loss")
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))

    for i in range(len(used_problems)):
        p = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs
        df = pd.concat(list(data["train"][p].values()), ignore_index=True)
        df.dropna(subset=["loss"])
        print(df.columns)
        sns.lineplot(data=df, x="training steps", y="loss", ax=ax, hue="group", ci="sd", alpha=0.6)

        ax.get_legend().remove()
        # if i != 0:
        #    ax.get_yaxis().set_visible(False)
        ax.set_title(used_problems[i], fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    plt.savefig(path + "/loss.jpg")


def plot_training_transitions(used_problems, used_files, data, path):
    print("Plotting training transitions")
    f, axs = plt.subplots(1, len(used_problems), figsize=(5 * len(used_problems), 6))

    for i in range(len(used_problems)):
        p = used_problems[i]
        ax = axs[i] if len(used_problems) > 1 else axs
        df = pd.concat(list(data["bucket train"][p].values()), ignore_index=True)

        # df["reward"] = -df["min transitions"]
        df["reward"] = -df["mean transitions"]
        # df = df.loc[(df["n"] == 3) & (df["k"] == 3)]

        print("Plotting", p)
        # sns.lineplot(data=df, x="training steps", y="reward", ax=ax, hue="group", ci="sd", alpha=0.6)

        # print(df.groupby("group")["reward"].max())
        sns.lineplot(data=df, x="training steps", y="reward", ax=ax, hue="group", ci="sd", alpha=0.6)

        plot_ra_and_random_trans(data, p, 2, 2, ax)

        ax.get_legend().remove()
        # if i != 0:
        #    ax.get_yaxis().set_visible(False)
        ax.set_title(used_problems[i], fontdict={"fontsize": 18})

    handles, labels = (axs[0] if len(used_problems) > 1 else axs).get_legend_handles_labels()
    plt.legend(handles, labels, prop={'size': 15}, loc="lower right")
    plt.tight_layout()
    plt.savefig(path + "/training.jpg")


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
            fout.write(file + " " + str(dfg["solved"].max()))
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

            sns.lineplot(data=df_solved[problem], x="idx", y=metric, ax=ax, ci="sd", hue="group")

            ax.set_title(problem)

        plt.tight_layout()
        plt.savefig(path + metric + " training.jpg", dpi=500)


def plot_15_15(used_problems, path, files1, file2, name):
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
                m[n, k] = func(a, r)
        return m

    trans_df = {}
    solved_df = {}
    for i in range(len(used_problems)):
        p = used_problems[i]
        dfs1 = [get_pivot(data["agent 10m"][p][f]) for f in files1]
        if file2 == "ra":
            df2 = get_pivot(data["ra 10m"][p])
        elif file2 == "random":
            df2 = get_pivot(data["random 10m"][p])
        else:
            df2 = get_pivot(data["agent 10m"][p][file2])

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
        plt.savefig(path + "15 15 " + name + (" trans" if trans else " solved") + ".jpg", dpi=200)


def mean_trans_table(used_problems, used_files, data, path):
    def mean_trans_metric(df):
        dfl = df.loc[df["instance"].isin(instances[df["problem"].iloc[0]])]
        return (dfl["expanded transitions"] / dfl["total transitions"]).mean()

    instances = {}
    for p in problems:
        agent_files = [(data["all"][p][f], 100) for f, g in used_files if f in data["all"][p].keys()]
        instances[p] = all_solved_instances(agent_files +
                                            [(data["ra 5s"][p], 1)] +
                                            [(data["random 10m"][p], 20)])
        instances[p] = [(p, n, k) for n, k in instances[p]]


def solved_table(used_problems, used_files, data, path):
    print("Writing solved table")
    def solved_metric(df, factor=1):
        return len(df["expanded transitions"].dropna()) / factor

    timeout = "10m"
    metric = solved_metric

    groups = list({g for f, g in used_files}.union({"random", "ra"}))
    results = {p: {ap: [] for ap in groups} for p in used_problems + ["all", "all (AT, BW, DP, TA)"]}

    for i in range(len(used_problems)):
        p = problems[i]
        for file, group in used_files:
            if group == "multiple" and (p not in ["AT", "TA", "BW", "DP"]):
                continue
            if timeout == "5s":
                df = data["all"][p][file]
                best_idx = best_generalization_agent(p, file)
                df = df.loc[df["idx"] == best_idx]
            else:
                df = data["agent 10m"][p][file]
            results[p][group].append(metric(df))

        if timeout == "10m":
            results[p]["random"].append(metric(data["random 10m"][p]))
            results[p]["ra"].append(metric(data["ra 10m"][p]))
        else:
            results[p]["random"].append(metric(data["random 10m"][p], factor=20))
            results[p]["ra"].append(metric(data["ra 10m"][p]))

    for group in groups:
        n = len(results[used_problems[0]][group])
        if group != "multiple":
            results["all"][group] = [0 for j in range(n)]
            for j in range(n):
                for problem in problems:
                    results["all"][group][j] += results[problem][group][j]

        results["all (AT, BW, DP, TA)"][group] = [0 for j in range(n)]
        for j in range(n):
            for problem in ["AT", "BW", "DP", "TA"]:
                results["all (AT, BW, DP, TA)"][group][j] += results[problem][group][j]

    with open(path + "ttest_results.txt", "w+") as f:
        for problem in problems + ["all"]:
            f.write(problem + "\n")
            random = results[problem]["random"][0]
            ra = results[problem]["ra"][0] / random
            agent = np.array(results[problem]["RL"]) / random
            f.write(str(np.round(np.mean(agent), 2)) + " ")
            f.write(str(np.round(ra, 2)) + " ")
            f.write(str(ttest_1samp(agent - ra, 0, alternative="two-sided").pvalue) + "\n")

    rows = [{"approach": ap} for ap in groups]
    for problem in problems + ["all", "all (AT, BW, DP, TA)"]:
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

    with open(path + "latex_table.txt", "w+") as f:
        f.write(dft.style.to_latex() + "\n")
        for problem in problems + ["all", "all (AT, BW, DP, TA)"]:
            f.write(problem + "\n")
            for group in groups:
                f.write(group + " " + str(results[problem][group]) + "\n")


def pipeline_plot_15_15(files1, file2, name):
    return lambda used_problems, used_files, data, path: plot_15_15(used_problems, path, files1, file2, name)


if __name__ == "__main__":
    path = "experiments/figures/paper/"
    if not os.path.exists(path):
        print("Creating dir", path)
        os.makedirs(path)

    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]

    focused_files = ["focused_1", "focused_2", "focused_3", "focused_4", "focused_5", "focused_6", "focused_7"]
    ffiles = ["pytorch_epsdec", "epsdec_2", "epsdec_3", "epsdec_4"]
    
    files = []
    files += [(f, "RL") for f in ffiles]
    files += [(f, "focused") for f in focused_files]
    #files += [("epsdec_1kk", "epsdec 1kk")]
    #files = [("epsdec_1kk", "epsdec 1kk"), ("huber", "huber")]
    #files = [("huber", "huber")]
    
    data = {}
    data["mono"] = read_monolithic()
    read_ra_and_random(problems, data)
    read_test(data, problems, files)
    read_training(data, problems, files)
    data["random small"] = read_random_small()
    data["agent 10m"] = read_agents_10m(problems, files)

    under_test = "huber"

    pipeline = [
        solved_table,
        lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=2, k=2, metric="expanded transitions"),
        lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=2, k=2, metric="mean transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=3, k=3, metric="expanded transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=3, k=3, metric="mean transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=4, k=4, metric="expanded transitions"),
        #lambda p, f, d, pth: plot_test_transitions(p, f, d, pth, n=4, k=4, metric="mean transitions"),
        train_transitions_min,
        plot_training_transitions,
        save_transitions_table,
        plot_solved_training,
        plot_loss,
        pipeline_plot_15_15(ffiles, "ra", "RL vs RA"),
        pipeline_plot_15_15(ffiles, "random", "RL vs Random"),
        #pipeline_plot_15_15(ffiles, under_test, "baseline vs under test"),
        #pipeline_plot_15_15([under_test], "ra", "under test vs ra"),
        #pipeline_plot_15_15([under_test], "random", "under test vs random")
    ]

    for e in pipeline:
        e(problems, files, data, path)

