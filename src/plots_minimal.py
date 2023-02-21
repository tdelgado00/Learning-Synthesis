import pandas as pd
from util import *
import matplotlib.pyplot as plt
import seaborn as sns

class selectionPreprocessingData():
    def __init__(self, abs_path):
        print("Yet to be implemented")
        self.df = pd.read_csv(abs_path)

class selectedAgentData():
    def __init__(self, abs_path):
        print("Yet to be implemented")
        self.df = pd.read_csv(abs_path)

def get_solved_series(filename, problems):

    solved_by_problem = pd.Series(index = problems ,dtype = float)
    insert_solved_instances_for_each_problem(filename, problems, solved_by_problem)
    return solved_by_problem


def insert_solved_instances_for_each_problem(filename, problems, solved_by_problem):
    for problem in problems:
        path_to_csv = results_path(problem, file=filename)
        solved_instances = solved_by_agent(path_to_csv)
        solved_by_problem[problem] = solved_instances[0]

def table_solved_instances(filenames, problems, agent_names):
    return pd.DataFrame([get_solved_series(filename, problems) for filename in filenames], index = agent_names)

def expanded_transitions_through_whole_problem_for_agent(filename):
    raise NotImplementedError

def scatterplot(filename):
    raise NotImplementedError

def metric_evolution_over_agents(path, n, k, metric_column_name):
    df = pd.read_csv(path)
    df = notExceeded(df)
    df = df[(df['n']==n) & (df['k']==k)]
    return -df[[metric_column_name, "idx"]]


def align_agent_rows_with_mono_by_parameters(mono_df, selected_agent_df):
    aligned = pd.DataFrame(columns=selected_agent_df.columns)
    mono_sizes = []
    for index, row in mono_df.iterrows():
        n, k = row["n"], row["k"]
        selected_row = selected_agent_df.loc[(selected_agent_df['n']==n) & (selected_agent_df['k']==k)]
        if len(selected_row) != 0:
            aligned = pd.concat([aligned,selected_row])
            mono_sizes.append(row["expanded transitions"])
    aligned["mono_size"] = mono_sizes

    return aligned

def aligned_transitions_by_total_plant_size_one_problem(selected_agent_path, problem_name, mono_dirs_path ="experiments/results/ResultsPaper/"):

    selected_agent_df = pd.read_csv(selected_agent_path)
    mono_path = mono_dirs_path+problem_name+".csv"
    monolithic_df = pd.read_csv(mono_path)
    monolithic_df = monolithic_df[monolithic_df["controllerType"]=="mono"]
    #order_by_plant_size = monolithic_df
    mono = monolithic_expansions(mono_path)[["expanded transitions", "n", "k"]]
    mono.sort_values("expanded transitions", inplace=True)
    selected_agent_df = align_agent_rows_with_mono_by_parameters(mono, selected_agent_df)

    return selected_agent_df[["expanded transitions","n","k", "mono_size"]]
def monolithic_expansions(path):
    monolithic_df = pd.read_csv(path)
    monolithic_df = monolithic_df.loc[monolithic_df["controllerType"] == "mono"]
    monolithic_df["n"] = monolithic_df["testcase"].apply(lambda t: int(t.split("-")[1]))
    monolithic_df["k"] = monolithic_df["testcase"].apply(lambda t: int(t.split("-")[2]))
    monolithic_df["expanded transitions"] = monolithic_df["expandedTransitions"]
    monolithic_df = monolithic_df.dropna(subset=["expanded transitions"])
    monolithic_df = fill_df(monolithic_df, 15)
    return monolithic_df

def aligned_transitions_by_total_plant_size_one_algorithm(algorithm_path_list, problem_name_list):
    dfs = {}
    assert(len(algorithm_path_list)==len(problem_name_list))
    for (path,problem) in zip(algorithm_path_list,problem_name_list):
        dfs.update({problem : aligned_transitions_by_total_plant_size_one_problem(path,problem)})

    return dfs
def aligned_transitions_by_total_plant_size(paths_by_algorithm, problem_name_list):
    dfs_by_algorithm = {}
    for (algorithm_name, paths) in paths_by_algorithm.items():
        dfs_by_algorithm[algorithm_name] = aligned_transitions_by_total_plant_size_one_algorithm(paths.values(),problem_name_list)
    return dfs_by_algorithm
problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
path_dict = {
"labelsThatReach": {problem: "/home/marco/Desktop/Learning-Synthesis-newdirstructure/experiments/results/"+ problem + "_2_2/labelsThatReach/all_15_15000_TO:10h.csv" for problem in problems},
"boolean": {problem: "/home/marco/Desktop/Learning-Synthesis-newdirstructure/experiments/results/"+ problem + "_2_2/boolean/all_15_15000_TO:10h.csv" for problem in problems},
"RA" : {problem: "/home/marco/Desktop/Learning-Synthesis-newdirstructure/experiments/results/"+ problem + "_2_2/all_ra_15000t.csv" for problem in problems}
}
if __name__ == '__main__':
    """problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    print(get_solved_series("labelsThatReach/all_15_15000_TO:10h.csv", problems))
    print(get_solved_series("labelsThatReach/all_15_-1_TO:10m.csv", problems))
    print(get_solved_series("all_random1_1500015000.csv", problems))
    print(get_solved_series("all_ra_30m_15.csv", problems))
    #TODO(2): make this more verbose. Classes?
    #TODO(3): scatterplot
    print(table_solved_instances(["labelsThatReach/all_15_15000_TO:10h.csv","all_random1_1500015000.csv", "all_ra_15000t.csv" ], problems, ["best/5000 with 15000","random 15000", "ra 15000"]))
    """
    path = "/home/marco/Desktop/Learning-Synthesis-newdirstructure/experiments/results/AT_2_2/labelsThatReach/generalization_all_15_10h_100_5000.csv"
    path2 = "/home/marco/Desktop/Learning-Synthesis-newdirstructure/experiments/results/AT_2_2/boolean_2/all_15_-1_TO:10m.csv"
    series = metric_evolution_over_agents(path, 2, 2, "expanded transitions")
    #series = metric_evolution_over_agents("/home/marco/Desktop/Learning-Synthesis/experiments/results/AT_2_2/boolean/upTo:15_timeout2h_ebudget5000_generalization_all.csv", 2, 2, "expanded transitions")

    sns.lineplot(data = series, y="expanded transitions", x="idx")

    #plt.show()
    ##the correct ordering for the scatterPlot...
    a = aligned_transitions_by_total_plant_size_one_problem(path_dict["boolean"]["AT"], "AT")

    boolean_paths = [path_dict["boolean"][problem] for problem in problems]
    ra_paths = [path_dict["RA"][problem] for problem in problems]
    labelsThatReach_paths = [path_dict["labelsThatReach"][problem] for problem in problems]
    dfs = aligned_transitions_by_total_plant_size_one_algorithm(boolean_paths, problems)

    dfsdfs = aligned_transitions_by_total_plant_size(path_dict, problems)
    breakpoint()
    i = 0

    fig, axs = plt.subplots(1,6,figsize=(5 * 6, 6))

    for ax,problem in zip(axs,problems):
        ax.scatter(x = dfsdfs["boolean"][problem]["mono_size"], y =dfsdfs["boolean"][problem]["expanded transitions"],color="red")
        ax.scatter(x=dfsdfs["labelsThatReach"][problem]["mono_size"], y=dfsdfs["labelsThatReach"][problem]["expanded transitions"],
                   color="blue")
 
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.autoscale(enable=None, axis="x", tight=True)
    plt.tight_layout()
    plt.savefig("scatter.jpg", dpi=500)
