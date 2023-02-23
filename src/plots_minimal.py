import pandas as pd
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, OrderedDict


def get_solved_series(filepath_by_problem : Dict, problem_names : list[str]):
    solved_by_problem = pd.Series(index = problem_names ,dtype = float)
    insert_solved_instances_for_each_problem(filepath_by_problem, solved_by_problem)
    return solved_by_problem


def insert_solved_instances_for_each_problem(filepath_by_problem : Dict, solved_by_problem : pd.Series):
    for problem, path in filepath_by_problem.items():
        solved_instances = solved_by_agent(path)
        solved_by_problem[problem] = solved_instances[0]

def table_solved_instances( filepaths_by_agent : OrderedDict[str,Dict[str,str]], problems : list[str] = ["AT", "BW", "CM", "DP", "TA", "TL"]):
    return pd.DataFrame([get_solved_series(filepaths, problems) for (agent_name ,filepaths) in filepaths_by_agent.items()], index = filepaths_by_agent.keys())

def expanded_transitions_through_whole_problem_for_agent(filename):
    raise NotImplementedError


def metric_evolution_over_agents(path, n, k, metric_column_name):
    df = pd.read_csv(path)
    df = notExceeded(df)
    df = df[(df['n']==n) & (df['k']==k)]
    return df[[metric_column_name, "idx"]]


def _align_agent_rows_with_mono_by_parameters(mono_df, selected_agent_df):
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

def _aligned_transitions_by_total_plant_size_one_problem(selected_agent_path, problem_name, mono_dirs_path ="experiments/results/ResultsPaper/"):
    selected_agent_df = pd.read_csv(selected_agent_path)
    mono_path = mono_dirs_path+problem_name+".csv"
    monolithic_df = pd.read_csv(mono_path)
    monolithic_df = monolithic_df[monolithic_df["controllerType"]=="mono"]
    #order_by_plant_size = monolithic_df
    mono = monolithic_expansions(mono_path)[["expanded transitions", "n", "k"]]
    mono.sort_values("expanded transitions", inplace=True)
    selected_agent_df = _align_agent_rows_with_mono_by_parameters(mono, selected_agent_df)

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
    """
    In: A list of paths to a selected algorithm evaluations up to (15,15) for a set of problems, and their respective problem names.
    Out:
    """
    dfs = {}
    assert(len(algorithm_path_list)==len(problem_name_list))
    for (path,problem) in zip(algorithm_path_list,problem_name_list):
        dfs.update({problem : _aligned_transitions_by_total_plant_size_one_problem(path, problem)})

    return dfs
def aligned_transitions_by_total_plant_size(paths_by_algorithm : Dict[str,  list[str]], problem_name_list : str = ["AT", "BW", "CM", "DP", "TA", "TL"]):
    """
    In: A dictionary where keys are algorithm names and keys are a list of the paths to each problem performance .csv files.
    Out: The same dictionary, but replacing the list by a dictionary where keys are problem names and values are the respective DataFrames of the .csv files.
    """
    dfs_by_algorithm = {}
    for (algorithm_name, paths) in paths_by_algorithm.items():
        dfs_by_algorithm[algorithm_name] = aligned_transitions_by_total_plant_size_one_algorithm(paths.values(),problem_name_list)
    return dfs_by_algorithm


def transitions_scatter_by_agent(agent_dfs : Dict[str , Dict[str, pd.DataFrame]], problems : list[str] = ["AT", "BW", "CM", "DP", "TA", "TL"], img_name : str = "scatter.jpg"):
    """
        In: A dictionary of dictionaries where each dictionary has the results of the final execution of the selected agent. (As returned in aligned_transitions_by_total_plant_size)
        Out: A scatterplot with the expanded transitions per finished problem, where the x-axis is the total plant size and the y-axis is the strategy's expanded transitions (log scale).
        Each DataFrame must have the columns 'mono_size' and 'expanded transitions' (as the x and y axis respectively).

    """
    fig, axs = plt.subplots(1, 6, figsize=(5 * 6, 6))
    colorMapping = {
        "boolean" : "red",
        "labelsThatReach" : "green",
        "RA" : "black"

    }

    for ax, problem in zip(axs, problems):
        legends = []
        for agent in agent_dfs.keys():
            assert agent in colorMapping.keys(), "Color not supported"
            ax.scatter(x=agent_dfs[agent][problem]["mono_size"], y=agent_dfs[agent][problem]["expanded transitions"],
                       color=colorMapping[agent])
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.autoscale(enable=None, axis="x", tight=True)
            legends.append(agent)
        ax.legend(legends)


    plt.tight_layout()
    plt.savefig(img_name, dpi=500)


problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
path_dict = {
"labelsThatReach": {problem: "/home/marco/Desktop/Learning-Synthesis/experiments/results/"+ problem + "_2_2/labelsThatReach/all_15_15000_TO:10h.csv" for problem in problems},
"boolean": {problem: "/home/marco/Desktop/Learning-Synthesis/experiments/results/"+ problem + "_2_2/boolean/all_15_15000_TO:10h.csv" for problem in problems},
"RA" : {problem: "/home/marco/Desktop/Learning-Synthesis/experiments/results/"+ problem + "_2_2/all_ra_15000t.csv" for problem in problems},
"GPUboolean255_255_5" : {problem: "/home/marco/Desktop/Learning-Synthesis/experiments/results/"+ problem + "_2_2/GPUboolean255_255_5/all_ra_15000t.csv" for problem in problems}
}




if __name__ == '__main__':

    path = "/home/marco/Desktop/Learning-Synthesis/experiments/results/AT_2_2/labelsThatReach/generalization_all_15_10h_100_5000.csv"
    path2 = "/home/marco/Desktop/Learning-Synthesis/experiments/experiments/results/AT_2_2/boolean_2/all_15_-1_TO:10m.csv"
    series = metric_evolution_over_agents(path, 2, 2, "expanded transitions")

    #sns.lineplot(data = series, y="expanded transitions", x="idx")
    #plt.show()
    ##the correct ordering for the scatterPlot...


    agents_results_by_name_and_problem = aligned_transitions_by_total_plant_size(path_dict)
    boolean_paths = path_dict["boolean"].values()
    dfs = aligned_transitions_by_total_plant_size_one_algorithm(boolean_paths, problems)


    res = table_solved_instances(path_dict)
    breakpoint()
    i = 0
    #transitions_scatter_by_agent(agents_results_by_name_and_problem, img_name="trial_1.jpg")
