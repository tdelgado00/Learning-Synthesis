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
    for index, row in mono_df.iterrows():
        n, k = row["n"], row["k"]
        selected_row = selected_agent_df.loc[(selected_agent_df['n']==n) & (selected_agent_df['k']==k)]
        if len(selected_row) != 0:
            aligned = pd.concat([aligned,selected_row])

    return aligned

def aligned_transitions_by_total_plant_size(selected_agent_path, mono_path ="experiments/results/ResultsPaper/AT.csv"):
    selected_agent_df = pd.read_csv(selected_agent_path)
    monolithic_df = pd.read_csv(mono_path)

    #order_by_plant_size = monolithic_df
    mono = monolithic_expansions(mono_path)[["expanded transitions", "n", "k"]]

    mono.sort_values("expanded transitions", inplace=True)

    selected_agent_df = align_agent_rows_with_mono_by_parameters(mono, selected_agent_df)

    return selected_agent_df[["expanded_transitions","n","k"]]
def monolithic_expansions(path):
    monolithic_df = pd.read_csv(path)
    monolithic_df = monolithic_df.loc[monolithic_df["controllerType"] == "mono"]
    monolithic_df["n"] = monolithic_df["testcase"].apply(lambda t: int(t.split("-")[1]))
    monolithic_df["k"] = monolithic_df["testcase"].apply(lambda t: int(t.split("-")[2]))
    monolithic_df["expanded transitions"] = monolithic_df["expandedTransitions"]
    monolithic_df = monolithic_df.dropna(subset=["expanded transitions"])
    monolithic_df = fill_df(monolithic_df, 15)
    return monolithic_df

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
    path = "/home/marco/Desktop/Learning-Synthesis/experiments/results/AT_2_2/labelsThatReach/generalization_all_15_10h_100_5000.csv"
    path2 = "/home/marco/Desktop/Learning-Synthesis/experiments/results/AT_2_2/boolean_2/all_AT_15_-1_TO:10m.csv"
    series = metric_evolution_over_agents(path, 2, 2, "expanded transitions")
    #series = metric_evolution_over_agents("/home/marco/Desktop/Learning-Synthesis/experiments/results/AT_2_2/boolean/upTo:15_timeout2h_ebudget5000_generalization_all.csv", 2, 2, "expanded transitions")

    sns.lineplot(data = series, y="expanded transitions", x="idx")

    #plt.show()
    ##the correct ordering for the scatterPlot...
    aligned_transitions_by_total_plant_size(path2)
