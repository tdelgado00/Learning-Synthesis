import pandas as pd
from util import *

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


if __name__ == '__main__':
    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    print(get_solved_series("labelsThatReach/all_15_15000_TO:10h.csv", problems))
    #print(get_solved_series("labelsThatReach/all_15_-1_TO:10m.csv", problems))
    print(get_solved_series("all_random1_1500015000.csv", problems))
    #TODO(1): why does it fail with 10 min random? where is 10 min ra?
    #TODO(2): make this more verbose
    #TODO(3): scatterplot
    print(table_solved_instances(["labelsThatReach/all_15_15000_TO:10h.csv","all_random1_1500015000.csv", "all_ra_15000t.csv" ], problems, ["best/5000 with 15000","random 15000", "ra 15000"]))

