import pandas as pd
import pickle
from util import filename

def get_df_from_results_old(results_file, base):
    with open(results_file, "rb") as f:
        results = pickle.load(f)
        df = pd.DataFrame(results)
        df["rounded training time"] = (df["training_time"].round().astype(int) // base) * base
        return df


def get_df_from_results(results_file):
    with open(results_file, "rb") as f:
        agents, performances, results, train_results = pickle.load(f)
        df = pd.DataFrame(results + train_results)
        return df


def exp_eta_to_df(problem, n, k, minutes):
    dfs = []
    for eta in [1e-3, 1e-4, 1e-5, 1e-6]:
        dfs.append(get_df_from_results("experiments/results/"+filename([problem, n, k])+"/eta_"
                                       + str(minutes)+"_"+str(eta)+".pkl"))
        dfs[-1]["eta"] = eta
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/eta.csv")


def exp_generalization_to_df(problem, n1, k1, n2, k2, minutes):

    df = get_df_from_results("experiments/results/"+filename([problem, n1, k1])+"/generalization_"+str(minutes)+"_"+str(n2)+"_"+str(k2)+".pkl")

    df.to_csv("experiments/results/"+filename([problem, n1, k1])+"/generalization.csv")

def exp_longwindow_to_df(problem, n, k, minutes):

    df = get_df_from_results("experiments/results/"+filename([problem, n, k])+"/longwindow_"+str(minutes)+".pkl")

    df.to_csv("experiments/results/"+filename([problem, n, k])+"/longwindow_"+str(minutes)+".csv")


if __name__ == "__main__":
    exp_generalization_to_df("TL", 2, 2, 3, 3, 90)
    exp_generalization_to_df("DP", 2, 2, 3, 3, 90)
    exp_generalization_to_df("AT", 2, 2, 3, 3, 90)
    exp_generalization_to_df("TA", 2, 2, 3, 3, 90)
    exp_generalization_to_df("BW", 2, 2, 3, 3, 90)
