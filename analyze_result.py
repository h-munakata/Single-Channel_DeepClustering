import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

def calc_statistic_of_result(path_csv):
    df = pd.read_csv(path_csv)
    si_sdri = []
    for i in range(2):
        si_sdri.append(df.loc[:, [f"SI-SDRi_{i+1}"]].rename(columns={f"SI-SDRi_{i+1}" :"SI-SDRi"}))
    si_sdri = pd.concat(si_sdri)

    mean_si_sdri = np.mean(si_sdri)
    print(float(np.mean(si_sdri)), float(np.std(si_sdri)))
    plot_result(si_sdri)



def plot_result(result):
    plt.hist(result, bins=50)
    plt.savefig("figure.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analayze .csv of reparated result')
    parser.add_argument('path_csv', help='path of .csv') 
    args = parser.parse_args()

    calc_statistic_of_result(args.path_csv)