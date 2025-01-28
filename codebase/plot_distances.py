from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


sns.set_theme(
    context="paper",
    style="white"
)


def load_distances(path_to_distances_file):
    with open(path_to_distances_file, "r") as file:
        return [float(line.split()[-1]) for line in file.readlines()]


def plot_distances_for_multiple_experiments(distances_list, experiment_names, plot_dimensions):
    if len(distances_list) != len(experiment_names):
        raise ValueError("Number of experiments and experiment names should be equal.")
    
    plt.figure(figsize=plot_dimensions)
    
    # Create DataFrame suitable for seaborn
    data = []
    for exp_name, distances in zip(experiment_names, distances_list):
        data.extend([(exp_name, distance) for distance in distances])
    data = pd.DataFrame(data, columns=["Experiment", "Distance"])
    
    # Use seaborn stripplot for showing individual points
    sns.stripplot(data=data, x="Experiment", y="Distance", hue="Experiment")
        
    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    
    plt.xlabel(None)
    plt.ylabel("Distance from origin\n(environment-dependent units)")
    plt.tight_layout()
    plt.show()


def main(args):
    paths_to_all_the_experiment_folders  = [Path(path) for path in args.paths_to_all_the_experiment_folders]
    
    experiment_names = args.experiment_names
    if experiment_names is None:
        experiment_names = [path.stem for path in paths_to_all_the_experiment_folders]
        
    distances_list = [load_distances(path / "best_distances_from_origin") for path in paths_to_all_the_experiment_folders]
    
    plot_distances_for_multiple_experiments(
        distances_list,
        experiment_names,
        args.plot_dimensions
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("paths_to_all_the_experiment_folders", type=str, nargs="+", help="Paths to the folders for all the experiments that are required to be plotted, where the data from all the experiment runs are stored in their respective subfolders.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=None, help="Plot dimensions (two values, x and y).")
    parser.add_argument("-n", "--experiment_names", type=str, nargs="+", help="Names for the individual experiments in the order as passed to the base path argument.")
    
    main(parser.parse_args())
