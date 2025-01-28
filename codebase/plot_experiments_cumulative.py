from data_analysis import plots, dataloading

import os
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    if args.experiment_names is not None:
        if len(args.experiment_names) != len(args.paths_to_experiment_folders):
            raise ValueError("The number of experiment names must match the number of base paths to the data folders.")
        experiment_names = args.experiment_names
    
    else:
        experiment_names = [str(i+1) for i in range(len(args.paths_to_experiment_folders))]
       
    if args.plot_type == "eval":
        print("Creating the plot for the evaluation fitnesses...")
        plots.create_plot_for_multiple_experiments(
            plot_dimensions=args.plot_dimensions
        )
        
        for i, experiment_path in enumerate(args.paths_to_experiment_folders):
            path_to_experiment_folder = Path(experiment_path)
            current_experiment_evaluation_data = []
            
            if args.experiment_names is not None:
                print(f"Processing data from experiment named {experiment_names[i]}...")

            else:
                print(f"Processing data from experiment number {i+1}:")

            print("Loading data...")
            paths_to_evaluation_data = [run_directory / "log.evaluations.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]
            
            single_run_identifiers = list()
            for path in paths_to_evaluation_data:
                print("Loading data from file:", path)
                new_evaluation_dataframe = dataloading.load_evaluation_fitnesses_from_csv(path, args.max_iterations)
                current_experiment_evaluation_data.append(new_evaluation_dataframe)
                single_run_identifiers.append(path.parent.stem)

            print("Adding evaluation fitnesses to the plot...")
            plots.add_evaluation_data_from_one_experiment_to_plot(
                *current_experiment_evaluation_data,
                experiment_name=experiment_names[i],
                single_run_identifiers=single_run_identifiers
            )
            
        print("Rendering the plot for the evaluation fitnesses...")
        plots.show_plot_for_multiple_experiments(
            num_of_iterations_to_plot=args.max_iterations,
            values_range=(0, args.max_fitness),
            disable_legend=(args.experiment_names is None)
        )
        
    elif args.plot_type == "runtime":
        print("Creating the plot for the runtimes...")
        plots.create_plot_for_multiple_experiments(
            plot_dimensions=args.plot_dimensions
        )
        
        for i, experiment_path in enumerate(args.paths_to_experiment_folders):
            path_to_experiment_folder = Path(experiment_path)
            current_experiment_runtime_data = []
            
            if args.experiment_names is not None:
                print(f"Processing data from experiment named {experiment_names[i]}...")

            else:
                print(f"Processing data from experiment number {i+1}:")

            print("Loading data...")
            paths_to_runtime_data = [run_directory / "log.runtime.csv" for run_directory in path_to_experiment_folder.iterdir() if run_directory.is_dir()]
            
            single_run_identifiers = list()
            for path in paths_to_runtime_data:
                print("Loading data from file:", path)
                new_runtime_dataframe = dataloading.load_runtimes_from_csv(path, args.max_iterations)
                current_experiment_runtime_data.append(new_runtime_dataframe)
                single_run_identifiers.append(path.parent.stem)

            print("Adding runtimes to the plot...")
            plots.add_runtime_data_from_one_experiment_to_plot(
                *current_experiment_runtime_data,
                experiment_name=experiment_names[i],
                single_run_identifiers=single_run_identifiers
            )
            
        print("Rendering the plot for the runtimes...")
        plots.show_plot_for_multiple_experiments(
            num_of_iterations_to_plot=args.max_iterations,
            disable_legend=(args.experiment_names is None)
        )
    
    else:
        raise ValueError("Invalid plot type. Use 'eval' for evaluation fitnesses or 'runtime' for runtimes.")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("paths_to_experiment_folders", type=str, nargs="+", help="Path(s) to the experiment folder(s) containing the directories of individual runs of the experiment with logged data to be plotted.")
    parser.add_argument("-t", "--plot_type", type=str, default="eval", help="Type of the data to plot from the experiments. Either 'eval' for evaluation fitnesses, or 'runtime' for runtimes.")
    parser.add_argument("-i", "--max_iterations", type=int, default=600, help="Maximal number of iterations to be plotted.")
    parser.add_argument("-f", "--max_fitness", type=float, default=10, help="Maximal fitness value on the graphs.")
    parser.add_argument("-pd", "--plot_dimensions", nargs=2, type=float, default=(3.8, 2.7), help="Plot dimensions (two values, x and y).")
    parser.add_argument("-n", "--experiment_names", type=str, nargs="+", help="Names for the individual experiments in the order as passed to the base path argument.")
    
    main(parser.parse_args())
