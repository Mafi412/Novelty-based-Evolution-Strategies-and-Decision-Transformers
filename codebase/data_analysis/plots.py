import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


sns.set_theme(
    context="paper",
    style="white"
)


# ---------- Plots showing data from a single experiment ----------

def plot_fitness(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Task", value_name="Fitness")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Fitness", hue="Experiment name", errorbar="pi", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_evaluation_fitness(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Data type", value_name="Fitness")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Fitness", hue="Experiment name", style="Data type", style_order=["Best yet result", "Evaluation result"], data=all_data)
    
    if disable_legend:
        # Show only different line styles (data types) explanations, not color (experiment name)
        import matplotlib.lines as mlines
        
        plt.legend(handles=(
                mlines.Line2D([0], [0], color='black', linestyle="--", label="Evaluation result"),
                mlines.Line2D([0], [0], color='black', linestyle="-", label="Best yet result")
            ), loc="upper left", shadow=True)
        
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_novelty(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Task", value_name="Novelty score")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Novelty score", hue="Experiment name", errorbar="pi", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_runtime(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration"})
        df = df.melt("Iteration", var_name="Task", value_name="Runtime (timesteps)")
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Runtime (timesteps)", hue="Experiment name", errorbar="pi", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


def plot_time(*dataframes, num_of_iterations_to_plot=200, values_range=(0,None), plot_dimensions=(3.8,2.7), line_names=None, plot_title=None):
    disable_legend = False
    if line_names is None:
        disable_legend = True
        line_names = map(str, range(1, len(dataframes)+1))
        
    # Create a new DataFrame to hold all the data
    all_data = []
        
    for df, name in zip(dataframes, line_names):
        # Reset the index so "Iteration" becomes a column in the DataFrame and get the DataFrame in the long format
        df = df.reset_index()
        df = df.rename(columns={"index": "Iteration", "Wall-clock time per iteration": "Wall-clock time per iteration (seconds)"})
        
        # Add a "Experiment name" column to the DataFrame for differenciation between the runs
        df["Experiment name"] = name
        
        # Append the data to the new DataFrame
        all_data.append(df)
        
    all_data = pd.concat(all_data)

    plt.figure(figsize=plot_dimensions)
    
    sns.lineplot(x="Iteration", y="Wall-clock time per iteration (seconds)", hue="Experiment name", data=all_data)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc="upper left", shadow=True)
    
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    if plot_title is not None:
        plt.title(plot_title)
    plt.tight_layout()
    plt.show()


# ---------- Plots accumulating data from multiple experiments ----------

def create_plot_for_multiple_experiments(plot_dimensions=None, plot_title=None):
    plt.figure(figsize=plot_dimensions)
    if plot_title is not None:
        plt.title(plot_title)
        
        
def aggregate_fitness_dataframe(df):
    aggregated_df = df.groupby("Iteration").agg(
        Fitness=("Fitness", "mean"),
        lower_percentile=("Fitness", lambda x: x.quantile(0.)),
        upper_percentile=("Fitness", lambda x: x.quantile(1.))
    ).reset_index()
    return aggregated_df
    
    
def add_evaluation_data_from_one_experiment_to_plot(*dataframes, experiment_name, single_run_identifiers=None, chunk_size=100):
    if single_run_identifiers is None:
        single_run_identifiers = range(1, len(dataframes) + 1)
    else:
        if len(single_run_identifiers) != len(dataframes):
            raise ValueError("The number of single run identifiers must be the same as the number of dataframes.")
    
    combined_df = pd.concat(dataframes, keys=single_run_identifiers, names=["Run", "Iteration"])
    combined_df.reset_index(inplace=True)
    combined_df.drop(columns="Evaluation result", inplace=True)
    combined_df.rename(columns={"Best yet result": "Fitness"}, inplace=True)
    
    # Aggregate the DataFrame
    aggregated_df = aggregate_fitness_dataframe(combined_df)
    
    # sns.lineplot(x="Iteration", y="Fitness", errorbar=("pi", 100), label=experiment_name, data=combined_df)
    lineplot = sns.lineplot(x="Iteration", y="Fitness", label=experiment_name, data=aggregated_df)
    plt.fill_between(aggregated_df["Iteration"], 
                     aggregated_df["lower_percentile"], 
                     aggregated_df["upper_percentile"], 
                     alpha=0.2)
    
    # Add subtle contours to the intervals
    line_color = lineplot.get_lines()[-1].get_color()
    plt.plot(aggregated_df["Iteration"], aggregated_df["lower_percentile"], color=line_color, alpha=0.3)
    plt.plot(aggregated_df["Iteration"], aggregated_df["upper_percentile"], color=line_color, alpha=0.3)
        
        
def aggregate_runtime_dataframe(df):
    aggregated_df = df.groupby("Iteration").agg(
        Runtime=("Runtime (timesteps)", "mean"),
        lower_percentile=("Runtime (timesteps)", lambda x: x.quantile(0.025)),
        upper_percentile=("Runtime (timesteps)", lambda x: x.quantile(0.975))
    ).reset_index()
    return aggregated_df
    
    
def add_runtime_data_from_one_experiment_to_plot(*dataframes, experiment_name, single_run_identifiers=None, chunk_size=100):
    if single_run_identifiers is None:
        single_run_identifiers = range(1, len(dataframes) + 1)
    else:
        if len(single_run_identifiers) != len(dataframes):
            raise ValueError("The number of single run identifiers must be the same as the number of dataframes.")
    
    combined_df = pd.concat(dataframes, keys=single_run_identifiers, names=["Run", "Iteration"])
    combined_df.reset_index(inplace=True)
    combined_df = combined_df.melt(("Run", "Iteration"), var_name="Task", value_name="Runtime (timesteps)")
    
    # Aggregate the DataFrame
    aggregated_df = aggregate_runtime_dataframe(combined_df)
    aggregated_df.rename(columns={"Runtime": "Runtime (timesteps)"}, inplace=True)
    
    # sns.lineplot(x="Iteration", y="Runtime (timesteps)", errorbar="pi", label=experiment_name, data=combined_df)
    lineplot = sns.lineplot(x="Iteration", y="Runtime (timesteps)", label=experiment_name, data=aggregated_df)
    plt.fill_between(aggregated_df["Iteration"], 
                     aggregated_df["lower_percentile"], 
                     aggregated_df["upper_percentile"], 
                     alpha=0.2)
    
    # Add subtle contours to the intervals
    line_color = lineplot.get_lines()[-1].get_color()
    plt.plot(aggregated_df["Iteration"], aggregated_df["lower_percentile"], color=line_color, alpha=0.3, linewidth=0.5)
    plt.plot(aggregated_df["Iteration"], aggregated_df["upper_percentile"], color=line_color, alpha=0.3, linewidth=0.5)
    
    
def show_plot_for_multiple_experiments(num_of_iterations_to_plot=200, values_range=(0,None), disable_legend=False, legend_location="upper left", legend_shadow=True):
    plt.ylim(values_range)
    plt.xlim(1, num_of_iterations_to_plot)
    
    if disable_legend:
        plt.legend().remove()
    else:
        plt.legend(loc=legend_location, shadow=legend_shadow)
        
    plt.tight_layout()
    plt.show()
