from pathlib import Path
from argparse import ArgumentParser
import warnings

import gymnasium as gym

from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapperNsEndXYWithNondirectionalDistanceReward
from es_utilities.play import simulate
from wrapped_components.model_ff_mujoco_wrappers import get_new_wrapped_ff_humanoid
from wrapped_components.model_dt_mujoco_wrappers import get_new_wrapped_dt_humanoid


def model_files(folder_path):
    path = Path(folder_path)
    for model_file in path.rglob("*.model"):
        if model_file.is_file():
            yield str(model_file.with_suffix(""))
        
        
def get_best_average_distance_from_origin_per_experiment_run(
    test_model,
    path_to_individual_experiment_run_folder,
    num_of_episodes
):
    best_distance = -float("inf")
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    environment = GymMujocoWrapperNsEndXYWithNondirectionalDistanceReward(
        env=gym.make("Humanoid-v4", render_mode=None),
        seed=None
    )
    
    for model_file_path in model_files(path_to_individual_experiment_run_folder):
        test_model.load_parameters(model_file_path)
        
        distances, _ = simulate(test_model, environment, num_of_episodes, False)
        distance = sum(distances) / len(distances)
        
        if distance > best_distance:
            best_distance = distance
            
    return best_distance


def get_best_average_distances_from_origin_for_experiment(
    test_model,
    path_to_experiment_folder,
    num_of_episodes
):
    path_to_experiment_folder = Path(path_to_experiment_folder)
    best_distances = dict()
    
    for individual_experiment_run_folder in path_to_experiment_folder.iterdir():
        if not individual_experiment_run_folder.is_dir():
            continue
        
        print("Processing folder:", individual_experiment_run_folder.stem)
        
        best_distances[individual_experiment_run_folder.stem] = get_best_average_distance_from_origin_per_experiment_run(
            test_model,
            individual_experiment_run_folder,
            num_of_episodes
        )
    
    return best_distances


def main(args):
    path_to_experiment_folder = Path(args.path_to_experiment_folder)
    if not path_to_experiment_folder.is_dir():
        raise ValueError("The path to the experiment folder is not a directory.")
    
    if args.model_type == "ff":
        test_model = get_new_wrapped_ff_humanoid()
    
    elif args.model_type == "dt":
        test_model = get_new_wrapped_dt_humanoid(7000)
    
    else:
        raise ValueError("The model type must be either 'ff' or 'dt'.")
    
    print("Getting the best distances from the origin for the individual experiment runs in the experiment folder:", path_to_experiment_folder)
    distances = get_best_average_distances_from_origin_for_experiment(
        test_model=test_model,
        path_to_experiment_folder=path_to_experiment_folder,
        num_of_episodes=args.num_of_episodes
    )
    
    file_with_distances = path_to_experiment_folder / "best_distances_from_origin"
    print("Writing the distances to the file:", file_with_distances)
    file_with_distances.write_text("\n".join((f"{folder}: {str(distance)}" for folder, distance in distances.items())))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_experiment_folder", type=str, help="Path to the folder, where the data from all the experiment runs are stored in their respective subfolders.")
    parser.add_argument("-n", "--num_of_episodes", type=int, default=10, help="Number of episodes to be used for the evaluation of each model.")
    parser.add_argument("-t", "--model_type", type=str, default="ff", help="Type of the model. Either 'ff' for feed-forward model, or 'dt' for decision transformer.")
    
    main(parser.parse_args())
