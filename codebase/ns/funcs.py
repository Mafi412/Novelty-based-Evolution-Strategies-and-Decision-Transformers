# Functions for the main loop of the processes.

from . import process_memory as pm
from es_utilities import utils
from es.funcs import evaluate_and_possibly_save, log_iteration_population_data

import random
from collections import namedtuple

import numpy as np


NoiseEvaluationResult = namedtuple(
    "NoiseEvaluationResult",
    [
        "task_index",
        "novelty_score_of_plus_noise",
        "runtime_of_plus_noise",
        "novelty_score_of_minus_noise",
        "runtime_of_minus_noise",
        "sum_",
        "sum_of_squares",
        "count"
    ]
)


def noise_evaluations(
    task_index,
    seed
):
    utils.set_seed(seed)
    
    # Get noise according to the task index
    noise = utils.get_noise(pm.models[pm.current_model_index], pm.shared_noise_table, pm.seed_array[task_index])
    
    # Run with +noise
    pm.test_model.copy_from(pm.models[pm.current_model_index])
    utils.add_noise(pm.test_model, noise)
    
    update_vbn_stats = (random.random() < pm.update_vbn_stats_probability)
    pm.test_environment.set_seed(seed)
    behavior_characteristic1, _, runtime1, sum1, sum_of_squares1, count1 = evaluation(pm.test_model, pm.test_environment, pm.max_runtime, update_vbn_stats)
    novelty_score1 = pm.archive.get_novelty_score_for_behavior(behavior_characteristic1)
    
    # Run with -noise
    pm.test_model.copy_from(pm.models[pm.current_model_index])
    utils.subtract_noise(pm.test_model, noise)
    
    update_vbn_stats = (random.random() < pm.update_vbn_stats_probability)
    pm.test_environment.set_seed(seed)
    behavior_characteristic2, _, runtime2, sum2, sum_of_squares2, count2 = evaluation(pm.test_model, pm.test_environment, pm.max_runtime, update_vbn_stats)
    novelty_score2 = pm.archive.get_novelty_score_for_behavior(behavior_characteristic2)
    
    if sum1 is None:
        sum_ = sum2
        sum_of_squares = sum_of_squares2
        count = count2
        
    elif sum2 is None:
        sum_ = sum1
        sum_of_squares = sum_of_squares1
        count = count1
        
    else:
        sum_ = sum1 + sum2
        sum_of_squares = sum_of_squares1 + sum_of_squares2
        count = count1 + count2
    
    return NoiseEvaluationResult(task_index, novelty_score1, runtime1, novelty_score2, runtime2, sum_, sum_of_squares, count)


def evaluation(
    test_model,
    test_environment,
    max_runtime,
    store_vbn_stats
):
    if store_vbn_stats:
        observed_states = list()
    
    if test_environment.timestep_limit is not None:
        if max_runtime is not None:
            max_timestep = min(test_environment.timestep_limit, max_runtime)
        else:
            max_timestep = test_environment.timestep_limit
    else:
        if max_runtime is not None:
            max_timestep = max_runtime
        else:
            max_timestep = int(1e18) # It's just, who would need or even want more timesteps then this...?
    
    episode_return, episode_length = 0, 0
    test_model.reset_inner_state()
    state = test_environment.reset()
    
    for timestep in range(max_timestep):
        action = test_model.choose_action(state)

        next_state, reward, terminated, truncated = test_environment.step(action)
        done = terminated or truncated

        test_model.update_after_step(state, next_state, action, reward, terminated, truncated)
            
        if store_vbn_stats:
            observed_states.append(state)
            
        state = next_state

        episode_return += reward
        episode_length += 1

        if done:
            break
    
    behavior_characteristic = test_environment.get_behavior_characteristic()
    
    if store_vbn_stats:
        observed_states = np.array([np.array(o) for o in observed_states])
        sum_ = observed_states.sum(axis=0)
        sum_of_squares = np.square(observed_states).sum(axis=0)
        count = len(observed_states)
    else:
        sum_ = None
        sum_of_squares = None
        count = None
    
    return behavior_characteristic, episode_return, episode_length, sum_, sum_of_squares, count


def update(
    weight_decay_factor,
    noise_deviation,
    batch_size
):
    # Get and weight noises
    noises = list()
    for task_index in range(len(pm.seed_array)):
        current_noise = utils.get_noise(pm.models[pm.current_model_index], pm.shared_noise_table, pm.seed_array[task_index])
        rank_weight = pm.rank_weights[task_index]
        noises.append(utils.get_weighted_noise(rank_weight, current_noise))
        if len(noises) >= batch_size:
            combined_noise = utils.get_combined_noises(noises)
            noises = [combined_noise]
    combined_noise = utils.get_combined_noises(noises)
    
    # Update the model by combined noises
    ## The combined noise is to be divided by the number of individuals evaluated and the noise deviation squared.
    ## That is because in the original paper they divide by the number of individuals evaluated and the noise deviation, but use noise drawn from distribution with sd=1 (and only scale it during evaluation).
    ## We, on the other hand, use noise drawn from distribution with other sd, which is basically noise drawn with sd=1 multiplied by our sd. Hence we have to divide this excess sd.
    combined_noise = utils.get_weighted_noise(1 / (2 * len(pm.seed_array) * (noise_deviation ** 2)), combined_noise)
    pm.models[pm.current_model_index].optimizer.update(combined_noise)
        
    # Weight decay
    utils.decay_weights(weight_decay_factor, pm.models[pm.current_model_index])
        
    # Update the virtual batch normalization stats
    pm.models[pm.current_model_index].vbn_stats.increment(
        pm.sum_of_encountered_states,
        pm.sum_of_squares_of_encountered_states,
        pm.count_of_encountered_states
    )
