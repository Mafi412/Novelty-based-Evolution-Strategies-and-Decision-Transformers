# Memory defining file with variables for the individual MPI processes to keep the things they need for the ns in.

# Models to be trained (metapopulation),
# index of the current model to be trained,
# test model used for testing of various noises
# and test environment the testing takes place in.
models = None
current_model_index = None
test_model = None
test_environment = None

# Shared noise table
shared_noise_table = None

# Array of seeds of the individual tasks
seed_array = None

# Upper bound on the length of runtime for the evaluations
max_runtime = None

# Array of ranks / weights for model update in each iteration (results from achieved fitnesses and novelty scores of individual noises)
rank_weights = None

# VBN-stats-related variables
update_vbn_stats_probability = None
sum_of_encountered_states = None
sum_of_squares_of_encountered_states = None
count_of_encountered_states = None

# Behavior archive for novelty score computation
archive = None