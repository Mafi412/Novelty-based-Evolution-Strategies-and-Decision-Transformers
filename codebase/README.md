# OpenAI-ES and its Novelty-based variants (on a Decision Transformer)

This project contains codebase for our implementation of three distributed algorithms from the class of evolution strategies, [OpenAI-ES](https://arxiv.org/abs/1703.03864) (*es* folder), its novelty-based variant [NS-ES](https://arxiv.org/abs/1712.06560) (*ns* folder), and a basic quality-diversity version [NSR-ES](https://arxiv.org/abs/1712.06560) (*qd* folder), as well as means to analyze the data gathered during the runs.

There are also a few scripts running experiments to test those algorithms in a MuJoCo Humanoid locomotion environment on a simple feed-forward model and on a [Decision Transformer](https://arxiv.org/abs/2106.01345) architecture.

For the Decision Transformer, there is a possibility to use a pretrained model to seed the search. Pretrained models can be found in folder *pretrained_ckpts*. Code used for their training can be found in *supervised_pretraining* folder.

## How-to:

### Start a training

The provided training scripts (*train_\*.py*) provide a showcase of how the training script should look like. The argumens are described in the scripts, or You can see a help by running the scripts with *-h*, or *--help* option.

The underlying implementations of all the algorithms require MPI for their parallelization, so they need to be run using an MPI launcher.

### Simulate a trained agent

There are scripts for simulating the agent in the environment (*play_\*.py*). For Humanoid environment, there can either be no visual output, just the return obtained and runtime of the epsiode; or there can be a video-recording of the agent's rollouts; or there can be a classical visual output. The provided scripts stand as a showcases of how to implement custom replay script.

### Perform a basic data analysis

Scripts for basic data visualization (*plot_\*.py*) are provided, which plot the data in several ways. *plot_experiment.py* plots the data collected during a single experiment run, whereas *plot_experiment_cumulative.py* aggregates chosen data per multiple experiment runs.

Using *get_humanoid_distances* and *plot_distances.py* allows us to obtain and visualize the best average distances from origin that the final agents from each run of each given experiment are able to achieve.

### Utilize a custom ...

#### Agent policy architecture

In *es_utilities* folder, *wrappers.py* file, there is a **EsModelWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions. The custom policy will be stored in a field *model*. There is even possibility to utilize a Virtual Batch Normalization, provided by the wrapper. Examples might be found in folder *wrapped_components*.

#### Environment

In *es_utilities* folder, *wrappers.py* file, there is a **EsEnvironmentWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions, and even the *state_shape* property, in case the current implemantation would not return the desired state shape of the given environment. The custom environment will be stored in a field *env*. For novelty based searches, the class should be derived from a **NsEnvironmentWrapper** class present in folder *novelty_utilities*, file *ns_wrappers.py*. This adds one more function to override, returning the behavior of an agent in the environment since last reset. Examples might be found in folder *wrapped_components*.

### Create and use a behavior characteristic

To create a custom behavior characteristic, You must derive your own custom class from the **AbstractBehaviorCharacterization** class found in *novelty_utilities* folder in file *behavior.py*, again overriding its compare_to method. (Yet, ideally invoking *super().compare_to(...)* at the beginning of the custom implementation.) This behavior should then be used in the 
class derived from **NsEnvironmentWrapper**, as mentioned earlier.
