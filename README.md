# Morphing Agents

The majority of MuJoCo benchmarks involve an agent with a fixed morphology. In this package, we provide a suite of continuous control tasks, where the agent samples its morphology at runtime. Have Fun! -Brandon

# Installation

You may install the package directly from PyPI using the following command.

```bash
pip install morphing-agents
```

# Usage

At this time I have created three agents with dynamic morphologies. You can instantiate them like this.

```python
from morphing_agents.mujoco.ant.env import MorphingAntEnv
env0 = MorphingAntEnv(num_legs=4, expose_design=True)

from morphing_agents.mujoco.dog.env import MorphingDogEnv
env1 = MorphingDogEnv(num_legs=4, expose_design=True)

from morphing_agents.mujoco.dkitty.env import MorphingDKittyEnv
env2 = MorphingDKittyEnv(expose_design=True)
```

These environments inherit from `gym.Env` and can be used as such. The parameter `num_legs` determines the number of legs in the agent's design specification, which is sampled at random at the beginning of an episode. The parameter `expose_design` vectorizes the design specification and concatenates it with the observation.

# Package Structure

The package is organized such that for every agent, there is a `elements.py` and `designs.py` file that specifies the atomic design elements, and combinations of these elements in a list format respectively. In addition, each agent has a default design present in its `designs.py` and a `sample_uniformly` function for sampling designs.

Each agent expects to have an `expose_design` parameter, and possibly a `num_legs` parameter if the number of legs can change. For environments like the DKitty, where this number cannot change, it is omitted from the constructor arguments of the environment class. Each environment conforms to `gym.Env`.
