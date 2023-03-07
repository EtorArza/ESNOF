<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>

# Generalized Early Stopping for Policy Learning

Lengthy evaluation times are common in many optimization problems such as policy learning tasks, especially when they involve the physical world. When evaluating solutions, it is sometimes clear that the objective function is not going to increase with additional computation time (for example when a two wheel robot continuously spins in place).
In those cases, it makes sense to stop the evaluation early to save computation time. However, these approaches are problem specific and need to be specifically designed for the task at hand.


In our paper, we propose an early stopping method for policy learning. The proposed method only looks at the objective value at each time step and requires no problem specific knowledge. We tested the introduced stopping criterion in 6 policy learning environments and showed that it usually saves a significant amount of time. We also compare it with problem specific stopping criteria and show that it performs comparably, while being more general (it is not task specific).


## Usage

It is quite easy to use the early stopping criterion in your project. Each time you evaluate a solution (e.g. a policy) $\theta$, you get an objective value $f[t](\theta)$ at each time step (e.g. the cumulative reward). Then you stop evaluating $\theta$ at time step $t$ if:

$$t > t_{grace}$$

and

$$\max\{f[t](\theta), f[t-t_{grace}](\theta) \} < \min\{f[t](\theta_{best}), f[t-t_{grace}](\theta_{best})\}$$



## Reproducing the experiments in the paper

### Installation

Clone the repo and check the instructions to install the 

### Usage

The experiments can be executed with the following four scripts:
- `scripts/experiment_garage_CMAES_gym.py`
- `scripts/experiment_simulatedARE.py`
- `scripts/experiment_supermario.py`
- `scripts/experiment_veenstra.py`

An extra parameter is required. 
- `--launch_local` to execute the experiments
- `--plot` to generate the figures in the paper from the results in `/results/data`. 

e.g.

```
python scripts/experiment_garage_CMAES_gym.py --plot
```


### Credits

List of external resources used in the project:

- [Gym Rem2D by FrankVeenstra](https://github.com/FrankVeenstra/gym_rem2D) with MIT licence.
- [Autonomous Robotics Evolution by Leni Le Goff et al.](https://bitbucket.org/autonomousroboticsevolution/evorl_gecco_2021/src/master/)
- [Super Mario by Vivek Verma](https://github.com/vivek3141/super-mario-neat)
- [Gym Garage](https://github.com/rlworkgroup/garage)
- [OpenAI Gym](https://www.gymlibrary.dev/environments/)
- [MuJoCo](https://mujoco.org/)