#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.
Here it runs CartPole-v1 environment with 100 epoches.
Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from garage import wrap_experiment
from garage._functions import *
import garage
import time
import argparse
import sys

TOTAL_COMPUTED_EPISODES = 0 # number of episodes (1 episode is equal to a complete begining -> [environment-> action -> reward] -> end cycle)
RUNTIMES = []
START_REF_TIME = None
POPSIZE = 4  # CMA-ES Population size.


parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('--method', required=True, metavar='method', type=str, help='Must be constant or bestasref.', default=None, nargs='?')
parser.add_argument('--gymEnvName', required=True, metavar='gymEnvName', type=str, help='Gym environment.', default=None, nargs='?')
parser.add_argument('--action_space', required=True, metavar='action_space', type=str, help='Must be continuous or discrete', default=None, nargs='?')
parser.add_argument('--seed', required=True, metavar='seed', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gracetime', required=True, metavar='gracetime', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gens', required=True, metavar='gens', type=int, help='NUmber of generations or epochs', default=None, nargs='?')
parser.add_argument('--max_episode_length', required=True, metavar='max_episode_length', type=int, help='Number of max frames per experiment', default=None, nargs='?')
parser.add_argument('--res_filepath', required=True, metavar='res_filepath', type=str, help='Result file path', default=None, nargs='?')

args = parser.parse_args()


modifyRuntime_flag = bool(["constant", "bestasref"].index(args.method)) 
gymEnvName = args.gymEnvName
is_action_space_discrete = bool(["continuous", "discrete"].index(args.action_space))
seed = args.seed
GRACE = args.gracetime
n_epochs = args.gens
MAX_EPISODE_LENGTH = args.max_episode_length
res_filepath = args.res_filepath

REF_CUMULATIVE_FITNESSES = np.array([-1e20] * MAX_EPISODE_LENGTH)
batch_size = MAX_EPISODE_LENGTH

print("----")
print(args)
print("----")

# monkeypatch.setattr('pytest_bug.b.foo', foo_patch)
# monkeypatch.setattr('another_package.bar', lambda: print('patched'))
def launch_experiment(ctxt=None, gymEnvName=gymEnvName, earlyStop=True, seed=seed):

    og_log_performance = garage.log_performance

    def custom_log_performance(itr, batch, discount, prefix='Evaluation'):
        """Evaluate the performance of an algorithm on a batch of episodes.

        Args:
            itr (int): Iteration number.
            batch (EpisodeBatch): The episodes to evaluate with.
            discount (float): Discount value, from algorithm's property.
            prefix (str): Prefix to add to all logged keys.

        Returns:
            numpy.ndarray: Undiscounted returns.

        """
        global REF_CUMULATIVE_FITNESSES
        global GRACE
        global RUNTIMES
        global START_REF_TIME
        global TOTAL_COMPUTED_EPISODES

        if START_REF_TIME is None:
            START_REF_TIME = time.time()

        print("Custom log_performance")

        returns = []
        undiscounted_returns = []
        termination = []
        success = []
        sum_of_returns = 0

    
        episode_start_ref_t = time.time()
        i = 0
        for eps in batch.split():
            i += 1
            returns.append(discount_cumsum(eps.rewards, discount))
            rewards_episode_sum = sum(eps.rewards)
            undiscounted_returns.append(rewards_episode_sum)

            # Calculate the cumulative reward in current 
            sum_of_returns += rewards_episode_sum

            # Halt computation cumulative reward is worse than ref
            if modifyRuntime_flag and i >= GRACE and REF_CUMULATIVE_FITNESSES[i - GRACE] > sum_of_returns:
                print("Stop computation: ref , sum of returns =  ", REF_CUMULATIVE_FITNESSES[i - GRACE], sum_of_returns)
                break

            termination.append(float(any(step_type == StepType.TERMINAL for step_type in eps.step_types)))
            if 'success' in eps.env_infos:
                success.append(float(eps.env_infos['success'].any()))

        TOTAL_COMPUTED_EPISODES += i

        print("f =",sum_of_returns)

        # Updating ref fitness.
        if sum_of_returns > REF_CUMULATIVE_FITNESSES[-1]:
            print("--Updating refs--")
            print("Old refs:", REF_CUMULATIVE_FITNESSES)
            REF_CUMULATIVE_FITNESSES[0:len(undiscounted_returns)] = np.cumsum(undiscounted_returns)
            REF_CUMULATIVE_FITNESSES[len(undiscounted_returns):] = np.cumsum(undiscounted_returns)[-1]
            print("New refs:", REF_CUMULATIVE_FITNESSES)
            print("--")



        average_discounted_return = np.mean([rtn[0] for rtn in returns])

        with tabular.prefix(prefix + '/'):
            tabular.record('Iteration', itr)
            tabular.record('NumEpisodes', len(returns))
            tabular.record('AverageDiscountedReturn', average_discounted_return)
            tabular.record('AverageReturn', np.mean(undiscounted_returns))
            tabular.record('StdReturn', np.std(undiscounted_returns))
            tabular.record('MaxReturn', np.max(undiscounted_returns))
            tabular.record('MinReturn', np.min(undiscounted_returns))
            tabular.record('TerminationRate', np.mean(termination))
            if success:
                tabular.record('SuccessRate', np.mean(success))

        RUNTIMES.append(time.time() - episode_start_ref_t)

        if itr%POPSIZE == POPSIZE-1:
            runtimes = "("+";".join(map(str, RUNTIMES))+")"
            with open(res_filepath, "a+") as f:
                print("seed_"+str(seed)+"_gymEnvName_"+gymEnvName, REF_CUMULATIVE_FITNESSES[-1], time.time() - START_REF_TIME, TOTAL_COMPUTED_EPISODES, itr, runtimes , file=f, sep=",", end="\n")
            RUNTIMES = []



        return undiscounted_returns



    cma_es_gym = None
    @wrap_experiment
    def cma_es_gym(ctxt=None, gymEnvName=gymEnvName, earlyStop=True, seed=seed):


        if earlyStop:
            garage.log_performance = custom_log_performance
        else:
            garage.log_performance = og_log_performance

        from garage.envs import GymEnv
        from garage.experiment.deterministic import set_seed
        from garage.np.algos import CMAES
        from garage.sampler import LocalSampler
        from garage.tf.policies import CategoricalMLPPolicy, ContinuousMLPPolicy
        from garage.trainer import TFTrainer


        set_seed(seed)

        with TFTrainer(ctxt) as trainer:
            global MAX_EPISODE_LENGTH
            env = GymEnv(gymEnvName, max_episode_length=MAX_EPISODE_LENGTH)
            
            if is_action_space_discrete:
                policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
            else:
                policy = ContinuousMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
            sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
            algo = CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=POPSIZE)

            trainer.setup(algo, env)
            trainer.train(n_epochs=n_epochs, batch_size=batch_size)


            # runtimes = "("+";".join(map(str, self.frames_in_gen))+")"
            # print("seed_"+str(self.seed)+"_level_"+self.level, self.best_fitness, time.time() -self.sw, self.total_frames, self.evals, runtimes , file=f, sep=",", end="\n")

    cma_es_gym()

if __name__ == "__main__":
    launch_experiment(earlyStop=True, seed=seed)
    
