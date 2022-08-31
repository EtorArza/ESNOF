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

MAX_EPISODE_LENGTH = 400
TOTAL_COMPUTED_EPISODES = 0 # number of episodes (1 episode is equal to: one [action -> environment -> reward] cycle)
RUNTIMES = []
REF_CUMULATIVE_FITNESSES = np.array([-1e20] * MAX_EPISODE_LENGTH)
GRACE = 20
START_REF_TIME = None
n_samples = 20  # CMA-ES Population size.
n_epochs = 100
batch_size = 1000








# monkeypatch.setattr('pytest_bug.b.foo', foo_patch)
# monkeypatch.setattr('another_package.bar', lambda: print('patched'))
def launch_experiment(ctxt=None, gymEnvName='CartPole-v1', earlyStop=True, seed=1):

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
        for i, eps in enumerate(batch.split()):

            returns.append(discount_cumsum(eps.rewards, discount))
            rewards_episode_sum = sum(eps.rewards)
            undiscounted_returns.append(rewards_episode_sum)

            # Calculate the cumulative reward in current 
            sum_of_returns += rewards_episode_sum

            # Halt computation cumulative reward is worse than ref
            if i >= GRACE and REF_CUMULATIVE_FITNESSES[i - GRACE] > sum_of_returns:
                break

            termination.append(float(any(step_type == StepType.TERMINAL for step_type in eps.step_types)))
            if 'success' in eps.env_infos:
                success.append(float(eps.env_infos['success'].any()))

        TOTAL_COMPUTED_EPISODES += i

        # Updating ref fitness.
        if rewards_episode_sum > REF_CUMULATIVE_FITNESSES[-1]:
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

        if itr%n_samples == n_samples-1:
            runtimes = "("+";".join(map(str, RUNTIMES))+")"
            with open("result.txt", "a") as f:
                print("seed_"+str(seed)+"_gymEnvName_"+gymEnvName, REF_CUMULATIVE_FITNESSES[-1], time.time() - START_REF_TIME, TOTAL_COMPUTED_EPISODES, itr, runtimes , file=f, sep=",", end="\n")
            RUNTIMES = []



        return undiscounted_returns



    cma_es_gym = None
    @wrap_experiment
    def cma_es_gym(ctxt=None, gymEnvName='CartPole-v1', earlyStop=True, seed=1):


        if earlyStop:
            garage.log_performance = custom_log_performance
        else:
            garage.log_performance = og_log_performance

        from garage.envs import GymEnv
        from garage.experiment.deterministic import set_seed
        from garage.np.algos import CMAES
        from garage.sampler import LocalSampler
        from garage.tf.policies import CategoricalMLPPolicy
        from garage.trainer import TFTrainer


        set_seed(seed)

        with TFTrainer(ctxt) as trainer:
            global MAX_EPISODE_LENGTH
            env = GymEnv(gymEnvName, max_episode_length=MAX_EPISODE_LENGTH)
            
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
            sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
            algo = CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=n_samples)

            trainer.setup(algo, env)
            trainer.train(n_epochs=n_epochs, batch_size=batch_size)


            # runtimes = "("+";".join(map(str, self.frames_in_gen))+")"
            # print("seed_"+str(self.seed)+"_level_"+self.level, self.best_fitness, time.time() -self.sw, self.total_frames, self.evals, runtimes , file=f, sep=",", end="\n")

    cma_es_gym()

if __name__ == "__main__":
    launch_experiment(earlyStop=True, seed=1)
    
