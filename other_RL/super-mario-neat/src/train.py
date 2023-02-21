import os
import neat
import gym, ppaquette_gym_super_mario
import pickle
import multiprocessing as mp
import visualize
import time
import numpy as np
import random
import sys

gym.logger.set_level(40)

FITNESS_REF_ARRAY_SIZE = 1001
MAX_EPISODE_LENGTH = 1000

class Train:
    def __init__(self, method:str, generations:int, seed:int, filename:str, level:str="1-1", gracetime:int=None,  fincrementsize:int=None):
        self.actions = [
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1],
        ]
        print("Initialize Train:")
        self.seed = seed
        random.seed(seed)
        self.best_fitness = 0
        self.sw = time.time()
        self.generations = generations
        self.lock = mp.Lock()
        self.level = level
        self.observed_fitnesses = np.zeros(FITNESS_REF_ARRAY_SIZE, dtype=np.int64)
        print("zeroe ref fitnesses.")
        self.ref_fitnesses = np.zeros(FITNESS_REF_ARRAY_SIZE, dtype=np.int64)
        self.total_frames = 0
        self.evals = 0
        self.frames_in_gen = []
        self.filename = filename
        print(method, gracetime)
        assert method in ("constant", "nokill", "bestasref")
        assert not (method == "bestasref" and gracetime is None)
        self.method = method
        self.time_grace = gracetime
        self.fincrementsize = fincrementsize



    def _get_actions(self, a):
        return self.actions[a.index(max(a))]

    def _fitness_func(self, genome, config, o = None):
        env = gym.make('ppaquette/SuperMarioBros-'+self.level+'-Tiles-v0')
        # env.configure(lock=self.lock)
        try:
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            old = 0
            self.evals += 1
            self.observed_fitnesses[:] = 0
            np.set_printoptions(threshold=sys.maxsize)

            while not done:
                self.total_frames += i
                state = state.flatten()
                output = net.activate(state)
                output = self._get_actions(output)
                s, reward, done, info = env.step(output)
                distance = info['distance'] # the distance is the fitness
                if not self.fincrementsize is None:
                    distance = distance - (distance % self.fincrementsize)
                self.observed_fitnesses[i] = distance
                state = s
                i += 1
                if i > MAX_EPISODE_LENGTH:
                    break

                if self.method == "constant":
                    if i % 50 == 0:
                        if old == distance:
                            break
                        else:
                            old = distance
                elif self.method == "bestasref":

                    # import code
                    # code.interact(local=locals())

                    if i > self.time_grace and self.ref_fitnesses[i - self.time_grace] > distance:
                        break
                elif self.method == "nokill":
                    pass
                else:
                    raise ValueError("self.method =" + self.method + "not recognized.")

            # [print(str(i) + " : " + str(info[i]), end=" ") for i in info.keys()]
            # print("\n******************************")
            self.frames_in_gen.append(i)
            fitness = -1 if distance <= 40 else distance
            genome.fitness = fitness
            if self.best_fitness < fitness:
                self.best_fitness = fitness
                self.ref_fitnesses[:] = self.observed_fitnesses[:]
                self.ref_fitnesses[(i-1):] = np.repeat(self.ref_fitnesses[(i-1)], FITNESS_REF_ARRAY_SIZE)[(i-1):]

            
            if not o is None:
                o.put(fitness)
            env.close()
        except KeyboardInterrupt:
            env.close()
            exit()

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        # # Parallel
        # for i in range(0, len(genomes), self.par):
        #     output = mp.Queue()

        #     processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in
        #                  genomes[i:i + self.par]]

        #     [p.start() for p in processes]
        #     [p.join() for p in processes]

        #     results = [output.get() for p in processes]

        #     for n, r in enumerate(results):
        #         genomes[i + n].fitness = r
        # with open("resultSuperMario.txt", "a") as f:
        #     print("fitnesses", [genomes[i].fitness for i in range(len(genomes))])

        # Sequential
        self.frames_in_gen = []

        for i in range(len(genomes)):
            self._fitness_func(genomes[i], config)
        
        with open(self.filename, "a") as f:
            runtimes = "("+";".join(map(str, self.frames_in_gen))+")"
            print("seed_"+str(self.seed)+"_level_"+self.level, self.best_fitness, time.time() -self.sw, self.total_frames, self.evals, runtimes , file=f, sep=",", end="\n")

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        # p.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        print("loaded checkpoint...")
        winner = p.run(self._eval_genomes, n)
        win = p.best_genome
        # pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open(self.filename.replace(".txt", ".pkl"), 'wb'))
        
        # visualize.draw_net(config, winner, True)
        # visualize.plot_stats(stats, ylog=False, view=True)
        # visualize.plot_species(stats, view=True)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)


if __name__ == "__main__":
    t = Train(1000)
    t.main()
