import numpy as np
import random

class EvolutionaryAgent:
    def __init__(self, env, population_size=50, generations=100, mutation_rate=0.1, elite_frac=0.2):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.num_actions = env.num_models
        self.num_vnfs = env.num_vnfs
        self.best_actions = None
        self.best_reward = -1e9

    def _evaluate(self, actions):
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        for a in actions:
            obs, r, done = self.env.step(a)
            total_reward += r
            if done:
                break
        return total_reward

    def _mutate(self, individual):
        new_individual = individual.copy()
        for i in range(self.num_vnfs):
            if np.random.rand() < self.mutation_rate:
                new_individual[i] = np.random.randint(self.num_actions)
        return new_individual

    def _crossover(self, parent1, parent2):
        point = np.random.randint(1, self.num_vnfs - 1)
        child = parent1[:point] + parent2[point:]
        return child

    def train(self):
        population = [
            [np.random.randint(self.num_actions) for _ in range(self.num_vnfs)]
            for _ in range(self.population_size)
        ]

        for gen in range(self.generations):
            scores = [self._evaluate(ind) for ind in population]
            elite_count = int(self.elite_frac * self.population_size)
            elite_indices = np.argsort(scores)[-elite_count:]
            elites = [population[i] for i in elite_indices]

            if max(scores) > self.best_reward:
                self.best_reward = max(scores)
                self.best_actions = population[np.argmax(scores)]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(elites, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"Gen {gen+1}/{self.generations} | Best Reward: {self.best_reward:.4f}")

        print("Training completed.")
        print("Best reward:", self.best_reward)
        print("Best actions:", self.best_actions)
        return self.best_reward, self.best_actions

    def evaluate_agent(self):
        if self.best_actions is None:
            print("No trained policy found. Run train() first.")
            return

        reward = self._evaluate(self.best_actions)
        print("Evaluation reward:", reward)
        for i, a in enumerate(self.best_actions):
            vnf = self.env.vnf_list[i]
            model = self.env.vnf_to_models[vnf][a]
            print((vnf, model["model"], model["accuracy"]))

    def print_selection_example(self):
        if self.best_actions is None:
            print("No selected policy available.")
            return

        print("\nSelected Model Combination:")
        for i, a in enumerate(self.best_actions):
            vnf = self.env.vnf_list[i]
            model = self.env.vnf_to_models[vnf][a]
            print((vnf, model["model"], model["accuracy"]))