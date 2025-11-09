import numpy as np

class RandomSearchAgent:
    def __init__(self, env, trials=50000):
        self.env = env
        self.trials = trials
        self.best_reward = -1e9
        self.best_actions = None

    def train(self):
        for _ in range(self.trials):
            obs = self.env.reset()
            episode_r = 0.0
            done = False
            actions_taken = []

            while not done:
                a = np.random.randint(self.env.num_models)
                actions_taken.append(a)
                obs, r, done = self.env.step(a)
                episode_r += r

            if episode_r > self.best_reward:
                self.best_reward = episode_r
                self.best_actions = actions_taken[:]

        print(f"Best reward from random search: {self.best_reward:.4f}")

    def evaluate_agent(self, runs=100):
        """Evaluate the best action sequence found during training."""
        if self.best_actions is None:
            print("Agent not trained yet.")
            return

        total_reward = 0.0
        for _ in range(runs):
            obs = self.env.reset()
            done = False
            i = 0
            while not done:
                a = self.best_actions[i]
                obs, r, done = self.env.step(a)
                total_reward += r
                i += 1

        avg_reward = total_reward / runs
        print(f"Evaluation avg reward over {runs} runs: {avg_reward:.4f}")
        return avg_reward

    def print_selection_example(self):
        print("Best Actions:", self.best_actions)
        for vnf, action in zip(self.env.vnf_list, self.best_actions):
            model = self.env.vnf_to_models[vnf][action]
            print((vnf, model["model"], model["accuracy"]))
