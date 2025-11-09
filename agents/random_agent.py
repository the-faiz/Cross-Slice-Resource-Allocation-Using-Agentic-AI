import numpy as np
import logging

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.logger = logging.getLogger("RandomAgent")
        self.actions = []

    def act(self, obs):
        return np.random.randint(self.env.num_models)

    def train(self):
        # Random agent doesn't learn
        self.logger.info("RandomAgent does not require training.")

    def evaluate_agent(self, episodes=100):
        rewards = []
        actions_log = []
        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            episode_actions = []
            while not done:
                action = self.act(obs)
                obs, reward, done = self.env.step(action)
                episode_actions.append(action)
                total_reward += reward
            rewards.append(total_reward)
            actions_log = episode_actions  # keep last run's actions for printing
        avg_reward = np.mean(rewards)
        self.logger.info(f"Evaluation over {episodes} runs: avg reward = {avg_reward:.4f}")
        print(f"Random Agent avg reward over {episodes} runs: {avg_reward:.4f}")
        self.actions = actions_log
        return avg_reward

    def print_selection_example(self):
        print("Random Agent Example actions:", self.actions)
        for (vnf, a) in zip(self.env.vnf_list, self.actions):
            model = self.env.vnf_to_models[vnf][a]
            print((vnf, model["model"], model["accuracy"]))
