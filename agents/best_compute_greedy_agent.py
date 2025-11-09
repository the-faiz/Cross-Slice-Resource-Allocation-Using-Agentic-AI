import numpy as np
from .base_agent import BaseAgent

class BestComputeAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        self.best_actions = []

    def train(self):
        self.best_actions = []
        for vnf in self.env.vnf_list:
            models = self.env.vnf_to_models[vnf]
            # Compute cost = weighted sum of compute metrics
            compute_costs = [
                float(m["latency_ms"]) +
                float(m["cpu_cycles"]) +
                float(m["gpu_cores"]) +
                float(m["memory_mb"])
                for m in models
            ]
            best_idx = int(np.argmin(compute_costs))
            self.best_actions.append(best_idx)

    def evaluate_agent(self, trials=1):
        total_reward = 0.0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            idx = 0
            while not done:
                action = self.best_actions[idx]
                obs, reward, done = self.env.step(action)
                idx += 1
            total_reward += reward

        avg_reward = total_reward / trials
        print(f"Average reward (BestComputeAgent): {avg_reward:.4f}")

    def print_selection_example(self):
        self.env.reset()
        print("Best Compute Model Selection:")
        for i, vnf in enumerate(self.env.vnf_list):
            model = self.env.vnf_to_models[vnf][self.best_actions[i]]
            print((vnf, model["model"], model["latency_ms"], model["cpu_cycles"], model["gpu_cores"], model["memory_mb"]))
