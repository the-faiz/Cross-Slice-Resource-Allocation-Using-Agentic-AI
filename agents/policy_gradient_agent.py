import numpy as np
from .base_agent import BaseAgent

class PolicyGradientAgent(BaseAgent):
    def __init__(self, env, lr=0.05, gamma=0.99, temp=0.7):
        super().__init__(env)
        self.lr = lr
        self.gamma = gamma
        self.temp = temp

        obs_dim = len(env.reset())
        n_actions = env.num_models

        rng = np.random.RandomState(0)
        self.W = rng.normal(scale=0.01, size=(obs_dim, n_actions))
        self.b = np.zeros(n_actions)

    def softmax(self, logits):
        z = logits - np.max(logits)  # numerical stability
        exp = np.exp(z)
        return exp / np.sum(exp)

    def select_action(self, obs):
        probs = self.softmax(obs @ self.W + self.b)
        a = np.random.choice(len(probs), p=probs)
        return a, probs

    def discounted_returns(self, rewards):
        G = 0.0
        out = np.zeros(len(rewards))
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            out[i] = G
        return out

    def _episode_run(self):
        obs = self.env.reset()
        obs_seq, act_seq, prob_seq, rew_seq = [], [], [], []
        done = False
        while not done:
            a, probs = self.select_action(obs)
            obs_seq.append(obs.copy())
            act_seq.append(a)
            prob_seq.append(probs)
            obs, r, done = self.env.step(a)
            rew_seq.append(r)
        return obs_seq, act_seq, prob_seq, rew_seq

    def train(self, epochs=100, batch_size=32):
        for epoch in range(epochs):
            all_obs = []
            all_actions = []
            all_probs = []
            all_returns = []
            batch_start_returns = []

            for _ in range(batch_size):
                obs_seq, act_seq, prob_seq, rew_seq = self._episode_run()
                returns = self.discounted_returns(rew_seq)

                all_obs.extend(obs_seq)
                all_actions.extend(act_seq)
                all_probs.extend(prob_seq)
                all_returns.extend(returns)
                batch_start_returns.append(returns[0])

            adv = np.array(all_returns)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            gW = np.zeros_like(self.W)
            gb = np.zeros_like(self.b)

            for obs, a, probs, A in zip(all_obs, all_actions, all_probs, adv):
                grad_logit = np.zeros(len(probs))
                grad_logit[a] = 1.0
                grad_logit -= probs
                gW += A * np.outer(obs, grad_logit)
                gb += A * grad_logit

            gW /= batch_size
            gb /= batch_size

            norm = np.linalg.norm(gW)
            if norm > 1.0:
                gW /= norm
                gb /= norm

            self.W += self.lr * gW
            self.b += self.lr * gb

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, mean reward = {np.mean(batch_start_returns):.4f}")

    def evaluate_agent(self, trials=200):
        total = 0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            while not done:
                logits = obs.dot(self.W) + self.b
                action = int(np.argmax(logits))
                obs, reward, done = self.env.step(action)
            total += reward
        avg_reward = total/trials
        print(f"Average policy gradient reward over {trials} episodes = {avg_reward}")

    def print_selection_example(self):
        obs = self.env.reset()
        selection = []
        done = False
        actions = []
        while not done:
            logits = obs.dot(self.W) + self.b
            action = int(np.argmax(logits))
            vnf = self.env.vnf_list[self.env.idx]
            selection.append((vnf, self.env.vnf_to_models[vnf][action]['model'], self.env.vnf_to_models[vnf][action]['accuracy']))
            obs, reward, done = self.env.step(action)
            actions.append(action)
        
        print("Example Selection Policy Gradient Agent : ")
        print(actions)
        for s in selection:
            print(s)
