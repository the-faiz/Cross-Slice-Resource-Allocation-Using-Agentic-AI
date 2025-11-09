import numpy as np
from .base_agent import BaseAgent

class A2CNetwork:
    """Simple shared-parameter Actor-Critic network."""
    def __init__(self, obs_dim, n_actions, hidden=64, seed=42):
        rng = np.random.RandomState(seed)
        # Shared layer
        self.W1 = rng.normal(scale=0.1, size=(obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        # Actor head
        self.Wa = rng.normal(scale=0.1, size=(hidden, n_actions))
        self.ba = np.zeros(n_actions)
        # Critic head
        self.Wv = rng.normal(scale=0.1, size=(hidden, 1))
        self.bv = np.zeros(1)

    def forward(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.Wa + self.ba
        value = (h @ self.Wv + self.bv).item()
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return probs, value, h

    def backward(self, obs, h, probs, a, td_error, advantage, lr=0.001):
        # Gradients for actor (policy)
        dlogpi = -probs
        dlogpi[a] += 1.0
        g_actor = advantage * np.outer(h, dlogpi)

        # Gradients for critic (value)
        g_critic = td_error * np.outer(h, np.ones(1))

        # Update actor weights
        self.Wa += lr * g_actor
        self.ba += lr * advantage * dlogpi

        # Update critic weights
        self.Wv += lr * g_critic
        self.bv += lr * td_error

        # Backprop through shared layer (simplified)
        grad_shared = (self.Wa @ dlogpi + self.Wv.flatten() * td_error) * (1 - h ** 2)
        self.W1 += lr * np.outer(obs, grad_shared)
        self.b1 += lr * grad_shared


class A2CAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, lr=0.001):
        super().__init__(env)
        self.gamma = gamma
        self.lr = lr

        obs_dim = len(env.reset())
        n_actions = env.num_models
        self.net = A2CNetwork(obs_dim, n_actions)

    def train(self, epochs=200, batch_size=16):
        for epoch in range(epochs):
            total_reward = 0.0

            for _ in range(batch_size):
                obs = self.env.reset()
                done = False
                ep_obs, ep_act, ep_rew, ep_val = [], [], [], []

                while not done:
                    probs, value, h = self.net.forward(obs)
                    a = np.random.choice(len(probs), p=probs)
                    next_obs, r, done = self.env.step(a)

                    ep_obs.append((obs, h))
                    ep_act.append(a)
                    ep_rew.append(r)
                    ep_val.append(value)

                    obs = next_obs
                    total_reward += r

                # Compute discounted returns
                returns = []
                G = 0
                for r in reversed(ep_rew):
                    G = r + self.gamma * G
                    returns.insert(0, G)

                # Update
                for (obs, h), a, R, V in zip(ep_obs, ep_act, returns, ep_val):
                    td_error = R - V
                    advantage = td_error
                    probs, _, _ = self.net.forward(obs)
                    self.net.backward(obs, h, probs, a, td_error, advantage, self.lr)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, total reward: {total_reward / batch_size:.4f}")

    def evaluate_agent(self, trials=200):
        total = 0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            while not done:
                probs, _, _ = self.net.forward(obs)
                a = int(np.argmax(probs))
                obs, r, done = self.env.step(a)
            total += r
        avg = total / trials
        print(f"A2C average reward over {trials} evals: {avg:.4f}")

    def print_selection_example(self):
        obs = self.env.reset()
        selection, actions = [], []
        done = False

        while not done:
            probs, _, _ = self.net.forward(obs)
            a = int(np.argmax(probs))
            vnf = self.env.vnf_list[self.env.idx]
            model = self.env.vnf_to_models[vnf][a]
            selection.append((vnf, model["model"], model["accuracy"]))
            obs, _, done = self.env.step(a)
            actions.append(a)

        print("A2C Example Actions:", actions)
        for s in selection:
            print(s)
