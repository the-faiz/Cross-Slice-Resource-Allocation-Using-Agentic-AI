import numpy as np
from .base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self,env,hidden_sizes=(128, 64),lr=1e-3,gamma=0.99,buffer_size=20000,batch_size=64,
                 eps_start=1.0,eps_end=0.05,eps_decay_steps=10000,target_update_freq=1000,min_replay_size=500
    ):
        super().__init__(env)
        self.env = env
        self.obs_dim = len(env.reset())
        self.n_actions = env.num_models

        # network sizes
        hs = list(hidden_sizes)
        layer_sizes = [self.obs_dim] + hs + [self.n_actions]

        rng = np.random.default_rng(0)
        # params: list of (W,b) per layer for online and target
        def init_params():
            params = []
            for i in range(len(layer_sizes)-1):
                in_dim = layer_sizes[i]
                out_dim = layer_sizes[i+1]
                # xavier init
                w = rng.normal(0, np.sqrt(2/(in_dim+out_dim)), size=(in_dim, out_dim))
                b = np.zeros(out_dim)
                params.append([w, b])
            return params

        self.params = init_params()
        self.target_params = init_params()
        self._sync_target()

        # optimizer state: simple SGD
        self.lr = lr

        # replay buffer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = {
            "obs": np.zeros((buffer_size, self.obs_dim), dtype=np.float32),
            "act": np.zeros(buffer_size, dtype=np.int32),
            "rew": np.zeros(buffer_size, dtype=np.float32),
            "next_obs": np.zeros((buffer_size, self.obs_dim), dtype=np.float32),
            "done": np.zeros(buffer_size, dtype=np.float32),
        }
        self.ptr = 0
        self.size = 0
        self.min_replay_size = min_replay_size

        # exploration
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_decay = (eps_start - eps_end) / max(1, eps_decay_steps)
        self.gamma = gamma

        self.train_steps = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

    # --- network forward/backward (simple MLP) ---
    def _forward(self, params, x):
        a = x
        caches = []
        for i, (W, b) in enumerate(params):
            z = a @ W + b
            caches.append((a, z))
            if i < len(params)-1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = z  # last layer: linear q-values
        return a, caches

    def _compute_gradients(self, params, caches, grad_out):
        # caches: list of (a_prev, z) for each layer
        # grad_out: dL/dz_last   shape (batch, n_actions)
        grads = [None] * len(params)
        delta = grad_out.copy()
        for i in reversed(range(len(params))):
            a_prev, z = caches[i]
            # weight grad: a_prev^T @ delta
            dW = a_prev.T @ delta / delta.shape[0]
            db = np.mean(delta, axis=0)
            grads[i] = [dW, db]
            if i > 0:
                W = params[i][0]
                # propagate through ReLU: dprev = delta @ W.T * (a_prev>0)
                da_prev = delta @ W.T
                _, z_prev = caches[i-1]
                delta = da_prev * (z_prev > 0)
        return grads

    def _apply_grads(self, params, grads, lr):
        for i in range(len(params)):
            params[i][0] -= lr * grads[i][0]
            params[i][1] -= lr * grads[i][1]

    def _sync_target(self):
        # copy online params to target
        for i in range(len(self.params)):
            self.target_params[i][0] = self.params[i][0].copy()
            self.target_params[i][1] = self.params[i][1].copy()

    # --- buffer ops ---
    def store_transition(self, obs, act, rew, next_obs, done):
        idx = self.ptr
        self.buffer["obs"][idx] = obs
        self.buffer["act"][idx] = act
        self.buffer["rew"][idx] = rew
        self.buffer["next_obs"][idx] = next_obs if next_obs is not None else np.zeros_like(obs)
        self.buffer["done"][idx] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = {k: v[idxs].copy() for k, v in self.buffer.items()}
        return batch

    # --- policy ---
    def act_greedy(self, obs):
        q, _ = self._forward(self.params, obs[None, :])
        return int(np.argmax(q[0]))

    def act(self, obs, training=True):
        if training and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return self.act_greedy(obs)

    # --- learning step ---
    def train_step(self):
        if self.size < self.min_replay_size:
            return None
        batch = self.sample_batch()
        obs_b = batch["obs"]
        act_b = batch["act"]
        rew_b = batch["rew"]
        next_obs_b = batch["next_obs"]
        done_b = batch["done"]

        # Q(s,a)
        q_s, caches = self._forward(self.params, obs_b)       # (B, n_actions)
        q_s_a = q_s[np.arange(self.batch_size), act_b]

        # target: r + gamma * max_a' Q_target(next, a') * (1-done)
        q_next_target, _ = self._forward(self.target_params, next_obs_b)
        max_q_next = np.max(q_next_target, axis=1)
        target = rew_b + self.gamma * max_q_next * (1.0 - done_b)

        # loss = 0.5 * (q_s_a - target)^2
        td_error = (q_s_a - target)  # shape (B,)

        # compute gradient of loss w.r.t. q outputs: dL/dq = zero except for chosen action
        grad_q = np.zeros_like(q_s)  # (B, n_actions)
        grad_q[np.arange(self.batch_size), act_b] = td_error / self.batch_size  # MSE derivative divided by batch

        # backprop through online network
        grads = self._compute_gradients(self.params, caches, grad_q)
        self._apply_grads(self.params, grads, self.lr)

        # epsilon decay
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # target sync
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._sync_target()

        return float(np.mean(np.abs(td_error)))

    # --- run episodes to collect experience ---
    def rollout_episode(self, training=True):
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            a = self.act(obs, training=training)
            next_obs, reward, done = self.env.step(a)
            # store transition; next_obs may be None when done; store zero-vector then
            self.store_transition(obs, a, reward, next_obs if next_obs is not None else np.zeros_like(obs), done)
            obs = next_obs if next_obs is not None else np.zeros_like(obs)
            total_reward += reward
            steps += 1
        return total_reward, steps

    # --- main train loop ---
    def train(self, num_steps=20000, log_interval=1000):
        step = 0
        episode = 0
        recent_rewards = []
        while step < num_steps:
            ret, steps = self.rollout_episode(training=True)
            recent_rewards.append(ret)
            episode += 1
            step += steps

            # run a few SGD steps per episode
            for _ in range(max(1, steps // 1)):
                td = self.train_step()

            if episode % 10 == 0:
                avg_r = np.mean(recent_rewards[-50:]) if recent_rewards else 0.0
                print(f"Episode {episode:4d}, steps {step:6d}, avg_return(last50) {avg_r:.4f}, eps {self.eps:.3f}")

            if step % log_interval == 0 and step > 0:
                # occasional evaluation
                self.evaluate_agent(trials=20)
        print("Training finished.")

    def evaluate_agent(self, trials=100):
        total = 0.0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            ep_r = 0.0
            while not done:
                a = self.act_greedy(obs)
                obs, r, done = self.env.step(a)
                ep_r += r
            total += ep_r
        avg = total / trials
        print(f"Evaluation avg reward over {trials} runs: {avg:.4f}")
        return avg

    def print_selection_example(self):
        obs = self.env.reset()
        sel = []
        done = False
        actions = []
        while not done:
            a = self.act_greedy(obs)
            vnf = self.env.vnf_list[self.env.idx]
            model_data = self.env.vnf_to_models[vnf][a]
            sel.append((vnf, model_data['model'], model_data['accuracy']))
            actions.append(a)
            obs, _, done = self.env.step(a)
        print("DQN Example actions:", actions)
        for s in sel:
            print(s)
