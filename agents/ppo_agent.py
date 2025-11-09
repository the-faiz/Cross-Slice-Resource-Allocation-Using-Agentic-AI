import numpy as np
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self,env,actor_hidden=(128,),critic_hidden=(128,),lr_actor=3e-4,lr_critic=1e-3,gamma=0.99,
                lam=0.95,clip_eps=0.2,epochs=10,minibatch_size=64,entropy_coef=0.01,value_coef=0.5,
                max_grad_norm=None,seed=42):
        super().__init__(env)
        self.env = env
        self.obs_dim = len(env.reset())
        self.n_actions = env.num_models

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        rng = np.random.default_rng(seed)
        # actor network sizes
        actor_sizes = [self.obs_dim] + list(actor_hidden) + [self.n_actions]
        critic_sizes = [self.obs_dim] + list(critic_hidden) + [1]

        def init_mlp(sizes):
            params = []
            for i in range(len(sizes)-1):
                in_dim, out_dim = sizes[i], sizes[i+1]
                w = rng.normal(0, np.sqrt(2/(in_dim+out_dim)), size=(in_dim, out_dim))
                b = np.zeros(out_dim)
                params.append([w, b])
            return params

        self.actor_params = init_mlp(actor_sizes)
        self.critic_params = init_mlp(critic_sizes)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.rng = rng

    # ---- forward utils ----
    def _mlp_forward(self, params, x):
        a = x
        caches = []
        for i, (W, b) in enumerate(params):
            z = a @ W + b
            caches.append((a, z))
            if i < len(params)-1:
                a = np.tanh(z)
            else:
                a = z
        return a, caches

    def _mlp_backward(self, params, caches, grad_output):
        grads = [None] * len(params)
        delta = grad_output.copy()
        for i in reversed(range(len(params))):
            a_prev, z = caches[i]
            dW = a_prev.T @ delta / delta.shape[0]
            db = np.mean(delta, axis=0)
            grads[i] = [dW, db]
            if i > 0:
                W = params[i][0]
                da_prev = delta @ W.T
                delta = da_prev * (1 - np.tanh(caches[i-1][1])**2)
        return grads

    # ---- policy / value wrappers ----
    def _policy_logits(self, obs):
        q, _ = self._mlp_forward(self.actor_params, obs)
        return q  # raw logits

    def _value(self, obs):
        v, _ = self._mlp_forward(self.critic_params, obs)
        return v[:, 0]

    def _softmax(self, logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def get_action(self, obs):
        obs = obs.reshape(1, -1)
        logits, _ = self._mlp_forward(self.actor_params, obs)
        probs = self._softmax(logits)
        a = self.rng.choice(self.n_actions, p=probs[0])
        return int(a), float(probs[0, a]), probs[0].copy()

    # ---- advantage (GAE) ----
    def compute_gae(self, rewards, values, dones):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            nextvalue = values[t+1] if t+1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * nextvalue * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values[:len(adv)]
        return adv, returns

    # ---- PPO update (vectorized minibatch) ----
    def ppo_update(self, obs_buf, act_buf, old_logp_buf, adv_buf, ret_buf, val_buf):
        N = obs_buf.shape[0]
        inds = np.arange(N)
        # normalize advantage
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        for _ in range(self.epochs):
            self.rng.shuffle(inds)
            for start in range(0, N, self.minibatch_size):
                mb_idx = inds[start:start+self.minibatch_size]
                obs_mb = obs_buf[mb_idx]
                act_mb = act_buf[mb_idx]
                old_logp_mb = old_logp_buf[mb_idx]
                adv_mb = adv_buf[mb_idx]
                ret_mb = ret_buf[mb_idx]

                # --- policy forward ---
                logits, actor_caches = self._mlp_forward(self.actor_params, obs_mb)
                probs = self._softmax(logits)
                # new log prob of taken actions
                new_prob_act = probs[np.arange(len(act_mb)), act_mb]
                new_logp = np.log(new_prob_act + 1e-12)
                old_logp = old_logp_mb
                ratio = np.exp(new_logp - old_logp)

                surrogate1 = ratio * adv_mb
                surrogate2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                policy_loss = -np.mean(np.minimum(surrogate1, surrogate2))

                # entropy for exploration
                entropy = -np.mean(np.sum(probs * np.log(probs + 1e-12), axis=1))
                # --- value forward ---
                values_pred, critic_caches = self._mlp_forward(self.critic_params, obs_mb)
                values_pred = values_pred[:, 0]
                value_loss = np.mean((ret_mb - values_pred)**2)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # --- compute gradients manually ---
                # Policy gradient part: derivative of -mean(min(...))
                # We compute derivative of surrogate using the branch chosen for each sample
                chosen_clip = (np.abs(ratio - 1.0) > 0)
                # for grad calculation we pick which term was used per sample
                use_clip = np.abs(surrogate2 - surrogate1) < 0  # not used; simpler: use surrogate2 when clipped helps reduce objective
                # Simpler correct approach: compute grad on surrogate1 if |ratio-1|<=clip_eps else surrogate2. Use mask:
                use_clip_mask = (ratio > 1.0 + self.clip_eps) | (ratio < 1.0 - self.clip_eps)
                # grad of logpi wrt logits: dlogpi/dlogits = one_hot - probs
                # d loss w.r.t logits = - (weight) * (one_hot - probs), weight = coeff per sample
                coeff = np.where(use_clip_mask, np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb, ratio * adv_mb)
                coeff = -coeff / len(act_mb)  # averaged gradient sign for policy loss

                grad_logit = np.zeros_like(probs)
                grad_logit[np.arange(len(act_mb)), act_mb] = 1.0
                grad_logit -= probs  # shape (B, n_actions)
                grad_logit *= coeff[:, None]  # scale each row

                # backprop policy
                actor_grads = self._mlp_backward(self.actor_params, actor_caches, grad_logit)
                # backprop value
                dvalue = (values_pred - ret_mb)[:, None] * (2.0 / len(ret_mb))  # mean MSE grad
                critic_grads = self._mlp_backward(self.critic_params, critic_caches, dvalue)

                # apply grads (SGD)
                for i in range(len(self.actor_params)):
                    self.actor_params[i][0] -= self.lr_actor * actor_grads[i][0]
                    self.actor_params[i][1] -= self.lr_actor * actor_grads[i][1]
                for i in range(len(self.critic_params)):
                    self.critic_params[i][0] -= self.lr_critic * critic_grads[i][0]
                    self.critic_params[i][1] -= self.lr_critic * critic_grads[i][1]

    # ---- rollout collection ----
    def collect_episodes(self, min_steps):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        steps = 0
        while steps < min_steps:
            obs = self.env.reset()
            done = False
            ep_obs, ep_acts, ep_rews, ep_vals, ep_logps, ep_dones = [], [], [], [], [], []
            while not done:
                a, prob = self.get_action(obs)[:2]
                val = float(self._value(obs.reshape(1, -1))[0])
                next_obs, r, done = self.env.step(a)
                logp = np.log(prob + 1e-12)
                ep_obs.append(obs.copy())
                ep_acts.append(a)
                ep_rews.append(r)
                ep_vals.append(val)
                ep_logps.append(logp)
                ep_dones.append(float(done))
                obs = next_obs if next_obs is not None else np.zeros(self.obs_dim)
                steps += 1
            # append last value bootstrap
            ep_vals.append(0.0)
            obs_buf.extend(ep_obs)
            act_buf.extend(ep_acts)
            rew_buf.extend(ep_rews)
            val_buf.extend(ep_vals)
            logp_buf.extend(ep_logps)
            done_buf.extend(ep_dones)
        return (np.array(obs_buf, dtype=np.float32),
                np.array(act_buf, dtype=np.int32),
                np.array(rew_buf, dtype=np.float32),
                np.array(val_buf, dtype=np.float32),
                np.array(logp_buf, dtype=np.float32),
                np.array(done_buf, dtype=np.float32))

    # ---- training loop ----
    def train(self, total_steps=20000, batch_steps=2048, log_interval=2048):
        steps = 0
        ep = 0
        while steps < total_steps:
            obs_b, act_b, rew_b, val_b, logp_b, done_b = self.collect_episodes(batch_steps)
            adv_b, ret_b = self.compute_gae(rew_b, val_b, done_b)
            # ppo update
            self.ppo_update(obs_b, act_b, logp_b, adv_b, ret_b, val_b)
            steps += len(obs_b)
            ep += 1
            if steps % log_interval == 0:
                print(f"PPO epoch {ep}, steps {steps}")

    # ---- evaluation / printing ----
    def evaluate_agent(self, trials=50):
        total = 0.0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            ep_r = 0.0
            while not done:
                logits, _ = self._mlp_forward(self.actor_params, obs.reshape(1, -1))
                probs = self._softmax(logits)
                a = int(np.argmax(probs[0]))
                obs, r, done = self.env.step(a)
                ep_r += r
            total += ep_r
        avg = total / trials
        print(f"PPO evaluate avg reward over {trials}: {avg:.4f}")
        return avg

    def print_selection_example(self):
        obs = self.env.reset()
        sel = []
        done = False
        actions = []
        while not done:
            logits, _ = self._mlp_forward(self.actor_params, obs.reshape(1, -1))
            probs = self._softmax(logits)
            a = int(np.argmax(probs[0]))
            vnf = self.env.vnf_list[self.env.idx]
            model_data = self.env.vnf_to_models[vnf][a]
            sel.append((vnf, model_data['model'], model_data['accuracy']))
            actions.append(a)
            obs, _, done = self.env.step(a)
        print("PPO example actions:", actions)
        for s in sel:
            print(s)
