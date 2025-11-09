import numpy as np

class VNFEnvironment:
    def __init__(self, vnf_list, vnf_to_models, vnf_priority, constraints, rewarder):
        self.vnf_list = vnf_list
        self.vnf_to_models = vnf_to_models
        self.vnf_priority = vnf_priority
        self.num_vnfs = len(vnf_list)
        self.num_models = len(vnf_to_models[vnf_list[0]])
        self.cpu_budget = constraints["CPU_BUDGET"]
        self.mem_budget = constraints["MEM_BUDGET"]
        self.gpu_budget = constraints["GPU_BUDGET"]
        self.latency_sla = constraints["LATENCY_SLA"]
        self.rewarder = rewarder
        self.reset()

    def reset(self):
        self.idx = 0
        self.chosen = []
        self.cpu_used = 0
        self.mem_used = 0
        self.gpu_used = 0
        self.lat_used = 0.0
        return self._get_obs()

    def _get_obs(self):
        v = np.zeros(self.num_vnfs, dtype=float)
        v[self.idx] = 1.0

        cpu_rem = max(0.0, self.cpu_budget - self.cpu_used) / max(1, self.cpu_budget)
        mem_rem = max(0.0, self.mem_budget - self.mem_used) / max(1, self.mem_budget)
        gpu_rem = max(0.0, self.gpu_budget - self.gpu_used) / max(1, self.gpu_budget) if self.gpu_budget else 0.0
        lat_rem = max(0.0, self.latency_sla - self.lat_used) / max(1.0, self.latency_sla)

        # FEATURES FOR AVAILABLE MODELS FOR THIS VNF
        vnf = self.vnf_list[self.idx]
        model_feats = []
        for model in self.vnf_to_models[vnf]:
            model_feats.append([
                float(model['accuracy']),
                float(model['cpu_cycles'])/self.cpu_budget,
                float(model['memory_mb'])/self.mem_budget,
                float(model['gpu_cores'])/max(1,self.gpu_budget),
                float(model['latency_ms'])/self.latency_sla
            ])
        model_feats = np.array(model_feats).flatten()
        return np.concatenate([v, [cpu_rem, mem_rem, gpu_rem, lat_rem], model_feats])

    def compute_reward(self):
        cpu_ratio = self.cpu_used / self.cpu_budget
        mem_ratio = self.mem_used / self.mem_budget
        gpu_ratio = self.gpu_used / self.gpu_budget if self.gpu_budget else 0.0
        lat_ratio = self.lat_used / self.latency_sla

        return self.rewarder.compute_reward(self.chosen, self.vnf_to_models, self.vnf_priority, cpu_ratio, mem_ratio, gpu_ratio, lat_ratio)

    def step(self, action):
        vnf = self.vnf_list[self.idx]
        model = self.vnf_to_models[vnf][action]

        self.cpu_used += int(model['cpu_cycles'])
        self.mem_used += int(model['memory_mb'])
        self.gpu_used += int(model['gpu_cores'])
        self.lat_used += float(model['latency_ms'])

        self.chosen.append((vnf, action))
        self.idx += 1

        done = (self.idx >= self.num_vnfs)
        if done:
            reward = self.compute_reward()
            obs = np.zeros(self.num_vnfs + 4 + self.num_models*5, dtype=float)
        else:
            reward = 0.0
            obs = self._get_obs()

        return obs, reward, done

