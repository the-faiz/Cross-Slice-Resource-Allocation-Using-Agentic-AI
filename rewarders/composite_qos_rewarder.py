import numpy as np
from .base_rewarder import BaseRewarder

from utils.logger_config import get_logger
logger = get_logger("COMPQoSRWD")

class CompositeQoSRewarder(BaseRewarder):
    """
    Reward balances accuracy, QoS, and resource efficiency.

    Formula mirrors user's _compute_reward:
      reward = (8.0 * qos_gain)
               - (0.8 * usage_penalty)
               - (overflow_penalty)
               - (0.6 * low_acc_pen)
    """
    def compute_reward(self, chosen, vnf_to_models, vnf_priority, cpu_ratio, mem_ratio, gpu_ratio, lat_ratio):
        weighted_sum = 0.0
        weight_total = 0.0
        low_acc_pen = 0.0

        # Compute weighted QoS
        for (v, i) in chosen:
            w = vnf_priority.get(v, 1.0)
            acc = float(vnf_to_models[v][i]["accuracy"])

            weighted_sum += w * acc
            weight_total += w
            low_acc_pen += (1.0 - acc) * w

        qos_score = weighted_sum / weight_total if weight_total else 0.0

        # Resource usage cost (quadratic)
        usage_penalty = (
            0.4 * cpu_ratio**2 +
            0.3 * mem_ratio**2 +
            0.2 * gpu_ratio**2 +
            0.9 * lat_ratio**2
        )

        # Overflow punishment
        overflow_penalty = (
            max(0, cpu_ratio - 1) * 5 +
            max(0, mem_ratio - 1) * 5 +
            max(0, gpu_ratio - 1) * 4 +
            max(0, lat_ratio - 1) * 8
        )

        # Amplify accuracy importance
        qos_gain = qos_score ** 2.2

        # Final reward
        reward = (
            8.0 * qos_gain
            - 0.8 * usage_penalty
            - overflow_penalty
            - 0.6 * low_acc_pen
        )

        return float(reward)
