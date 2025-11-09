import abc

class BaseRewarder(abc.ABC):
    """Base class for all reward computation strategies."""

    @abc.abstractmethod
    def compute_reward(chosen, vnf_to_models, vnf_priority, cpu_ratio, mem_ratio, gpu_ratio, lat_ratio):
        pass

