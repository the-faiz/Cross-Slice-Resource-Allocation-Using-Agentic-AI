import abc

class BaseAgent(abc.ABC):
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def select_action(self, obs):
        pass
