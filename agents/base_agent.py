import abc

class BaseAgent(abc.ABC):
    def __init__(self, env):
        self.env = env
