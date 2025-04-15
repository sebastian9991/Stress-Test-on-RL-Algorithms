from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, policy_net):
        self.policy_net = policy_net

    @abstractmethod
    def select_action(self, state):
        pass
