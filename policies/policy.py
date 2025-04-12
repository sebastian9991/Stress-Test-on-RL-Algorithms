from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def select_action(self):
        pass
