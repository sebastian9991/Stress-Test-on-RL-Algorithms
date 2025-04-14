from policy import Policy


class Deterministic(Policy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self):
        pass
