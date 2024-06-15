class Agent:
    def __init__(self, name, task, device, algo):
        self.name = name
        self.task = task
        self.device = device
        self.policy = algo

    def __repr__(self):
        raise NotImplementedError

    def update(self, replay_iter, step):
        raise NotImplementedError

    # def act(self, obs, step, eval_mode, eval_eps=None):
    #     return self.policy.act(obs, step, eval_mode, eval_eps=eval_eps)

    def act(self, obs, step, eval_mode):
        return self.policy.act(obs, step, eval_mode)
