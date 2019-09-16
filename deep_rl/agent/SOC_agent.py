from ..network import *
from ..component import *
from .BaseAgent import *


class SOCAgent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0

        self.worker_index = tensor(np.arange(config.num_workers)).long()
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = tensor(np.zeros(config.num_workers)).long()

        self.count = 0

        self.all_options = []

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()

            self.logger.add_scalar('beta', prediction['beta'][self.worker_index, self.prev_options], log_level=5)
            self.logger.add_scalar('option', options[0], log_level=5)
            self.logger.add_scalar('pi_hat_ent', dist.entropy(), log_level=5)
            self.logger.add_scalar('pi_hat_o', dist.log_prob(options).exp(), log_level=5)

            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()

            pi_bar = self.compute_pi_bar(options.unsqueeze(-1), actions,
                                         prediction['mean'], prediction['std'])

            next_states, rewards, terminals, info = self.task.step(to_np(actions))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)

            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'a': actions,
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         's': tensor(states),
                         'init': self.is_initial_states.unsqueeze(-1),
                         'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1),
                         'log_pi_bar': pi_bar.add(1e-5).log(),
                         })

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
        dist = torch.distributions.Categorical(pi_hat)
        options = dist.sample()

        storage.add(prediction)
        storage.add({
            'v': prediction['q_o'][self.worker_index, options].unsqueeze(-1)
        })
        storage.placeholder()

        self.compute_adv(storage)
        self.learn(storage)
