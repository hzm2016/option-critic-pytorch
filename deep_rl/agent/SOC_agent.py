from ..network import *
from ..component import *
from .BaseAgent import *


class SOCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())

        self.total_steps = 0
        self.worker_index = tensor(np.arange(config.num_workers)).long()

        self.states = self.config.state_normalizer(self.task.reset())
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            # get q value
            q_option = torch.min(prediction['q_o_1'], prediction['q_o_1'])
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))

            # greedy policy
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()

            options = torch.where(is_intial_states, options, options_hat)
        return options

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

        # run one episode
        for _ in range(config.rollout_length):

            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)

            # select option
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)

            # Gaussian policy
            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)

            # select action
            actions = dist.sample()

            # entropy
            log_pi = dist.log_prob(actions).sum(-1).unsqueeze(-1)
            entropy = dist.entropy().sum(-1).unsqueeze(-1)

            # execute actions
            next_states, rewards, terminals, info = self.task.step(to_np(actions))

            # record data
            self.record_online_return(info)

            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy,
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'log_pi': log_pi,
                         'eps': epsilon})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            # target network
            prediction = self.target_network(self.states)
            storage.placeholder()
            betas = prediction['beta'][self.worker_index, self.prev_options]

            # option policy :: calculate the target
            ret = (1 - betas) * prediction['q_o'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q_o'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            # calculate option value with target network
            ret = storage.r[i] + config.discount * storage.m[i] * ret

            # calculate advantage value
            adv = ret - storage.q_o[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = storage.q_o[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + \
                storage.q_o[i].mean(-1).unsqueeze(-1) * storage.eps[i]
            q = storage.q_o[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + config.beta_reg

        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
            storage.cat(['q_o', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

        # calculate loss function
        q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi * adv.detach()) - config.entropy_weight * ent
        pi_loss = pi_loss.mean()
        beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() * (1 - initial_states)).mean()

        # backward and train
        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

        # run one episode
        for _ in range(config.rollout_length):

            prediction = self.network(self.states)
            epsilon = config.random_option_prob(config.num_workers)

            # select option
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)

            # Gaussian policy
            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)

            # select action
            actions = to_np(dist.sample())
            actions = np.clip(actions, self.task.action_space.low, self.task.action_space.high)

            # log_pi and entropy
            log_pi = dist.log_prob(actions).sum(-1).unsqueeze(-1)
            entropy = dist.entropy().sum(-1).unsqueeze(-1)

            # execute actions
            next_states, rewards, terminals, info = self.task.step(actions)

            # record data
            self.record_online_return(info)

            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy,
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'log_pi': log_pi,
                         'eps': epsilon})

            experiences = list(zip(self.states, options.unsqueeze(-1), actions, rewards, next_states, terminals))

            self.replay.feed_batch(experiences)
            if terminals[0]:
                self.random_process.reset_states()
            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += config.num_workers
            if self.total_steps // config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            # off-policy learning
            if self.replay.size() >= config.warm_up:

                # sample experience
                experiences = self.replay.sample()
                states, options, actions, rewards, next_states, terminals = experiences
                states = tensor(states)
                actions = tensor(actions)
                options = options.unsqueeze(-1)
                rewards = tensor(rewards).unsqueeze(-1)
                next_states = tensor(next_states)
                mask = tensor(1 - terminals).unsqueeze(-1)

                # calculation
                phi = self.target_network.feature(states)
                prediction = self.target_network(phi)
                phi_next = self.target_network.feature(next_states)
                prediction_next = self.target_network(phi_next)

                # target q value
                ret_q = rewards + \
                        config.discount * prediction['v_o'][self.worker_index, self.prev_options]
                q_loss = (prediction['q_o'].gather(1, options) - ret_q.detach()).pow(2).mul(0.5).mean()

                current_action, current_policy =
                ret_v = prediction['q_o'].gather(1, options) - current_policy
                v_loss = (prediction['v_o'].gather(1, options) - ret_v.detach()).pow(2).mul(0.5).mean()

                critic_loss = q_loss + v_loss

                # training critic process
                self.network.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()

                # training option process
                phi = self.network.feature(states)
                action, policy, log_pi = self.network.actor(phi)
                policy_loss = (- self.network.critic(phi.detach(), action) + log_pi).mean()

                q_prev_omg = prediction_next['q_o'].gather(1, next_options)
                v_prev_omg = torch.max(q_options, dim=1, keepdim=True)[0]
                advantage_omg = q_prev_omg - v_prev_omg
                advantage_omg.add_(config.termination_regularizer).detach()
                betas = betas.gather(1, prev_options)
                betas = betas * (1 - is_initial_betas)
                beta_loss = (betas * advantage_omg).mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)
