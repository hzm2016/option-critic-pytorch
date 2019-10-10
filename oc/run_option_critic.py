from deep_rl import *


def set_tasks(config):
    if config.game == 'dm-walker':
        tasks = ['walk', 'run']
    elif config.game == 'dm-walker-1':
        tasks = ['squat', 'stand']
        config.game = 'dm-walker'
    elif config.game == 'dm-walker-2':
        tasks = ['walk', 'backward']
        config.game = 'dm-walker'
    elif config.game == 'dm-finger':
        tasks = ['turn_easy', 'turn_hard']
    elif config.game == 'dm-reacher':
        tasks = ['easy', 'hard']
    elif config.game == 'dm-cartpole-b':
        tasks = ['balance', 'balance_sparse']
        config.game = 'dm-cartpole'
    elif config.game == 'dm-cartpole-s':
        tasks = ['swingup', 'swingup_sparse']
        config.game = 'dm-cartpole'
    elif config.game == 'dm-fish':
        tasks = ['upright', 'downleft']
    elif config.game == 'dm-hopper':
        tasks = ['stand', 'hop']
    elif config.game == 'dm-acrobot':
        tasks = ['swingup', 'swingup_sparse']
    elif config.game == 'dm-manipulator':
        tasks = ['bring_ball', 'bring_peg']
    elif config.game == 'dm-cheetah':
        tasks = ['run', 'backward']
    else:
        raise NotImplementedError

    games = ['%s-%s' % (config.game, t) for t in tasks]
    config.tasks = [Task(g, num_envs=config.num_workers) for g in games]
    config.game = games[0]


# Option-Critic
def option_critic_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    run_steps(OptionCriticAgent(config))


def option_critic_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(OptionCriticAgent(config))


# Option-critic continuous
def oc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        phi_body=DummyBody(config.state_dim),
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )

    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.max_steps = 1e6
    config.rollout_length = 2048
    config.beta_reg = 0.01
    config.log_interval = 2048
    config.save_interval = 100
    config.algorithm_name = 'oc'
    run_steps(OCAgent(config))


def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 1e6
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim, num_options=8,
        actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh),
        option_body_fn=FCBody(config.state_dim)
                                                             )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 1e6
    config.beta_reg = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.algorithm_name = 'ppoc'
    run_steps(PPOCAgent(config))


# DOC
def doc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicOptionCriticNet(action_dim=config.action_dim, num_options=2,
                                                                phi_body=DummyBody(config.state_dim),
                                                                actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
                                                                critic_body=TwoLayerFCBodyWithAction(
                                                                config.state_dim, config.action_dim, (400, 300), gate=F.relu),
                                                                beta_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
                                                                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
                                                                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                                                             )

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3
    run_steps(OptionD3PGAgent(config))


# DDPG
def ddpg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3
    run_steps(DDPGAgent(config))


# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


if __name__ == '__main__':
    mkdir('./log/ppoc')
    mkdir('./tf_log/ppoc')
    mkdir('data')
    set_one_thread()
    random_seed()
    select_device(-1)
    env_list = ['RoboschoolHopper-v1', 'RoboschoolWalker2d-v1',
                'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1',
                'RoboschoolHumanoid-v1']

    # game = 'HalfCheetah-v2'
    # oc_continuous(game=game)
    # doc_continuous(game=game)
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    # td3_continuous(game=game)
    for i in range(len(env_list)):
        game = env_list[i]
        ppoc_continuous(game=game)

