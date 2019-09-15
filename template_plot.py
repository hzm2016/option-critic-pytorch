import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
from deep_rl import *


def plot_ppo():
    plotter = Plotter()
    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Hopper-v2',
        'Swimmer-v2',
        'Reacher-v2',
    ]

    patterns = [
        'remark_ppo',
    ]

    labels = [
        'PPO'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/ppo',
                       interpolation=100,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/PPO.png', bbox_inches='tight')


def plot_ddpg_td3():
    plotter = Plotter()
    games = [
        'RoboschoolWalker2d-v1'
    ]

    patterns = [
        '',
    ]

    labels = [
        'TD3',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='median',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./tf_log/logger-RoboschoolWalker2d-v1--run-0-190915-120750',
                       interpolation=0,
                       window=10,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/mujoco_eval.png', bbox_inches='tight')


def plot_atari():
    plotter = Plotter()
    games = [
        'BreakoutNoFrameskip-v4',
    ]

    patterns = [
        'remark_a2c',
        'remark_categorical',
        'remark_dqn',
        'remark_n_step_dqn',
        'remark_option_critic',
        'remark_ppo',
        'remark_quantile',
    ]

    labels = [
        'A2C',
        'C51',
        'DQN',
        'N-Step DQN',
        'OC',
        'PPO',
        'QR-DQN',
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='agg',
                       downsample=100,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TRAIN,
                       root='./data/benchmark/atari',
                       interpolation=0,
                       window=10,
                       )

    # plt.show()
    plt.tight_layout()
    plt.savefig('images/Breakout.png', bbox_inches='tight')


if __name__ == '__main__':
    mkdir('images')
    # plot_ppo()
    plot_ddpg_td3()
    # plot_atari()
