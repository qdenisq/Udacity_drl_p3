import json
from pprint import pprint
from src.environment import ReacherEnvironment
from src.mappo import MAPPO
from src.models import SimpleMAPPOAgent
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt


def train(*args, **kwargs):
    print(kwargs)

    device = 'cpu'
    kwargs['mappo']['device'] = device

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim
    kwargs['agent']['num_agents'] = env.get_num_agents()

    agent = SimpleMAPPOAgent(**kwargs['agent']).to(device)
    alg = MAPPO(agent=agent, **kwargs['mappo'])
    scores = alg.train(env, 5000)

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = "../models/mappo_tennis_{}.pt".format(dt)
    torch.save(agent, model_fname)

    scores_fname = "../reports/mappo_tennis_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100)[:-100])
    fig_name = "../reports/mappo_tennis_{}.png".format(dt)
    plt.savefig(fig_name)


if __name__ == '__main__':
    with open('../config.json') as data_file:
        kwargs = json.load(data_file)
    pprint(kwargs)
    train(**kwargs)