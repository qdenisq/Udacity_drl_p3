import numpy as np
import torch
from src.replay_buffer import ReplayBuffer


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, seed=0):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu
        self.seed = np.random.seed(seed)

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class DDPG:
    def __init__(self, *args, agent=None, target_agent=None, **kwargs):
        self.agent = agent
        self.target_agent = target_agent
        # hard update
        self.hard_update(self.target_agent, self.agent)
        self.replay_buffer = ReplayBuffer(buffer_size=int(kwargs['buffer_size']), minibatch_size=kwargs['minibatch_size'],
                                          seed=kwargs['seed'], device=kwargs['device'])

        self.__minibatch = kwargs['minibatch_size']

        self.actor_optim = torch.optim.Adam(self.agent.get_actor_parameters(), lr=kwargs['actor_lr'])
        self.critic_optim = torch.optim.Adam(self.agent.get_critic_parameters(), lr=kwargs['critic_lr'])

        self.__discount = kwargs['discount']
        self.__tau = kwargs['tau']
        return

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """
        noise_gen = OrnsteinUhlenbeckActionNoise(env.get_action_dim())
        noise_gen.reset()
        mean_score = []
        scores = []
        self.target_agent.eval()
        for episode in range(num_episodes):
            state = env.reset(train_mode=True)
            # roll out
            j = 0
            score = np.zeros(env.get_num_agents())
            self.agent.eval()
            while True:
                # step
                inp_state = np.concatenate([state, np.flip(state, axis=0)], axis=1)

                action = self.agent.act(torch.Tensor(inp_state)).detach().cpu().numpy()
                noise = [noise_gen.sample() for _ in range(env.get_num_agents())]
                noised_action = action + noise
                noised_action = np.clip(noised_action, -1., 1.)
                next_state, reward, done = env.step(noised_action.squeeze())

                score += reward

                # add experience to replay buffer
                for i in range(action.shape[0]):
                    self.replay_buffer.add(np.concatenate([state, np.flip(state, axis=0)], axis=1)[i], action[i], reward[i],
                                           np.concatenate([next_state, np.flip(next_state, axis=0)], axis=1)[i], done[i])

                state = next_state

                if self.replay_buffer.size() < self.__minibatch:
                    continue

                self.agent.train()
                # sample minibatch
                states, actions, rewards, next_states, dones = self.replay_buffer.sample()
                # compute critic loss
                target_actions = self.target_agent.act(next_states)
                target_Q = rewards + self.__discount * self.target_agent.Q(next_states, target_actions) * (1 - dones)
                Q = self.agent.Q(states, actions)
                critic_loss = (Q - target_Q).pow(2).mean()
                # update critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # compute actor objective
                actor_actions = self.agent.act(states)
                Q = self.agent.Q(states, actor_actions)
                actor_objective = -Q.mean()
                # update actor
                self.actor_optim.zero_grad()
                actor_objective.backward()
                self.actor_optim.step()

                # soft update of target agent
                self.soft_update(self.target_agent, self.agent, self.__tau)

                if np.any(done):
                    break

            print("episode: {:d} | score: {}".format(episode, score))
            scores.append(score)
        return scores
