from unityagents import UnityEnvironment

class ReacherEnvironment:
    def __init__(self, env_file=None, seed=1234, **kwargs):
        self.__env = UnityEnvironment(file_name=env_file, seed=seed)  # create environment
        self.__brain_name = self.__env.brain_names[0]
        env_info = self.__env.reset()[self.__brain_name]
        print(env_info)
        self.__num_agents = env_info.vector_observations.shape[0]
        self.__state_dim = env_info.vector_observations.shape[1]
        self.__action_dim = self.__env.brains[self.__brain_name].vector_action_space_size

    def step(self, action):
        env_info = self.__env.step(action)[self.__brain_name]  # step
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished
        return next_state, reward, done

    def reset(self, train_mode=True):
        env_info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        state = env_info.vector_observations
        return state

    def get_num_agents(self):
        return self.__num_agents

    def get_state_dim(self):
        return self.__state_dim

    def get_action_dim(self):
        return self.__action_dim

    def get_episode_len(self):
        return 1001

    def close(self):
        self.__env.close()



class TennisEnvironment:
    def __init__(self, env_file=None, seed=1234, **kwargs):
        self.__env = UnityEnvironment(file_name=env_file, seed=seed)  # create environment
        self.__brain_name = self.__env.brain_names[0]
        env_info = self.__env.reset()[self.__brain_name]
        print(env_info)
        self.__num_agents = env_info.vector_observations.shape[0]
        self.__state_dim = env_info.vector_observations.shape[1]
        self.__action_dim = self.__env.brains[self.__brain_name].vector_action_space_size

    def step(self, action):
        env_info = self.__env.step(action)[self.__brain_name]  # step
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished
        return next_state, reward, done

    def reset(self, train_mode=True):
        env_info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        state = env_info.vector_observations
        return state

    def get_num_agents(self):
        return self.__num_agents

    def get_state_dim(self):
        return self.__state_dim

    def get_action_dim(self):
        return self.__action_dim

    def get_episode_len(self):
        return 1001

    def close(self):
        self.__env.close()