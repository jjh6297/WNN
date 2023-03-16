import gym
from ddpg_learn_WNN import DDPGagent
import tensorflow.keras.backend as K
def main():
    for trial in range(20):
        max_episode_num = 200
        env = gym.make("Pendulum-v1")
        agent = DDPGagent(env)
        agent.train(max_episode_num, trial=trial)
    K.clear_session()


if __name__=="__main__":
    main()