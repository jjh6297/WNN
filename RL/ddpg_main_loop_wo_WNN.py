import gym
from ddpg_learn_Naive import DDPGagent
import tensorflow.keras.backend as K
def main():
    for trial in range(20,40):
        max_episode_num = 200
        env = gym.make("Pendulum-v1")
        agent = DDPGagent(env)
        agent.train(max_episode_num, trial=trial)
        # agent.plot_result()
    K.clear_session()


if __name__=="__main__":
    main()