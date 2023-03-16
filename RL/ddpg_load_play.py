# DDPG load and play (tf2 subclassing API version)
# coded by St.Watermelon

import gym
from ddpg_learn import DDPGagent
import tensorflow as tf

def main():

    env = gym.make("Pendulum-v0")
    agent = DDPGagent(env)

    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
        print(action.shape)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()