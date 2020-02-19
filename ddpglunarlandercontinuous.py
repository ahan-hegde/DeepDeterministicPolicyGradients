import gym
import numpy as np

def train_lunarlander_field(env, network_field):
    np.random.seed(0)

    num_episodes = 1000
    save_episodes = 25
    done = False
    score_hist = []
    score = 0

    network_field.load_field()

    for e in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = network_field.choose_action(state)
            state_, reward, done, _ = env.step(action)
            network_field.store_experience(state, action, reward, state_, done)
            network_field.learn()
            score += reward
            state = state_

        if e % save_episodes == 0:
            network_field.save_field()

        score_hist.append(score)
        running_avg = np.mean(score_hist[-100:])
        print('episode: '+str(e)+' score: '+str(score)+
              ' running avg (last 100): '+str(running_avg))

env = gym.make('LunarLanderContinuous-v2')

network_field = NetworkField(env=env, input_dims=[8],
                             num_actions=env.action_space.n,
                             batch_size=64,
                             alpha=0.000025,
                             beta=0.00025)

train_lunarlander_field(env=env, network_field=network_field)
