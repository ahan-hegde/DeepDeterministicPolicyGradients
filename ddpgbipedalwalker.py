def train_bipedalwalker_field(env, network_field):
    np.random.seed(0)

    num_episodes = 5000
    save_episodes = 25
    done = False
    score_hist = []
    score = 0
    frame = 0
    str_episode = 0
    eppath = 'drive/My Drive/ddpg/episodesv1.txt'
    if os.path.exists(eppath):
        epfile = open(eppath, 'r')
        str_episode = epfile.read()

    for e in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        frame = 0
        while not done:
            action = network_field.choose_action(state)
            state_, reward, done, _ = env.step(action)
            network_field.store_experience(state, action, reward, state_, done)
            network_field.learn()
            score += reward
            state = state_
            frame += 1

        if e % save_episodes == 0:
            network_field.save_field()
            epfile = open('drive/My Drive/ddpg/episodesv1.txt', 'w')
            epfile.write(str(e+int(str_episode)))

        score_hist.append(score)
        running_avg = np.mean(score_hist[-100:])
        print('episode: ', e+int(str_episode),
              ' score: %.2f' % score,
              ' running avg (last 100): %.3f' % running_avg,
              ' num frames: %d' % frame)


def test_bipedalwalker_field(env, network_field):

    num_episodes = 100
    done = False
    score_hist = []
    score = 0

    for e in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        frame = 0

        while not done:
            action = network_field.make_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            state = state_
            frame += 1

        score_hist.append(score)
        running_avg = np.mean(score_hist)
        print('episode: ', e,
              ' score: %.2f' % score,
              ' running avg (last 100): %.3f' % running_avg,
              ' num frames: %d' % frame)
