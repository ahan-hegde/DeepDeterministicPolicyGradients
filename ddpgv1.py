import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float


def matrix_to_tensor(matrix, device=device, dtype=dtype):
    tensor = torch.tensor(matrix, dtype=dtype).to(device)
    return tensor


class OrnstienUhlenbeckNoise(object):
    def __init__(self, mu, dt=1e-2, theta=0.15, sigma=0.2, x0=None):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.xprev + self.theta * (self.mu - self.xprev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.xprev  = x

        return x

    def reset(self):
        self.xprev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, buffer_size, batch_size, input_dims, num_actions):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_count = 0
        self.states_memory = np.zeros((buffer_size, *input_dims))
        self.nextstates_memory = np.zeros((buffer_size, *input_dims))
        self.actions_memory = np.zeros((buffer_size, num_actions))
        self.rewards_memory = np.zeros((buffer_size))
        self.dones_memory = np.zeros((buffer_size))

    def store_transition(self, state, action, reward, state_, done):
        idx = self.buffer_count % self.buffer_size

        self.states_memory[idx] = state
        self.actions_memory[idx] = action
        self.rewards_memory[idx] = reward
        self.nextstates_memory[idx] = state_
        self.dones_memory[idx] = 1 - int(done)

        self.buffer_count += 1

    def sample_memory(self):
        max_mem = min(self.buffer_count, self.buffer_size)
        idxs = np.random.choice(max_mem, self.batch_size)

        states = self.states_memory[idxs]
        actions = self.actions_memory[idxs]
        rewards = self.rewards_memory[idxs]
        nextstates = self.nextstates_memory[idxs]
        dones = self.dones_memory[idxs]

        return states, actions, rewards, nextstates, dones


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, num_actions, beta, name, batch_size=128,
                 chkpt_dir='drive/My Drive/ddpg/'):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.model_file = self.chkpt_dir+name+'_bwddpg'

        self.num_l1 = 400
        self.num_l2 = 300

        self.fc1 = nn.Linear(*self.input_dims, self.num_l1)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.num_l1)

        self.fc2_val1 = nn.Linear(self.num_l1, self.num_l2)
        f2 = 1. / np.sqrt(self.fc2_val1.weight.data.size()[0])
        nn.init.uniform_(self.fc2_val1.weight.data, -f2, f2)
        nn.init.uniform_(self.fc1.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.num_l2)

        self.fc2_val2 = nn.Linear(self.num_actions, self.num_l2)

        self.qval = nn.Linear(self.num_l2, 1)
        f3 = 0.003
        nn.init.uniform_(self.qval.weight.data, -f3, f3)
        nn.init.uniform_(self.qval.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.to(device)

    def forward(self, state, action):
        layer = self.fc1(state)
        layer = self.bn1(layer)
        layer = F.relu(layer)

        layer = self.fc2_val1(layer)
        value1 = self.bn2(layer)
        value2 = self.fc2_val2(action)

        sumval = F.relu(torch.add(value1, value2))
        qval = self.qval(sumval)

        return qval

    def save_model(self):
        print('... saving model ...')
        torch.save(self.state_dict(), self.model_file)

    def load_model(self):
        print('... loading model ...')
        print(self.model_file)
        if os.path.exists(self.model_file):
            self.load_state_dict(torch.load(self.model_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, num_actions, alpha, action_bound, name,
                 batch_size=128, chkpt_dir='drive/My Drive/ddpg/'):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.model_file = self.chkpt_dir+name+'_bwddpg'
        self.action_bound = action_bound

        self.num_l1 = 400
        self.num_l2 = 300

        self.fc1 = nn.Linear(*self.input_dims, self.num_l1)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.num_l1)

        self.fc2 = nn.Linear(self.num_l1, self.num_l2)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc1.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.num_l2)

        self.mu = nn.Linear(self.num_l2, self.num_actions)
        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.to(device)

    def forward(self, state):
        layer = self.fc1(state)
        layer = self.bn1(layer)
        layer = F.relu(layer)
        layer = self.fc2(layer)
        layer = self.bn2(layer)
        layer = F.relu(layer)
        layer = self.mu(layer)
        mu = torch.tanh(layer)

        return mu

    def save_model(self):
        print('... saving model ...')
        torch.save(self.state_dict(), self.model_file)

    def load_model(self):
        print('... loading model ...')
        print(self.model_file)
        if os.path.exists(self.model_file):
            self.load_state_dict(torch.load(self.model_file))


class NetworkField(object):
    def __init__(self, env, input_dims, num_actions, batch_size=64,
                 alpha=0.00005, beta=0.0005, gamma=0.99, tau=0.001):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size

        self.critic_eval = CriticNetwork(input_dims, num_actions, beta,
                           'critic_eval')
        self.actor_eval = ActorNetwork(input_dims, num_actions, alpha,
                          env.action_space.high, 'actor_eval')

        self.critic_target = CriticNetwork(input_dims, num_actions, beta,
                            'critic_target')
        self.actor_target = ActorNetwork(input_dims, num_actions, alpha,
                          env.action_space.high, 'actor_target')

        self.noise = OrnstienUhlenbeckNoise(np.zeros(num_actions))
        self.replay_buffer = ReplayBuffer(1000000, batch_size, input_dims,
                                        num_actions)

        self.critic_target.load_state_dict(
                                dict(self.critic_eval.named_parameters()))
        self.actor_target.load_state_dict(
                                dict(self.actor_eval.named_parameters()))

    def choose_action(self, state):
        state = matrix_to_tensor(state[np.newaxis, :])
        self.actor_eval.eval()
        mu = self.actor_eval.forward(state)
        mu = mu + matrix_to_tensor(self.noise())
        mu *= matrix_to_tensor(self.actor_eval.action_bound)
        self.actor_eval.train()

        return mu.cpu().detach().numpy()[0]

    def make_action(self, state):
        state = matrix_to_tensor(state[np.newaxis, :])
        self.actor_eval.eval()
        mu = self.actor_eval.forward(state)
        mu *= matrix_to_tensor(self.actor_eval.action_bound)

        return mu.cpu().detach().numpy()[0]

    def store_experience(self, state, action, reward, state_, done):
        self.replay_buffer.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.replay_buffer.buffer_count < self.batch_size:
            return False

        states, actions, rewards, nextstates, dones = \
                                self.replay_buffer.sample_memory()

        states = matrix_to_tensor(states)
        actions = matrix_to_tensor(actions)
        nextstates = matrix_to_tensor(nextstates)
        rewards = matrix_to_tensor(rewards)
        dones = matrix_to_tensor(dones)

        self.critic_target.eval()
        self.actor_target.eval()
        self.critic_eval.eval()

        nextactions = self.actor_target.forward(nextstates)
        nextqvals = self.critic_target.forward(nextstates, nextactions)
        qvals = self.critic_eval.forward(states, actions)

        yvals = []

        for i in range(self.batch_size):
            yvals.append(rewards[i] + self.gamma*nextqvals[i]*dones[i])

        yvals = matrix_to_tensor(yvals)
        yvals = yvals.view(self.batch_size, 1)

        #critic_loss: where we optimize critic_eval by calculating yvals
        self.critic_eval.train()
        self.critic_eval.optimizer.zero_grad()
        critic_loss = F.mse_loss(yvals, qvals)
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        #actor_loss: where we optimize actor_eval by calculating mus
        self.critic_eval.eval()
        self.actor_eval.optimizer.zero_grad()
        mus = self.actor_eval.forward(states)
        self.actor_eval.train()
        actor_loss = torch.mean(-self.critic_eval(states, mus))
        actor_loss.backward()
        self.actor_eval.optimizer.step()

        self.soft_target_update()

    def soft_target_update(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_target_weights = dict(self.critic_target.named_parameters())
        actor_target_weights = dict(self.actor_target.named_parameters())
        critic_eval_weights = dict(self.critic_eval.named_parameters())
        actor_eval_weights = dict(self.actor_eval.named_parameters())

        for name in critic_eval_weights:
            critic_target_weights[name] = tau*critic_eval_weights[name].clone() + \
            (1-tau)*critic_target_weights[name].clone()

        self.critic_target.load_state_dict(critic_target_weights)

        for name in actor_eval_weights:
            actor_target_weights[name] = tau*actor_eval_weights[name].clone() + \
            (1-tau)*actor_target_weights[name].clone()

        self.actor_target.load_state_dict(actor_target_weights)

    def save_field(self):
        self.critic_target.save_model()
        self.critic_eval.save_model()
        self.actor_target.save_model()
        self.actor_eval.save_model()

    def load_field(self):
        self.critic_target.load_model()
        self.critic_eval.load_model()
        self.actor_target.load_model()
        self.actor_eval.load_model()
