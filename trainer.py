import copy
from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc','l0','l1','l2' ))#'maploss'


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.3)
        #self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available else

    def ground_truth_gen(self, true_map, decode):
        t_map = copy.deepcopy(true_map)
        temp = decode.view(decode.shape[0] , t_map.shape[0] , t_map.shape[1] , t_map.shape[2])
        # actv = nn.Softmax(dim=2)
        # zero = torch.zeros(decode.shape[0], t_map.shape[0],t_map.shape[1]*t_map.shape[2])  #
        # one = torch.ones( decode.shape[0], t_map.shape[0],t_map.shape[1]*t_map.shape[2])  #
        # temp = decode.view(decode.shape[0], t_map.shape[0], t_map.shape[1]*t_map.shape[2]) # softmax use *
        for i in range(t_map.shape[1]):
            for j in range(t_map.shape[2]):
                if t_map[2,i,j] == 0:
                    t_map[1,i,j] = 0
                    temp[:, 1,i,j] = 0

        #temp = torch.where(temp>0, one, zero)
        '''
        #temp = actv(temp)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                #temp[i,j,:] = torch.where(temp[i,j,:] > temp[i,j,:].mean(), one, zero)
                temp[i, j, :] = torch.where(temp[i, j, :] > 0.5, one, zero)

        '''
        return t_map, temp


    def get_episode(self, epoch):
        ############### data collection for boss son ##################
        ### memo record dd = agent_map.detach().numpy() predator loc and prey loc
        ##################################################################
        tracj = np.ones([self.args.nagents,self.args.dim,self.args.dim])*-1
        comm_history = np.ones([self.args.nagents, self.args.dim, self.args.dim]) * -1
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        obs_layer = np.zeros([ self.env.env.true.shape[1], self.env.env.true.shape[2]])
        # np.save('./data/prey_loc_map'+str(episodes) + '.npy', self.env.env.prey_loc)
        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                #if np.any(info['comm_action'] == 1):
                    #checkpoint = x
                action_out, value, prev_hid, decoded= self.policy_net(x, info) #
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)

            agent_map = decoded  # .cpu().detach()


            # for i in range(obs_layer.shape[0]):
            obs_layer = obs_layer + self.env.env.true[ 2, :, :]
            obs_layer[obs_layer > 0] = 1
            self.env.env.true[2, :, :] = obs_layer
            gt, agent_map = self.ground_truth_gen(self.env.env.true, agent_map)
            gt = gt[np.newaxis]
            # temp = gt.flatten().reshape(1, int(
            #    self.env.env.true.shape[1] * self.env.env.true.shape[2] * self.env.env.true.shape[0]))
            #ground_truthg = torch.tensor(np.tile(gt, (self.args.nagents, 1)), requires_grad=True)
            ground_truthg = torch.tensor(np.repeat(gt, self.args.nagents, axis=0), requires_grad=True)
            ground_truthg = ground_truthg.view(ground_truthg.shape[0], ground_truthg.shape[1],ground_truthg.shape[2],ground_truthg.shape[3])
            agent_map = agent_map.view(agent_map.shape[0], agent_map.shape[1], agent_map.shape[2],
                                               agent_map.shape[3])
            Loss_func = nn.MSELoss(reduction='sum')
            #maploss = []
            #var0 = torch.ones(x[0].shape, requires_grad=True)
            #maploss = Loss_func(agent_map, ground_truthg.detach())

            l0 = Loss_func(agent_map[:, 0, :, :], ground_truthg[:, 0, :, :].detach())#/self.args.nagents
            l1 = Loss_func(agent_map[:, 1, :, :], ground_truthg[:, 1, :, :].detach())#/self.args.nagents
            l2 = Loss_func(agent_map[:, 2, :, :], ground_truthg[:, 2, :, :].detach())#/self.args.nagents
            '''
            ############### son dd
            dd = agent_map.detach().numpy()
            np.save('./data/decoder/decoded_map_'+str(episodes)+'_' + str(t) + '.npy', dd)
            np.save('./data/agent_loc/predator_loc_map'+str(episodes)+'_' + str(t) + '.npy', self.env.env.predator_loc)
            np.save('./data/ground_truth/ground_truth_map'+str(episodes)+'_' + str(t) + '.npy', gt[0])
            np.save('./data/comm/communication_map'+str(episodes)+'_' + str(t) + '.npy', action[-1])
            '''

            #record trajectory and comm history
            for i, p in enumerate(self.env.env.predator_loc):
                tracj[i, p[0], p[1]] = t
                if comm_history[i, p[0], p[1]] == -1:
                    comm_history[i, p[0], p[1]] = 0
                comm_history[i, p[0], p[1]] = comm_history[i, p[0], p[1]] + action[-1][i]

            next_state, reward, done, info = self.env.step(actual)
            next_state = next_state.squeeze().view(1, self.args.nagents, 3, self.args.dim, self.args.dim)
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()


            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, l0,l1,l2)
            #trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, maploss)

            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        ############# sons data


        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        # stat['min_steps'] = self.env.env.min_steps # pretrain vision
        # stat['min_steps'] = 0 # pretrain vision 2
        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)
        '''
        rewards = rewards.to(self.device)
        episode_masks = episode_masks.to(self.device)
        episode_mini_masks = episode_mini_masks.to(self.device)
        actions = actions.to(self.device)
        '''
        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        #alive_masks = alive_masks.to(self.device)
        coop_returns = torch.Tensor(batch_size, n)#.cuda()
        ncoop_returns = torch.Tensor(batch_size, n)#.cuda()
        returns = torch.Tensor(batch_size, n)#.cuda()
        deltas = torch.Tensor(batch_size, n)#.cuda()
        advantages = torch.Tensor(batch_size, n)#.cuda()
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        #map_loss = torch.stack(batch.maploss, dim=0)

        map_loss_0 = torch.stack(batch.l0, dim = 0)
        map_loss_1 = torch.stack(batch.l1, dim=0)
        map_loss_2 = torch.stack(batch.l2, dim=0)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks
        '''
        #map_loss_sum = map_loss.mean()
        map_loss_m0 = map_loss_0.mean()
        map_loss_m1 = map_loss_1.mean()
        map_loss_m2 = map_loss_2.mean()
        '''
        map_loss_m0 = map_loss_0.sum()
        map_loss_m1 = map_loss_1.sum()
        map_loss_m2 = map_loss_2.sum()

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()

        stat['map_loss'] = ((map_loss_m1 + map_loss_m0 + map_loss_m2)/n).item()

        stat['map_loss_0'] = map_loss_m0.item()
        stat['map_loss_1'] = map_loss_m1.item()
        stat['map_loss_2'] = map_loss_m2.item()
        map_loss = (map_loss_m1 + map_loss_m0 + map_loss_m2)/n
        loss = action_loss + self.args.value_coeff * (value_loss )+ self.args.value_coeff *map_loss


        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        stat['loss'] = loss.item()
        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size: # commended for data collection
        # while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
