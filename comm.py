import math

import copy

import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.mapsize = args.dim*args.dim*3
        self.continuous = args.continuous
        #self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available else
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
            '''
            self.mapdecode = nn.Sequential(
                nn.Linear(self.args.hid_size, self.args.hid_size),
                #nn.ReLU(),
                nn.Linear(self.args.hid_size, self.args.hid_size),
                #nn.ReLU(),
                nn.Linear(self.args.hid_size, self.mapsize)
                #nn.ReLU()
            )  #nn.Tanh())
            '''
            self.mapdecode = nn.Sequential(
                nn.Linear(args.hid_size, (self.args.dim-2*3+2)*(self.args.dim-2*3+2)*4),
                nn.Unflatten(1, torch.Size([4, (self.args.dim-2*3+2), (self.args.dim-2*3+2)])),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(4, 16, 3),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(16, 3, 3),
            )

        '''
        self.W_Q = nn.Linear(args.hid_size, 32 * args.hid_size )
        self.W_K = nn.Linear(args.hid_size, 32 * args.hid_size )
        self.W_V = nn.Linear(args.hid_size, 32 * args.hid_size)
        self.W_O = nn.Linear(32 * args.hid_size, args.hid_size, bias=False)
        '''
        # my multi-head attention model
        # --- tarmac model ----
        self.state2query = nn.Linear(args.hid_size, 16)
        self.state2key = nn.Linear(args.hid_size, 16)
        self.state2value = nn.Linear(args.hid_size, 1*args.hid_size)
        self.W_O = nn.Linear(1 * args.hid_size, args.hid_size, bias=False)

        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        '''
        self.encoder = nn.Sequential(
                nn.Linear(num_inputs, args.hid_size),
                nn.Linear(args.hid_size, args.hid_size))
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3,3)),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 4, (3,3)),
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear((self.args.dim-2*3+2)*(self.args.dim-2*3+2)*4, args.hid_size)
        )

        # self.encoder.load_state_dict(torch.load("/home/qhuang18/IC3Net_env/pycharm_proj/pretrained_model/model_dim7_pt"))
        # self.encoder.eval()
        '''
        self.decoder = nn.Linear(args.hid_size, self.mapsize)

        self.autodecoder0 = nn.Linear(args.hid_size, args.hid_size)
        self.autodecoder1 = nn.Linear(args.hid_size, 256)
        self.autodecoder2 = nn.Linear(256, num_inputs)

        self.autodecoder = nn.Sequential(
            nn.Linear(args.hid_size, 3*3*4),
            nn.Unflatten(1, torch.Size([4, 3, 3])),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3),
            nn.Sigmoid()
        )
        '''
        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(self.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(self.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(self.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.args.hid_size, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n
        #agent_mask = agent_mask.to(self.device)
        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            #x = x.to(self.device)
            x = torch.squeeze(x)
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)
        batch_size = x.size()[0]
        batch_size = 1
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            #comm_action = torch.ones(self.nagents)
            #comm_action = comm_action.to(self.device)
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            # agent_mask *= comm_action_mask.double()
            agent_mask = agent_mask * comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.args.hid_size) if self.args.recurrent else hidden_state

            masked_msg = torch.zeros(comm.shape) * -1e9
            for idx, agt in enumerate(comm_action):
                if agt == 1:
                    masked_msg[0, idx, :] = comm[0, idx, :]
            #comm = comm.to(self.device)
            '''
            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.args.hid_size)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)
            mask = mask.expand_as(comm)
            comm = comm * mask
            ''''''
            # --- our attention starts here ---
            # attn_inp = comm.squeeze()
            # for j in range(self.nagents):
            #     attn_inp[j,j,:] = torch.ones(self.args.hid_size)*-1
            attn_inp = comm.squeeze()
            num_agents, input_dim = attn_inp.size()
            # attn_inp = attn_inp.view(n, n*self.args.hid_size)
            assert input_dim == self.args.hid_size
            # prev_hid = hidden_state.unsqueeze(-2).expand(n, n, self.args.hid_size)
            # prev_hid = prev_hid.view(n, n* self.args.hid_size)
            # prev_hid = hidden_state.repeat((1,n))

            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            q_s = self.W_Q(attn_inp).view(32, self.nagents, -1)#.transpose(1, 2)  # q_s: [batch_size x num_heads x num_agents x output_dim]
            k_s = self.W_K(attn_inp).view(32,self.nagents, -1)#.transpose(1, 2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
            v_s = self.W_V(attn_inp).view(32,self.nagents, -1)#.transpose(1, 2)
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (
                self.args.hid_size ** 0.5)  # scores : [batch_size x n_heads x num_agents x num_agents]
            attn = F.softmax(scores, dim=-1).double()
            context = torch.matmul(attn, v_s)
            context = context.transpose(1, 2).contiguous().view(self.nagents,32*self.args.hid_size)  # context: [batch_size x len_q x n_heads * d_v]
            attn_c = self.W_O(context)
            '''
            # --- our attention ends here ---
            '''
            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            '''
            query = self.state2query(comm).view(1, self.nagents, -1)
            key = self.state2key(masked_msg).view(1, self.nagents, -1)
            value = self.state2value(masked_msg).view(1, self.nagents, -1)

            # scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hid_size)
            #scores = scores.masked_fill(comm_action_mask.squeeze(-1) == 0, -1e9)
            # softmax + weighted sum

            attn = F.softmax(scores, dim=-1)
            comm = torch.matmul(attn, value)

            context = comm.transpose(0, 1).contiguous().view(self.nagents,
                                                             1 * self.args.hid_size)  # context: [batch_size x len_q x n_heads * d_v]
            # comm = self.W_O(context)

            c = self.C_modules[i](context).squeeze()

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c
                #c = c.view(1, 1, n, self.args.hid_size)
                # inp = torch.cat((x, c), 1)

                #inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

                #hid = hidden_state.view(batch_size * n, self.args.hid_size)

                decoded = self.mapdecode(hidden_state)  # for map decoder
                # for auto encoder
                #auto_res = self.autodecoder(x)

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.args.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone()), decoded #, auto_res
        else:
            return action, value_head, decoded

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.args.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.args.hid_size, requires_grad=True)))


