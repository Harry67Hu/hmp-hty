import torch, math, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, my_view, repeat_at
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ALGORITHM.common.attention import SimpleAttention
from ALGORITHM.common.norm import DynamicNormFix
from ALGORITHM.common.net_manifest import weights_init
from config import GlobalConfig


class E_GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=1):
        super(E_GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        assert n_heads == 1, '目前还没有涉及多头的形式！'

        
        # 不采用多头的形式
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.a = nn.Linear(hidden_dim*2, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.out_attn = nn.Linear(hidden_dim, output_dim)


        # 多头的遗弃版本
        # self.W = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        # self.a = nn.Parameter(torch.Tensor(n_heads, hidden_dim * 2))
        # self.attn_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_heads)])
        # self.out_attn = nn.Linear(n_heads * hidden_dim, output_dim)

        self.init_parameters()


    def init_parameters(self):
        param = self.W
        # for param in self.parameters():
        stdv = 1. / math.sqrt(param.size(-1))
        param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, message, mask):
        # h维度为[input_dim]   
        # message维度为[n_agent, input_dim]   
        # mask维度为[n_agent]    e.g: mask = torch.randint(0,2, (n_agent,))  
        # MASK  mask 只保留距离内的 + 同类的，排除掉自己
        mask = torch.tensor(mask).to(GlobalConfig.device)
        # OBS [n_entity, input_dim] 用作知识直接提取信息

        # n_agent = message.shape[0]
        n_agent = AlgorithmConfig.n_agent

        # 自身信息
        h = torch.matmul(h, self.W)               #  (hidden_dim）
        h_repeat = repeat_at(h, -2, n_agent)        #  (n_agent, hidden_dim）

        # 接收到的观测信息（理论上应该是mask掉的，但是此处没有）
        H = torch.matmul(message, self.W)          # （n_agent, hidden_dim）
        
        # 求权重(记得最后还得mask一遍)
        H_cat = torch.cat((h_repeat, H), dim=-1)   # （n_agent, hidden_dim * 2）
        E = self.a(H_cat)                               # （n_agent, 1）
        E = self.act(E)
        E_mask = E * mask.unsqueeze(-1) 
        alpha = F.softmax(E_mask, dim=-2)                    # （n_agent, 1）
        alpha_mask = alpha * mask.unsqueeze(-1)         # （n_agent, 1）
        # 为了在这里解决 0*nan = nan 的问题，输入必须将nan转化为0
        alpha_mask = torch.nan_to_num(alpha_mask, 0)

        weighted_sum = torch.mul(alpha_mask, H).sum(dim=-2)  #  (hidden_dim）
        H_E = self.out_attn(h + weighted_sum)
        H_E = F.elu(H_E)
        assert not torch.isnan(H_E).any(), ('nan problem!')

        return H_E




class Net(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        super().__init__()
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.policy_resonance
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        self.skip_connect = True
        self.n_action = n_action

        # observation pre-process part
        self.rawob_dim = rawob_dim
        self.use_obs_pro_uhmp = AlgorithmConfig.use_obs_pro_uhmp
        obs_process_h_dim = AlgorithmConfig.obs_process_h_dim

        # observation and advice message part
        act_dim = AlgorithmConfig.act_dim
        obs_h_dim = AlgorithmConfig.obs_h_dim
        adv_h_dim = AlgorithmConfig.adv_h_dim
        obs_abs_h_dim = AlgorithmConfig.obs_abs_h_dim
        act_abs_h_dim = AlgorithmConfig.act_abs_h_dim
        rnn_h_dim = AlgorithmConfig.rnn_h_dim

        # PGAT net part
        GAT_h_dim = AlgorithmConfig.GAT_h_dim
        H_E_dim = AlgorithmConfig.H_E_dim
        H_I_dim = AlgorithmConfig.H_I_dim
        h_dim = AlgorithmConfig.obs_h_dim   # TODO 

        

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        # observation pre-process (if needed)
        if self.use_obs_pro_uhmp:
            self.state_encoder = nn.Sequential(nn.Linear(rawob_dim, obs_process_h_dim), nn.ReLU(inplace=True), nn.Linear(obs_process_h_dim, obs_process_h_dim))
            self.entity_encoder = nn.Sequential(nn.Linear(rawob_dim * (self.n_entity_placeholder-1), obs_process_h_dim), nn.ReLU(inplace=True), nn.Linear(obs_process_h_dim, obs_process_h_dim))
    
            self.AT_obs_encoder = nn.Sequential(nn.Linear(obs_process_h_dim + obs_process_h_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, obs_h_dim))
            self.AT_obs_abstractor = nn.Sequential(nn.Linear(obs_process_h_dim + obs_process_h_dim, obs_abs_h_dim), nn.ReLU(inplace=True), nn.Linear(obs_abs_h_dim, obs_abs_h_dim))
            self.AT_act_abstractor = nn.Sequential(nn.Linear(act_dim, act_abs_h_dim), nn.ReLU(inplace=True), nn.Linear(act_abs_h_dim, act_abs_h_dim))

        else: 
            self.AT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim  * self.n_entity_placeholder, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, obs_h_dim))
            self.AT_obs_abstractor = nn.Sequential(nn.Linear(rawob_dim  * self.n_entity_placeholder, obs_abs_h_dim), nn.ReLU(inplace=True), nn.Linear(obs_abs_h_dim, obs_abs_h_dim))
            self.AT_act_abstractor = nn.Sequential(nn.Linear(act_dim, act_abs_h_dim), nn.ReLU(inplace=True), nn.Linear(act_abs_h_dim, act_abs_h_dim))

        # actor network construction ***
        # 1st flow
            # self.AT_obs_encoder
        self.AT_E_Het_GAT = E_GAT(input_dim=obs_h_dim, hidden_dim=GAT_h_dim, output_dim=H_E_dim)
        # 2nd flow
            # self.AT_obs_abstractor
            # self.AT_act_abstractor
        self.gru_cell_memory = None
        self.fc1_rnn = nn.Linear(obs_abs_h_dim + act_abs_h_dim, rnn_h_dim)
        self.gru = nn.GRUCell(rnn_h_dim, rnn_h_dim)
        self.fc2_rnn = nn.Linear(rnn_h_dim, adv_h_dim)
        self.AT_I_Het_GAT = E_GAT(input_dim=adv_h_dim, hidden_dim=GAT_h_dim, output_dim=H_I_dim)
        # together
        # self.AT_PGAT_mlp = nn.Sequential(nn.Linear(H_E_dim + H_I_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, obs_h_dim))  # 此处默认h_dim是一致的
        self.AT_PGAT_mlp = nn.Sequential(nn.Linear(adv_h_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, adv_h_dim))  # 此处默认h_dim是一致的
        
        self.AT_policy_head = nn.Sequential(
            nn.Linear(obs_h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            nn.Linear(h_dim//2, self.n_action))

        # critic network construction ***
        self.CT_get_value = nn.Sequential(nn.Linear(obs_h_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))
        # self.CT_get_threat = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))

        self.is_recurrent = False
        self.apply(weights_init)

        # 知识部分的参数
        self.n_agent = AlgorithmConfig.n_agent

        return
    
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        act = self._act
        return act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        act = self._act
        return act(*args, **kargs, eval_mode=True)
    
    def _act(self, obs=None, act=None, message_obs=None, message_adv=None, type_mask=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        # if not test_mode: assert not self.forbidden
        eval_act = eval_actions if eval_mode else None
        others = {}
        assert self.n_entity_placeholder == obs.shape[-2], 'observation structure wrong!'

        # 数据处理
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs, freeze=(eval_mode or test_mode))
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents

        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0  obs [n_threads, n_agents, n_entity, rawob_dim]
        assert type_mask is not None, 'type_mask wrong'
        # E_Het_mask  = torch.ones(obs.shape[0], obs.shape[1], self.n_agent)  # 不使用复杂计算的消融实验
        # E_Het_mask = self.get_E_Het_mask(obs=obs, type_mask=type_mask, dead_mask=mask_dead)     # [n_threads, n_agents, n_agent] # warning n_agents是共享网络的智能体数据，n_agent是全局智能体数目
        # I_Het_mask = self.get_I_Het_mask(obs=obs, type_mask=type_mask, dead_mask=mask_dead)

        # Obs预处理部分
        if self.use_obs_pro_uhmp:
            s, other = self.div_entity(obs,       type=[(0,), range(1, self.n_entity_placeholder)], n=self.n_entity_placeholder)
            s = s.squeeze(-2)                                               # [n_threads, n_agents, rawob_dim]
            other = other.reshape(other[0], other[1], other[-2]*other[-1])  # [n_threads, n_agents, n_entity-1 * rawob_dim]
            print(other.size)
            zs = self.state_encoder(s)          # [n_threads, n_agents, obs_process_h_dim]
            zo = self.entity_encoder(other)     # [n_threads, n_agents, obs_process_h_dim]
            obs = torch.cat((zs, zo), -1)     # [n_threads, n_agents, obs_process_h_dim * 2]
        else:
            obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[-2]*obs.shape[-1])  # [n_threads, n_agents, n_entity * rawob_dim]

        # # 环境观测理解部分
        # h_obs = self.AT_obs_encoder(obs)
        
        # 环境策略建议部分
        abstract_obs = self.AT_obs_abstractor(obs)
        abstract_act = self.AT_act_abstractor(act)
        abstract_cat = torch.cat((abstract_obs, abstract_act), -1)
        gru_input = F.relu(self.fc1_rnn(abstract_cat))

            # GRU此处处理的时候先碾平再恢复原有形状
        gru_input_expand = gru_input.view(gru_input.shape[0]*gru_input.shape[1], gru_input.shape[2])
        self.gru_cell_memory = self.gru(gru_input_expand)
        self.gru_cell_memory = self.gru_cell_memory.view(gru_input.shape)
        h_adv = self.fc2_rnn(self.gru_cell_memory)

        # PGAT部分
        # H_E = self.AT_E_Het_GAT(h_obs, message_obs, E_Het_mask)
        # H_I = self.AT_I_Het_GAT(h_adv, message_adv, I_Het_mask)
        # H_sum = self.AT_PGAT_mlp(torch.cat((H_E, H_I), -1))
        H_sum = self.AT_PGAT_mlp(h_adv)
    
        # 策略网络部分
        logits = self.AT_policy_head(H_sum)

        # Critic网络部分
        value = self.CT_get_value(H_sum)
            
            
        logit2act = self._logit2act
        if self.use_policy_resonance and self.is_resonance_active():
            logit2act = self._logit2act_rsn
            
        act, actLogProbs, distEntropy, probs = logit2act(   logits, eval_mode=eval_mode,
                                                            test_mode=test_mode, 
                                                            eval_actions=eval_act, 
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)

        message_obs_output = h_adv
        message_adv_output = h_adv
        # message_adv_output = h_obs


        if not eval_mode: return act, value, actLogProbs, message_obs_output, message_adv_output
        else:             return value, actLogProbs, distEntropy, probs, others, message_obs_output, message_adv_output


    def _logit2act_rsn(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, eprsn=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)
        
        if not test_mode: act = self.ccategorical.sample(act_dist, eprsn) if not eval_mode else eval_actions
        else:             act = torch.argmax(act_dist.probs, axis=2)
        # the policy gradient loss will feedback from here
        actLogProbs = self._get_act_log_probs(act_dist, act) 
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, **kwargs):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:  act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
    
    def div_entity(self, mat, type=[(0,), (1, 2, 3, 4, 5),(6, 7, 8, 9, 10, 11)], n=12):
        if mat.shape[-2]==n:
            tmp = (mat[..., t, :] for t in type)
        elif mat.shape[-1]==n:
            tmp = (mat[..., t] for t in type)

        return tmp
    

    def get_E_Het_mask(self, obs, type_mask, dead_mask):
        """
        利用知识的方式直接读取信息   知识构图准则：
        1. 移除dead agent     (obs内的信息已经满足条件)
        2. 移除非同队伍agent
        3. 移除非同类型agent  (type_mask已经满足条件)
        4. 通信距离内最近的几个 (obs内的信息已经满足条件)
        5. 排除自己
        """
        return self.get_E_Het_mask_uhmp(obs=obs, type_mask=type_mask, dead_mask=dead_mask)
    
    def get_E_Het_mask_uhmp(self, obs, type_mask, dead_mask):
        # uhmp部分对obs进行了切片[self=0, allay=1~int(n_entity_placeholder/2), enemy=int(n_entity_placeholder/2)~-2, wall=-1]
        type_mask = type_mask.cpu().numpy() if type_mask is not None else type_mask  # type_mask [n_agent, n_agent]
        dead_mask = dead_mask.cpu().numpy() if dead_mask is not None else dead_mask  # dead_mask [n_threads, n_agents, n_entity]
        obs = obs.cpu().numpy() if 'cuda' in GlobalConfig.device else obs.numpy()    # obs[n_threads, n_agents, n_entity, state_dim]
        assert obs.shape[-1] == self.rawob_dim, '错误的观测信息，应该为没有经过预处理的信息！'
        # 由于此处只考虑队友通信，故将enemy和wall的信息放到了不用的Other里面
        zs, ally, other = self.div_entity(obs, type=[(0,), range(1, int(self.n_entity_placeholder/2)), range(int(self.n_entity_placeholder/2), self.n_entity_placeholder)], n=self.n_entity_placeholder)
        # zs [n_threads, n_agents, 1, state_dim]
        # allay [n_threads, n_agents, n_entity-1-n_enemy-wall, state_dim] = [n_threads, n_agents, ally, state_dim]

        # 提取uid信息(排除掉自己的信息)
        UID_binary = ally[..., :10] # [n_threads, n_agents, ally, 10]
        weights = np.power(2, np.arange(10, dtype=np.float32))
        UID = (UID_binary * weights).sum(axis=-1, keepdims=True)    # [n_threads, n_agents, ally, 1]
        UID = UID.squeeze(-1)   # [n_threads, n_agents, ally]

        s_UID_binary = zs[..., :10] # [n_threads, n_agents, 1, 10]
        s_UID_binary = s_UID_binary.squeeze(-2)
        s_UID = (s_UID_binary * weights).sum(axis=-1, keepdims=True) # [n_threads, n_agents, 1]
        s_UID = s_UID.squeeze(-1)   # [n_threads, n_agents]


        # 生成最终掩码 [n_threads, n_agents, n_agent]
        # 传递信息为[[n_threads, n_agent, n_agent]]

        # UID信息(移除非同队伍agent)*type_mask
        n_threads, n_agents, n_entity, _ = obs.shape
        n_agent = AlgorithmConfig.n_agent # 这里是全局智能体的数目，而不是batch内采样到智能体的数目
        output = np.zeros((n_threads, n_agents, n_agent), dtype=np.float32)

        for i in range(n_threads):
            for j in range(n_agents):
                s_id = int(s_UID[i,j])
                for m, M in enumerate(UID[i,j]):
                    a_id = int(UID[i,j,m])
                    if type_mask[s_id, a_id]==1 and s_id != a_id:  # 【type_mask 移除非同类型agent】【排除自己】
                        output[i,j,a_id] = 1

        return output
    
    def get_I_Het_mask(self, obs, type_mask, dead_mask):
        """
        利用知识的方式直接读取信息   知识构图准则：
        1. 移除dead agent     (obs内的信息已经满足条件)
        2. 移除非同队伍agent
        3. 通信距离内最近的几个 (obs内的信息已经满足条件)
        4. 排除自己
        """
  
        return self.get_I_Het_mask_uhmp(obs=obs, type_mask=type_mask, dead_mask=dead_mask)
    
   
    def get_I_Het_mask_uhmp(self, obs, type_mask, dead_mask):
        # uhmp部分对obs进行了切片[self=0, allay=1~int(n_entity_placeholder/2), enemy=int(n_entity_placeholder/2)~-2, wall=-1]
        # type_mask = type_mask.cpu().numpy() if type_mask is not None else type_mask  # type_mask [n_agent, n_agent]
        dead_mask = dead_mask.cpu().numpy() if dead_mask is not None else dead_mask  # dead_mask [n_threads, n_agents, n_entity]
        obs = obs.cpu().numpy() if 'cuda' in GlobalConfig.device else obs.numpy()    # obs[n_threads, n_agents, n_entity, state_dim]
        assert obs.shape[-1] == self.rawob_dim, '错误的观测信息，应该为没有经过预处理的信息！'
        # 由于此处只考虑队友通信，故将enemy和wall的信息放到了不用的Other里面
        zs, ally, other = self.div_entity(obs, type=[(0,), range(1, int(self.n_entity_placeholder/2)), range(int(self.n_entity_placeholder/2), self.n_entity_placeholder)], n=self.n_entity_placeholder)
        # zs [n_threads, n_agents, 1, state_dim]
        # allay [n_threads, n_agents, n_entity-1-n_enemy-wall, state_dim] = [n_threads, n_agents, ally, state_dim]

        # 提取uid信息(排除掉自己的信息)
        UID_binary = ally[..., :10] # [n_threads, n_agents, ally, 10]
        weights = np.power(2, np.arange(10, dtype=np.float32))
        UID = (UID_binary * weights).sum(axis=-1, keepdims=True)    # [n_threads, n_agents, ally, 1]
        UID = UID.squeeze(-1)   # [n_threads, n_agents, ally]

        s_UID_binary = zs[..., :10] # [n_threads, n_agents, 1, 10]
        s_UID_binary = s_UID_binary.squeeze(-2)
        s_UID = (s_UID_binary * weights).sum(axis=-1, keepdims=True) # [n_threads, n_agents, 1]
        s_UID = s_UID.squeeze(-1)   # [n_threads, n_agents]


        # 生成最终掩码 [n_threads, n_agents, n_agent]
        # UID信息(移除非同队伍agent)
        n_threads, n_agents, n_entity, _ = obs.shape
        n_agent = AlgorithmConfig.n_agent # 这里是全局智能体的数目，而不是batch内采样到智能体的数目
        output = np.zeros((n_threads, n_agents, n_agent), dtype=np.float32)

        for i in range(n_threads):
            for j in range(n_agents):
                s_id = int(s_UID[i,j])
                for m, M in enumerate(UID[i,j]):
                    a_id = int(UID[i,j,m])
                    if s_id != a_id:  # 【type_mask 移除非同类型agent】【排除自己】
                        output[i,j,a_id] = 1

        return output


# class I_GAT(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_heads=1, version=2):
#         super(I_GAT, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.n_heads = n_heads
#         self.version = version

#         assert n_heads == 1, '目前还没有涉及多头的形式!'
#         assert version == 2, '目前只有version2的形式! version2指adv信息直接用作权重计算'

#         # Version==1: 根据OA直接生成mask
#         # Version==2: 与E_GAT共享mask, 但是weighted_sum的权重根据OA进行计算而不是Wh TODO 这里的推导明显有问题

        
#         # 不采用多头的形式
#         self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
#         self.a = nn.Linear(hidden_dim*2, 1, bias=False)
#         self.act = nn.LeakyReLU(negative_slope=0.2)
#         # self.out_attn = nn.Linear(hidden_dim, output_dim)


#     def init_parameters(self):
#         for param in self.parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)
    
#     def forward(self, h, message, mask):
#         if self.version == 2:
#             return self.forward_version2(h, message, mask)

#     def forward_version2(self, h, message, mask):
#         # h        [input_dim]   
#         # Message  [n_agent, input_dim]   
#         # mask     [n_agent]    e.g: mask = torch.randint(0,2, (n_agent,))  
#         # MASK  mask 只保留距离内的 + 同类的，排除掉自己
#         n_agent = message.shape[0]

#         # 自身信息
#         h = torch.matmul(h, self.W)                #  (hidden_dim）
#         h_repeat = repeat_at(h, 0, n_agent)         #  (n_agent, hidden_dim）

#         # 接收到的观测信息（理论上应该是mask掉的，但是此处没有）
#         H = torch.matmul(message, self.W)          # （n_agent, hidden_dim）
        
#         # 求权重(记得最后还得mask一遍)
#         H_cat = torch.cat((h_repeat, H), dim=-1)   # （n_agent, hidden_dim * 2）
#         E = self.a(H_cat)                               # （n_agent, 1）
#         E = self.act(E)
#         E_mask = E * mask.unsqueeze(-1) 
#         alpha = F.softmax(E_mask, dim=0)                    # （n_agent, 1）
#         alpha_mask = alpha * mask.unsqueeze(-1)         # （n_agent, 1）
#         # 为了在这里解决 0*nan = nan 的问题，输入必须将nan转化为0
#         alpha_mask = torch.nan_to_num(alpha_mask, 0)

#         weighted_sum = torch.mul(alpha_mask, H).sum(dim=0)  #  (hidden_dim）
#         # H_E = self.out_attn(h + weighted_sum)
#         H_E = F.elu(h + weighted_sum)

#         return H_E





# # Original Version
# def get_E_Het_mask_uhmp(self, obs, type_mask, dead_mask):
#         # uhmp部分对obs进行了切片
#         # type_mask [n_agent, n_agent]
#         type_mask = type_mask.cpu().numpy() if type_mask is not None else type_mask
#         # dead_mask [n_threads, n_agents, n_entity]
#         dead_mask = dead_mask.cpu().numpy() if dead_mask is not None else dead_mask
#         # obs[n_threads, n_agents, n_entity, state_dim]
#         obs = obs.cpu().numpy() if 'cuda' in GlobalConfig.device else obs.numpy()
#         assert obs.shape[-1] == self.rawob_dim, '错误的观测信息，应该为没有经过预处理的信息！'

#         zs, ze = self.div_entity(obs, type=[(0,), range(1, self.n_entity_placeholder)], n=self.n_entity_placeholder)
#         # zs [n_threads, n_agents, 1, state_dim]
#         # ze [n_threads, n_agents, n_entity-1, state_dim]

#         # 提取所属队伍号信息
#         s_team = zs[..., 11]  # [n_threads, n_agents, 1]
#         o_team = ze[..., 11]  # [n_threads, n_agents, n_entity-1]
#         s_team = s_team.squeeze(-1)
#         s_team_expanded = np.broadcast_to(s_team[..., np.newaxis], o_team.shape)
#         is_equal_team = np.equal(s_team_expanded, o_team) # [n_threads, n_agents, n_entity-1]

#         # 提取uid信息(排除掉自己的信息)
#         UID_binary = ze[..., :10] # [n_threads, n_agents, n_entity-1, 10]
#         weights = np.power(2, np.arange(10, dtype=np.float32))
#         UID = (UID_binary * weights).sum(axis=-1, keepdims=True)    # [n_threads, n_agents, n_entity-1, 1]
#         UID = UID.squeeze(-1)   # [n_threads, n_agents, n_entity-1]

#         s_UID_binary = zs[..., :10] # [n_threads, n_agents, 1, 10]
#         s_UID_binary = s_UID_binary.squeeze(-2)
#         s_UID = (s_UID_binary * weights).sum(axis=-1, keepdims=True) # [n_threads, n_agents, 1]
#         s_UID = s_UID.squeeze(-1)   # [n_threads, n_agents]


#         # 生成最终掩码 [n_threads, n_agents, n_agent]
#         # 传递信息为[[n_threads, n_agent, n_agent]]

#         # UID信息(移除非同队伍agent)*type_mask
#         n_threads, n_agents, n_entity, _ = obs.shape
#         n_agent = AlgorithmConfig.n_agent # 这里是全局智能体的数目，而不是batch内采样到智能体的数目
#         output = np.zeros((n_threads, n_agents, n_agent), dtype=np.float32)
#         for i in range(n_threads):
#             for j in range(n_agents):
#                 for m, M in enumerate(is_equal_team[i,j]):
#                     if M: # 【移除非同队伍agent】
#                         if type_mask[int(s_UID[i,j]), int(UID[i,j,m])]==1:  # 【type_mask 移除非同类型agent】
#                             output[i,j,int(UID[i,j,m])] = 0 if int(s_UID[i,j])==int(UID[i,j,m]) else 1 # 【排除自己】
#         return output