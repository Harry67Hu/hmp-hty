import torch, math, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, repeat_at
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from config import GlobalConfig
from .foundation import AlgorithmConfig
from ALGORITHM.common.norm import DynamicNormFix
from ALGORITHM.common.conc import Concentration
from ALGORITHM.common.net_manifest import weights_init




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
        # self.out_attn = nn.Linear(hidden_dim, output_dim)


        # 多头的遗弃版本
        # self.W = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        # self.a = nn.Parameter(torch.Tensor(n_heads, hidden_dim * 2))
        # self.attn_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_heads)])
        # self.out_attn = nn.Linear(n_heads * hidden_dim, output_dim)


    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, message, mask):
        # h维度为[input_dim]   
        # message维度为[n_agent, input_dim]   
        # mask维度为[n_agent]    e.g: mask = torch.randint(0,2, (n_agent,))  
        # MASK  mask 只保留距离内的 + 同类的，排除掉自己
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
        alpha = F.softmax(E_mask, dim=0)                    # （n_agent, 1）
        alpha_mask = alpha * mask.unsqueeze(-1)         # （n_agent, 1）

        weighted_sum = torch.mul(alpha_mask, H).sum(dim=-2)  #  (hidden_dim）
        # H_E = self.out_attn(h + weighted_sum)
        H_E = F.elu(h + weighted_sum)

        return H_E

class I_GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=1, version=1):
        super(I_GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.version = version

        assert n_heads == 1, '目前还没有涉及多头的形式!'
        assert version == 2, '目前只有version2的形式! version2指adv信息直接用作权重计算'

        # Version==1: 根据OA直接生成mask
        # Version==2: 与E_GAT共享mask, 但是weighted_sum的权重根据OA进行计算而不是Wh TODO 这里的推导明显有问题

        
        # 不采用多头的形式
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.a = nn.Linear(hidden_dim*2, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        # self.out_attn = nn.Linear(hidden_dim, output_dim)


    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, message, mask):
        if self.version == 2:
            return self.forward_version2(h, message, mask)

    def forward_version2(self, h, message, mask):
        # h        [input_dim]   
        # Message  [n_agent, input_dim]   
        # mask     [n_agent]    e.g: mask = torch.randint(0,2, (n_agent,))  
        # MASK  mask 只保留距离内的 + 同类的，排除掉自己
        n_agent = message.shape[0]

        # 自身信息
        h = torch.matmul(h, self.W)                #  (hidden_dim）
        h_repeat = repeat_at(h, 0, n_agent)         #  (n_agent, hidden_dim）

        # 接收到的观测信息（理论上应该是mask掉的，但是此处没有）
        H = torch.matmul(message, self.W)          # （n_agent, hidden_dim）
        
        # 求权重(记得最后还得mask一遍)
        H_cat = torch.cat((h_repeat, H), dim=-1)   # （n_agent, hidden_dim * 2）
        E = self.a(H_cat)                               # （n_agent, 1）
        E = self.act(E)
        E_mask = E * mask.unsqueeze(-1) 
        alpha = F.softmax(E_mask, dim=0)                    # （n_agent, 1）
        alpha_mask = alpha * mask.unsqueeze(-1)         # （n_agent, 1）

        weighted_sum = torch.mul(alpha_mask, H).sum(dim=0)  #  (hidden_dim）
        # H_E = self.out_attn(h + weighted_sum)
        H_E = F.elu(h + weighted_sum)

        return H_E







"""
    network initialize
"""
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

        # act_dim = ???
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
        self.AT_I_Het_GAT = I_GAT(input_dim=adv_h_dim, hidden_dim=GAT_h_dim, output_dim=H_I_dim)


        self.AT_PGAT_mlp = nn.Sequential(nn.Linear(H_E_dim + H_I_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, obs_h_dim))  # 此处默认h_dim是一致的
        
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

        self.weights = torch.pow(2, torch.arange(10, dtype=torch.float32)).to(GlobalConfig.device)
        # self.Temp = torch.zeros_like(UID).to(GlobalConfig.device)


        return

    def act(self, *args, **kargs):
        act = self._act
        return act(*args, **kargs)

    def evaluate_actions(self, *args, **kargs):
        act = self._act
        return act(*args, **kargs, eval_mode=True)

    # div entity for DualConc models, distincting friend or hostile (present or history)
    # def div_entity(self, mat, n=22, core_dim=None):
    #     assert n == self.n_entity_placeholder
    #     assert n == mat.shape[core_dim]
    #     type =  AlgorithmConfig.entity_distinct
    #     if core_dim == -2:
    #         tmp = (mat[..., t, :] for t in type)
    #     elif core_dim == -1:
    #         tmp = (mat[..., t] for t in type)
    #     else:
    #         assert False, "please make sure that the number of entities is correct, should be %d"%mat.shape[-2]
    #     return tmp
    
    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, type=[(0,), (1, 2, 3, 4, 5),(6, 7, 8, 9, 10, 11)], n=12):
        if mat.shape[-2]==n:
            tmp = (mat[..., t, :] for t in type)
        elif mat.shape[-1]==n:
            tmp = (mat[..., t] for t in type)
        return tmp
    

    def get_E_Het_mask(self, obs):
        """
        利用知识的方式直接读取信息   知识构图准则：
        1. 移除dead agent     (obs内的信息已经满足条件)
        2. 移除非同队伍agent
        3. 移除非同类型agent 
        4. 通信距离内最近的几个 (obs内的信息已经满足条件)
        5. 排除自己
        """
        return self.get_E_Het_mask_uhmp(obs=obs)
    
    def get_E_Het_mask_uhmp(self, obs):
        # obs[n_threads, n_agents,||| n_entity, state_dim]
        assert obs[-1] == self.rawob_dim, '错误的观测信息，应该为没有经过预处理的信息！'
        zs, ze = self.div_entity(obs,       type=[(0,), range(1, self.n_entity_placeholder)], n=self.n_entity_placeholder)
        # zs [n_threads, n_agents,||| 1, state_dim]  ze [n_threads, n_agents,||| n_entity-1, state_dim]
        # 提取type信息
        s_type = zs[...,(-3,-2,-1)]             # [n_threads, n_agents,||| 1, 3]
        o_type = ze[...,(-3,-2,-1)]             # [n_threads, n_agents,||| n_entity-1, 3]    
        # 提取所属队伍号信息      
        s_team = zs[...,(10)]
        o_team = ze[...,(10)]      

        # 提取uid信息
        UID_binary = ze[...,range(10)]           # [n_threads, n_agents,||| n_entity-1, 10]
        # uid二进制转换十进制的方法（并行计算）
        # weights = torch.pow(2, torch.arange(10, dtype=torch.float32)).to(GlobalConfig.device)
        UID = (UID_binary * self.weights).sum(dim=-1, keepdim=True)

        # 生成掩码后的UID信息
        # 将s_type扩展到o_type相同的维度
        is_equal_type = torch.eq(s_type, o_type)   # 比较两个张量的每个元素是否相等
        is_equal_team = torch.eq(s_team, o_team)
        # 根据类别判断生成mask后的UID
        is_all_equal_type = torch.all(is_equal_type, dim=-1) # 检查比较结果中是否所有元素都为True
        is_all_equal_team = torch.all(is_equal_team, dim=-1)
        UID = UID.squeeze(-1)
        Temp = torch.zeros_like(UID).to(GlobalConfig.device)
        # UID_masked = torch.where(is_all_equal, UID, torch.empty_like(UID).fill_(float('0')))  # [n_threads, n_agents,||| n_entity-1]
        UID_masked_temp = torch.where(is_all_equal_type, UID, Temp)  # [n_threads, n_agents,||| n_entity-1]
        UID_masked = torch.where(is_all_equal_team, UID_masked_temp, Temp)

        # 生成最终掩码 [n_threads, n_agents,||| n_agent] 
        n_threads = obs.size(0)
        n_agents = obs.size(1)
        n_agent = AlgorithmConfig.n_agent
        output = torch.zeros((n_threads, n_agents, n_agent), dtype=torch.float32).to(GlobalConfig.device)
        mask = output.scatter_(-1, UID_masked.long(), 1.0) 

        # 最后简单粗暴的将所有智能体对0号智能体屏蔽掉（影响不大）
        mask[:,:,0] = 0

        return mask
    
    def get_I_Het_mask(self, obs):
        """
        利用知识的方式直接读取信息   知识构图准则：
        1. 移除dead agent     (obs内的信息已经满足条件)
        2. 移除非同队伍agent
        3. 通信距离内最近的几个 (obs内的信息已经满足条件)
        4. 排除自己
        """
        return self.get_I_Het_mask_uhmp(obs=obs)
    
    def get_I_Het_mask_uhmp(self, obs):
        # obs[n_threads, n_agents,||| n_entity, state_dim]
        assert obs[-1] == self.rawob_dim, '错误的观测信息，应该为没有经过预处理的信息！'
        zs, ze = self.div_entity(obs,       type=[(0,), range(1, self.n_entity_placeholder)], n=self.n_entity_placeholder)
        # zs [n_threads, n_agents,||| 1, state_dim]  ze [n_threads, n_agents,||| n_entity-1, state_dim]
        
        # 提取所属队伍号信息      
        s_team = zs[...,(10)]
        o_team = ze[...,(10)]      

        # 提取uid信息
        UID_binary = ze[...,range(10)]           # [n_threads, n_agents,||| n_entity-1, 10]
        UID = (UID_binary * self.weights).sum(dim=-1, keepdim=True)

        # 生成掩码后的UID信息
        is_equal_team = torch.eq(s_team, o_team)
        # 根据类别判断生成mask后的UID
        is_all_equal_team = torch.all(is_equal_team, dim=-1)
        UID = UID.squeeze(-1)
        Temp = torch.zeros_like(UID).to(GlobalConfig.device)
        # UID_masked = torch.where(is_all_equal, UID, torch.empty_like(UID).fill_(float('0')))  # [n_threads, n_agents,||| n_entity-1] 
        UID_masked = torch.where(is_all_equal_team, UID, Temp) # [n_threads, n_agents,||| n_entity-1]

        # 生成最终掩码 [n_threads, n_agents,||| n_agent] 
        n_threads = obs.size(0)
        n_agents = obs.size(1)
        n_agent = AlgorithmConfig.n_agent
        output = torch.zeros((n_threads, n_agents, n_agent), dtype=torch.float32).to(GlobalConfig.device)
        mask = output.scatter_(-1, UID_masked.long(), 1.0) 

        # 最后简单粗暴的将所有智能体对0号智能体屏蔽掉（影响不大）
        mask[:,:,0] = 0

        return mask

    




    def _act(self, obs=None, act=None, message_obs=None, message_adv=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        assert not (self.forbidden)
        if self.static:
            assert self.gp >=1
        # if not test_mode: assert not self.forbidden
        eval_act = eval_actions if eval_mode else None
        others = {}
        assert self.n_entity_placeholder == obs[-2], 'observation structure wrong!'

        # Obs预处理部分
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs, freeze=(eval_mode or test_mode or self.static))
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0  obs [n_threads, n_agents, n_entity, rawob_dim]
        E_Het_mask = self.get_E_Het_mask(obs=obs)     # [n_threads, n_agents, n_agent] # warning n_agents是共享网络的智能体数据，n_agent是全局智能体数目
        I_Het_mask = self.get_I_Het_mask(obs=obs)
        if self.use_obs_pro_uhmp:
            s, other = self.div_entity(obs,       type=[(0,), range(1, self.n_entity_placeholder)], n=self.n_entity_placeholder)
            s = s.squeeze(-2)                                               # [n_threads, n_agents, rawob_dim]
            other = other.reshape(other[0], other[1], other[-2]*other[-1])  # [n_threads, n_agents, n_entity-1 * rawob_dim]
            print(other.size)
            zs = self.state_encoder(s)          # [n_threads, n_agents, obs_process_h_dim]
            zo = self.entity_encoder(other)     # [n_threads, n_agents, obs_process_h_dim]
            obs = torch.cat((zs, zo), -1)     # [n_threads, n_agents, obs_process_h_dim * 2]
        else:
            obs = obs.reshape(obs[0], obs[1], obs[-2]*obs[-1])  # [n_threads, n_agents, n_entity * rawob_dim]

        # 环境观测理解部分
        h_obs = self.AT_obs_encoder(obs)
        
        # 环境策略建议部分
        abstract_obs = self.AT_obs_abstracter(obs)
        abstract_act = self.AT_act_abstractor(act)

        abstract_cat = torch.cat((abstract_obs, abstract_act), -1)
        gru_input = F.relu(self.fc1_rnn(abstract_cat))

        self.gru_cell_memory = self.gru(gru_input, self.gru_cell_memory)
        h_adv = self.fc2_rnn(self.gru_cell_memory)


        # PGAT部分
        H_E = self.AT_E_Het_GAT(h_obs, message_obs, E_Het_mask)
        H_I = self.AT_I_Het_GAT(h_adv, message_adv, I_Het_mask)
        H_sum = self.AT_PGAT_mlp(torch.cat((H_E, H_I), -1))
    
        # 策略网络部分
        logits = self.AT_policy_head(H_sum)






        # # motivation objectives
        if eval_mode: 
            # threat = self.CT_get_threat(v_M_fuse)
            value = self.CT_get_value(H_sum)
            # others['threat'] = self.re_scale(threat, limit=12)
            others['value'] = value
            
        logit2act = self._logit2act
        if self.use_policy_resonance and self.is_resonance_active():
            logit2act = self._logit2act_rsn
            
        act, actLogProbs, distEntropy, probs = logit2act(   logits, eval_mode=eval_mode,
                                                            test_mode=(test_mode or self.static), 
                                                            eval_actions=eval_act, 
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)

        message_obs_output = h_obs 
        message_adv_output = h_adv


        if not eval_mode: return act, 'vph', actLogProbs, message_obs_output, message_adv_output
        else:             return 'vph', actLogProbs, distEntropy, probs, others, message_obs_output, message_adv_output

    @staticmethod
    def re_scale(t, limit):
        r = 1. /2. * limit
        return (torch.tanh_(t/r) + 1.) * r

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
    
    

    
    
    
"""
    之后的部分在此处没有用到！
"""
class NetCentralCritic(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        super().__init__()

        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.policy_resonance
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        h_dim = AlgorithmConfig.net_hdim
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        self.skip_connect = True
        self.n_action = n_action

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.CT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))


        tmp_dim = h_dim if not self.dual_conc else h_dim*2
        self.CT_get_value = nn.Sequential(nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))


        self.is_recurrent = False
        self.apply(weights_init)
        return

    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, n=22, core_dim=None):
        assert n == self.n_entity_placeholder
        assert n == mat.shape[core_dim]
        type =  AlgorithmConfig.entity_distinct
        if core_dim == -2:
            tmp = (mat[..., t, :] for t in type)
        elif core_dim == -1:
            tmp = (mat[..., t] for t in type)
        else:
            assert False, "please make sure that the number of entities is correct, should be %d"%mat.shape[-2]
        return tmp


    def estimate_state(self, obs=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass # 某一种类型的智能体全体阵亡
            else:
                obs = self._batch_norm(obs, freeze=(eval_mode or test_mode))

        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.CT_obs_encoder(obs)

        zs, ze_f, ze_h          = self.div_entity(obs,       n=self.n_entity_placeholder, core_dim=-2)
        vs, ve_f, ve_h          = self.div_entity(v,         n=self.n_entity_placeholder, core_dim=-2)
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead, n=self.n_entity_placeholder, core_dim=-1)

        # concentration module
        _, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        _, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # motivation encoding fusion
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        value = self.CT_get_value(v_M_fuse)

        return value



