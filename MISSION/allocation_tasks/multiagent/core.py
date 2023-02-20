import numpy as np
from .basic_knowledge import Knowledge

# State
class EntityState(object):
    def __init__(self):
        '''
            此处定义实体模型的基本状态
            使用样例 e.g :entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
        '''
        self.is_alive = None

class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        '''
            此处定义基地智能体的基本状态 (可以跟随动作变化)
            1. 基地内各类飞机剩余数目
            2. 基地内各类载荷剩余数目(可能有）
        '''

        self.num_plane = None # 飞机剩余数目列表，在scenario中进行初始化

class TargetState(EntityState):
    def __init__(self):
        super(TargetState, self).__init__()
        '''
            此处定义目标的基本状态 (可以跟随动作变化)
            1. 目标需求
            2. 目标优先级
        '''
        
        self.num_requirement = None # 目标需求列表,可以在在scenario中进行初始化,也可以通过type进行加载
        self.priority = None


# Action 
class Action(object):
    def __init__(self):
        '''
            此处定义基地智能体的动作
            使用样例 e.g : state[t+1] = state[t] + agent.action.example 
        '''

        self.example = None

# Entity Model
class Entity(object):
    def __init__(self):
        '''
            此处定义可以调用的实体模型
        '''
        self.state = EntityState()
        # name 
        self.name = ''
        # properties:
        self.size = None
        self.location = None
        

class Target(Entity):
     def __init__(self):
        super(Target, self).__init__()
        '''
            此处定义目标模型的基本属性
            基本状态以外的模型和参数
            1. 目标位置
            2. 目标不确定性因素 TODO ?
        '''
        # state
        self.state = TargetState()
        # e.g: 
        self.color = None
        self.type = 0 # 目标种类

class Agent(Entity):
    def __init__(self, iden=None):
        super(Agent, self).__init__()
        '''
            此处定义基地智能体模型的基本属性
            基本状态以外的模型和参数
            1. 基地位置
            2. 
        '''
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        self.action_callback = None

        

# World Model with entity model 
class World(object):
    def __init__(self):

        self.num_requirement_type = 
        self.num_plane_type = 

        self.agents = []
        self.targets = []
        self.steps = 0
        self.MaxEpisodeStep = 50

    # debug使用
    @property
    def entities(self):
        return self.agents + self.targets

    # 调用基于强化学习的基地智能体
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # 给运筹等其他算法开放的接口智能体
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        '''
            此处为环境的基本运行规则   (有些部分虽然与reward相关, 但是放到规则部分, reward负责直接调用)
            0. 环境基本状态转移模型
            1. 载荷及飞行成本模型
            2. 作战效益模型
        '''
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # 0. 环境基本状态转移模型
        # 环境整体模型
        self.integrate_state()
        # 基地智能体状态模型
        for agent in self.agents:
            self.update_agent_state(agent)
        self.steps += 1

    def integrate_state(self, p_force):
        # e.g.:
        for i,entity in enumerate(self.entities):
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    
