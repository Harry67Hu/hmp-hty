import os, torch
import numpy as np
from numba import njit, jit
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
import pickle
from config import GlobalConfig
DEBUG = True

# @njit
def distance_matrix(A):
    assert A.shape[-1] == 2 # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis

def stack_padding(l):
    import itertools
    return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))

def dir_to_rad_angle(delta_pos):
    result = np.empty(delta_pos.shape[:-1], dtype=complex)
    result.real = delta_pos[...,0]; result.imag = delta_pos[...,1]
    rad_angle = np.angle(result) 
    return rad_angle

def reg_angle_deg(deg):
    return (deg + 180)%360 -180

def reg_angle(rad):
    # it's OK to show "RuntimeWarning: invalid value encountered in remainder"
    return (rad + np.pi)%(2*np.pi) -np.pi

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, RL_functional, 
                                          alg_config, scenario_config):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        self.n_basic_dim = scenario_config.obs_vec_length

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(scenario_config, 'AvailActProvided'):
            self.AvailActProvided = scenario_config.AvailActProvided 

        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
        self.cold_start = True

    @staticmethod
    def get_binary_array(n, n_bits, dtype=np.float32):
        arr = np.zeros(n_bits, dtype=dtype)
        pointer = 0
        while True:
            arr[pointer] = int(n%2==1)
            n = n >> 1
            pointer += 1
            if n == 0: break
        return arr

    def interact_with_env(self, State_Recall):
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = State_Recall['Latest-Obs']
        n_thread = obs.shape[0]

        previous_obs = State_Recall['_Previous_Obs_'] if '_Previous_Obs_' in State_Recall else np.zeros_like(obs)

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        obs_feed = obs[~ENV_PAUSE]
        prev_obs_feed = previous_obs[~ENV_PAUSE]

        obs_feed_in = self.solve_duplicate(obs_feed, prev_obs_feed)

        I_State_Recall = {'obs':obs_feed_in, 
            'Test-Flag':State_Recall['Test-Flag'], 
            'threads_active_flag':~ENV_PAUSE, 
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~ENV_PAUSE],
            }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(State_Recall['Latest-Team-Info'][~ENV_PAUSE], dtype=object)])
            I_State_Recall.update({'avail_act':avail_act})

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~ENV_PAUSE] = act_active
        actions_list = []
        act = np.expand_dims(act, -1)
        for i in range(self.n_agent): actions_list.append(act[:,i,:])
        actions_list = np.array(actions_list)

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        State_Recall['_Previous_Obs_'] = obs
        State_Recall['_hook_'] = internal_recall['_hook_']
        assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

    def solve_duplicate(self, obs_feed, prev_obs_feed):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed = my_view(obs_feed,[0, 0, -1, self.n_basic_dim])

        # turning all zero padding to NaN, used for normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan

        return obs_feed


    def get_mask_id(self, obs_feed):
        mask_and_id = np.zeros_like(obs_feed)[:,:,:, 0] # thread,agent,agent_obs
        binary = obs_feed[...,-8:]
        alive = obs_feed[..., 0]
        for i in range(8):
            mask_and_id += binary[..., i]* 2**i
        # print(mask_and_id)
        mask_and_id = np.where(alive==1, mask_and_id, np.nan)
        return mask_and_id


