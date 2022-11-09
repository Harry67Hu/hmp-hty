base = """
{
    "config.py->GlobalConfig": {
        "note": "qmix-catobs-run1",
        "env_name": "uhmap",
        "env_path": "MISSION.uhmap",
        "draw_mode": "Img",
        "num_threads": 8,
        "report_reward_interval": 256,
        "test_interval": 5120,
        "test_epoch": 256,
        "interested_team": 0,
        "seed": 8529,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "ALGORITHM/pymarl2_compat",
            "MISSION/uhmap"
        ]
    },
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "n_team1agent": 10,
        "n_team2agent": 10,
        "MaxEpisodeStep": 125,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false,
        "UElink2editor": false,
        "AutoPortOverride": true,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapLargeScale",
        "UhmapRenderExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxServer_Ver1.0/UHMPServer.sh",
        "TimeDilation": 64,
        "TEAM_NAMES": [
            "ALGORITHM.pymarl2_compat.pymarl2_compat->PymarlFoundation",
            "ALGORITHM.script_ai.uhmap_ls->DummyAlgorithmLinedAttack"
        ]
    },
    "MISSION.uhmap.SubTasks.UhmapLargeScaleConf.py->SubTaskConfig":{
        "agent_list": [
            { "team":0,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

            { "team":1,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
        ]
    },


    "ALGORITHM.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },
    "ALGORITHM.pymarl2_compat.pymarl2_compat.py->AlgorithmConfig": {
        "use_shell": "mini_shell_uhmap",
        "state_compat": "obs_cat",
        "pymarl_config_injection": {
            "controllers.my_n_controller.py->PymarlAlgorithmConfig": {
                "use_normalization": "True",
                "use_vae": "False"
            },
            "config.py->GlobalConfig": {
                "batch_size": 128,
                "load_checkpoint": "False"
            }
        }
    }
}
"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 4
n_run_mode = [
    {
        "addr": "localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "uhmap-qmix-catobs"
conf_override = {


    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],

    "config.py->GlobalConfig-->device":
        [
            'cuda:2',
            'cuda:2',
            'cuda:4',
            'cuda:4',
        ],

    "config.py->GlobalConfig-->gpu_fraction":
        [
            0.5,
            0.5,
            0.5,
            0.5,
        ],


    "config.py->GlobalConfig-->note":
        [
            "run1",
            "run2",
            "run3",
            "run4",
        ],

}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override)