import numpy as np

class Knowledge():
    def __init__(self, num_requirement_type, num_plane_type):
        '''
            用来存储场景中的已知信息, 如每类飞机能够满足的需求
        '''
        self.num_requirement_type = num_requirement_type # 有几类需求飞机能力向量和目标需求向量的维度就是几维
        self.num_plane_type = num_plane_type # 这里是飞机挂载的类型而非飞机类型
        self.plane_type = {}
        # 存储每类飞机挂载选择的能力向量
        self.PLANE_CAPACITY = [
            [4,4,2,0,0,0,0], # 第一类飞机挂载种类




        ]
        assert  self.PLANE_CAPACITY.shape[-1] == self.num_requirement_type, ("PLANE_CAPACITY 的格式有问题！")
        assert  self.PLANE_CAPACITY.shape[-2] == self.num_plane_type, ("PLANE_CAPACITY 的格式有问题！")

        # 存储每类飞机挂载实际对应的飞机类型
        self.REAL_PLANE = [
            0,  # 飞机类型1
            0,
            0,
            1,  # 飞机类型2 
            1,
            1,
            2,  # 飞机类型3 
            2,
            2,
            2,
            3,  # 飞机类型4
            4,  # 飞机类型5
        ]
        assert self.REAL_PLANE.shape[-1] == num_plane_type, ("REAL_PLANE格式有问题！")
        # 存储每类子目标的需求向量
        TARGET_REQUIREMENT = [
            [4,4,0,0,0,0,0], # 第一类目标需求种类


        ]
        assert TARGET_REQUIREMENT.shape[-1] == self.num_requirement_type, ("TARGET_REQUIREMENT 的格式有问题！")
        return TARGET_REQUIREMENT
    

    

    def get_plane_capacity(self):
        '''
            返回每类飞机挂载选择的能力向量
        '''
        return self.PLANE_CAPACITY
    def get_plane_type(self):
        ''' 
            返回每类飞机挂载实际对应的飞机类型
        '''
        return self.REAL_PLANE
    def get_target_capacity(self):
        '''
            返回每类子目标的需求向量
        '''

 