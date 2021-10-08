#from DQNagent import DQN
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import math
import gym
#import random
#from gym import spaces, logger
from gym.utils import seeding
#import scipy.io as scio
from scipy.stats import wasserstein_distance
from collections import defaultdict
from keras.models import load_model
import time

class sensorbelief(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
     """

    def __init__(self):
       
        dataFile1 = 'C:/code/10221tran.txt'
        with open(dataFile1, 'r') as f:
            self.tran1022  = [[float(num) for num in line.split('\t')] for line in f]
            self.tran1022  = np.array(self.tran1022)
        dataFile2 = 'C:/code/10221emission.txt'
        with open(dataFile2, 'r') as f:
            self.emission1022  = [[float(num) for num in line.split('\t')] for line in f]
            self.emission1022 = np.array(self.emission1022)
        dataFile3 = 'C:/code/10221prior.txt'
        self.prior1022 = np.loadtxt(dataFile3)
#        with open(dataFile3, 'r') as f:
#            self.prior1022  = [[float(num) for num in line.split('\t')] for line in f]
#            self.prior1022 = np.array(self.prior1022)
        self.tz1022=839
        self.ts1022=429
        self.span1022=8555
        
        dataFile4 = 'C:/code/10291tran.txt'
        with open(dataFile4, 'r') as f:
            self.tran1029  = [[float(num) for num in line.split('\t')] for line in f]
            self.tran1029  = np.array(self.tran1029)
        dataFile5 = 'C:/code/10291emission.txt'
        with open(dataFile5, 'r') as f:
            self.emission1029  = [[float(num) for num in line.split('\t')] for line in f]
            self.emission1029  = np.array(self.emission1029)
        dataFile6 = 'C:/code/10291prior.txt'
        self.prior1029 = np.loadtxt(dataFile6)
#        with open(dataFile6, 'r') as f:
#            self.prior1029  = [[float(num) for num in line.split('\t')] for line in f]
#            self.prior1029 = np.array(self.prior1029)
        self.tz1029=794
        self.ts1029=508
        self.span1029=8374
        
        dataFile7 = 'C:/code/10551tran.txt'
        with open(dataFile7, 'r') as f:
            self.tran1055  = [[float(num) for num in line.split('\t')] for line in f]
            self.tran1055  = np.array(self.tran1055)
        dataFile8 = 'C:/code/10551emission.txt'
        with open(dataFile8, 'r') as f:
            self.emission1055  = [[float(num) for num in line.split('\t')] for line in f]
            self.emission1055  = np.array(self.emission1055)
        dataFile9 = 'C:/code/10551prior.txt'
        self.prior1055 = np.loadtxt(dataFile9)
#        with open(dataFile9, 'r') as f:
#            self.prior1055  = [[float(num) for num in line.split('\t')] for line in f]
#            self.prior1055  = np.array(self.tran1022)
        self.tz1055=944
        self.ts1055=458
        self.span1055=8520
        
        dataFile10 = 'C:/code/10981tran.txt'
        with open(dataFile10, 'r') as f:
            self.tran1098  = [[float(num) for num in line.split('\t')] for line in f]
            self.tran1098  = np.array(self.tran1098)
        dataFile11 = 'C:/code/10981emission.txt'
        with open(dataFile11, 'r') as f:
            self.emission1098  = [[float(num) for num in line.split('\t')] for line in f]
            self.emission1098  = np.array(self.emission1098)
        dataFile12 = 'C:/code/10981prior.txt'
        self.prior1098 = np.loadtxt(dataFile12)
#        with open(dataFile12, 'r') as f:
#            self.prior1098  = [[float(num) for num in line.split('\t')] for line in f]
#            self.prior1098 = np.array(self.prior1098)
        self.tz1098=784
        self.ts1098=379
        self.span1098=8604
        
        dataFile13 = 'C:/code/11871tran.txt'
        with open(dataFile13, 'r') as f:
            self.tran1187  = [[float(num) for num in line.split('\t')] for line in f]
            self.tran1187 = np.array(self.tran1187)
        dataFile14 = 'C:/code/11871emission.txt'
        with open(dataFile14, 'r') as f:
            self.emission1187  = [[float(num) for num in line.split('\t')] for line in f]
            self.emission1187 = np.array(self.emission1187)
        dataFile15 = 'C:/code/11871prior.txt'
        self.prior1187 = np.loadtxt(dataFile15)
#        with open(dataFile15, 'r') as f:
#            self.prior1187  = [[float(num) for num in line.split('\t')] for line in f]
#            self.prior1187 = np.array(self.prior1187)
        self.tz1187=992
        self.ts1187=386
        self.span1187=8523
        
        
        self.tem = np.linspace(-5,24,num=291,dtype=float) ##temperature  range
        self.zero = np.arange(11950,13191,dtype=np.int)  ## zero range
        self.CO2 = np.arange(0,801,dtype=np.int)  ## true CO2 range
        
        self.obs1022 = np.arange(315,650,dtype=np.int)  ## observation range
        self.obs1029 = np.arange(409,697,dtype=np.int)
        self.obs1055 = np.arange(437,701,dtype=np.int)
        self.obs1098 = np.arange(129,349,dtype=np.int)
        self.obs1187 = np.arange(378,650,dtype=np.int)
        
        self.IR1022 = np.arange(38431,43542,dtype=np.int) ## IR range
        self.IR1029 = np.arange(38024,41855,dtype=np.int)
        self.IR1055 = np.arange(38162,43052,dtype=np.int)
        self.IR1098 = np.arange(38035,43904,dtype=np.int)
        self.IR1187 = np.arange(38129,43774,dtype=np.int)
        
        self.tablex=np.array([0,4096,8192,12290,16380,20480,24580,28670,32770,36860,40960,45060,49150,53250,57340,61440,65540])
        self.table1022=np.array([10000,8594,7356,6263,5293,4432,3667,2991,2397,1881,1436,1055,729,450,209,0,-209])
        self.table1029=np.array([10000,8586,7343,6249,5282,4424,3664,2990,2396,1878,1431,1047,721,443,205,0,-205])
        self.table1055=np.array([10000,8596,7360,6267,5297,4436,3672,2996,2402,1885,1440,1058,731,451,210,0,-210])
        self.table1098=np.array([10000,8591,7351,6256,5286,4425,3661,2985,2392,1877,1433,1053,728,449,209,0,-209])
        self.table1187=np.array([10000,8591,7351,6256,5286,4425,3661,2985,2392,1877,1433,1053,728,449,209,0,-209])

        self.action_space = np.arange(0,5,dtype=np.int)
        self.observation_space = np.zeros(801,dtype=float)

        self.state = None

        self.steps_beyond_done = None

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        ## err_msg = "%r (%s) invalid" % (action, type(action))
        ## assert self.action_space.contains(action), err_msg
#        改成向量运算
#        print('[step]: staret')
#        tic = time.time()
#        state_transpose=self.state.reshape(801,1)
        if action==0:
#            print('[step]: action 0')
            self.beliefCO2_calibrated=self.CO2_1022dis # calibration with leader sensor
            sensor_index=0  
            ylprioir_1022=np.zeros(len(self.obs1022),dtype=float)
            self.new_emission_1022dis=np.zeros([len(self.CO2),len(self.obs1022)],dtype=float)
            for j in range (len(self.obs1022)):
                ylvector_1022=self.emission1022[:,j]
                self.new_emission_1022dis[:,j]=self.calculateemission(ylvector_1022,sensor_index)              
            ylprioir_1022=np.dot(self.state,self.new_emission_1022dis)               
            reward_item=np.zeros([len(self.CO2),len(self.obs1022)],dtype=float)
            reward_item1=np.zeros(len(self.obs1022),dtype=float)
            for j in range (len(self.obs1022)):
#                if j == (len(self.obs1022) - 1):
#                    print('[step]: action 0-for-loop', i)
                if ylprioir_1022[j]== 0:
                    reward_item[:,j]=np.zeros(len(self.CO2),dtype=float)
                else:
                    reward_item[:,j]=(self.state*self.new_emission_1022dis[:,j])/ylprioir_1022[j]
                    nonzero_idx=np.nonzero(reward_item[:,j])
                    term1=reward_item[:,j].take(nonzero_idx)
                    term2=np.log(term1)
                    reward_item1[j]=(term1*term2).sum()
            reward=(reward_item1*ylprioir_1022).sum()
#            for i in range(len(self.CO2)):
#                for j in range (len(self.obs1022)):
#                    ylvector_1022=self.emission1022[:,j]
#                    self.calculateemission(ylvector_1022,sensor_index)              
#                    ylprioir_1022[j]=(self.state*self.emission_1022dis).sum()
#                    if j == (len(self.obs1022) - 1):
#                        print('[step]: action 0-for-loop', i)
#                    if ylprioir_1022[j]== 0:
#                        reward_item[i,j]=0
#                    else:
#                        reward_item[i,j]=(self.state[i]*self.emission_1022dis[i])/ylprioir_1022[j] #calculate term P(CO2_t|y_{1:t-1}^{1:N},y_t^l,Tem_{1:t},IR_{1:t}^{1:N})
#            for k in range(len(self.obs1022)):
#                term1=reward_item[:,k]
#                term2=np.log(reward_item[:,k])
#                reward_item1[k]=(term1*term2).sum() # calculate P*logP
           
#            reward=-(reward_item1*ylprioir_1022).sum()
            """
            tran=self.1022tran
            emission=self.1022emission
            prior=self.1022prior
            """
        if action==1:
#            print('[step]: action 1')
            self.beliefCO2_calibrated=self.CO2_1029dis
            sensor_index=1  
            ylprioir_1029=np.zeros(len(self.obs1029),dtype=float)
            self.new_emission_1029dis=np.zeros([len(self.CO2),len(self.obs1029)],dtype=float)
            for j in range (len(self.obs1029)):
                ylvector_1029=self.emission1029[:,j]
                self.new_emission_1029dis[:,j]=self.calculateemission(ylvector_1029,sensor_index)         
            ylprioir_1029=np.dot(self.state,self.new_emission_1029dis)
            reward_item=np.zeros([len(self.CO2),len(self.obs1029)],dtype=float)
            reward_item1=np.zeros(len(self.obs1029),dtype=float)
            for j in range (len(self.obs1029)):
#                if j == (len(self.obs1022) - 1):
#                    print('[step]: action 0-for-loop', i)
                if ylprioir_1029[j]== 0:
                    reward_item[:,j]=np.zeros(len(self.CO2),dtype=float)
                else:
                    reward_item[:,j]=(self.state*self.new_emission_1029dis[:,j])/ylprioir_1029[j]
                    nonzero_idx=np.nonzero(reward_item[:,j])
                    term1=reward_item[:,j].take(nonzero_idx)
                    term2=np.log(term1)
                    reward_item1[j]=(term1*term2).sum()
            reward=(reward_item1*ylprioir_1029).sum()
#            for i in range(len(self.CO2)):
#                for j in range (len(self.obs1029)):
#                    ylvector_1029=self.emission1029[:,j]
#                    self.calculateemission(ylvector_1029,sensor_index)
#                    ylprioir_1029[j]=(self.state*self.emission_1029dis).sum()
##                    print('[step]: action 1-for-loop')
#                    if j == (len(self.obs1029) - 1):
#                        print('[step]: action 0-for-loop', i)
#                    if ylprioir_1029[j]==0:
#                        reward_item[i,j]=0
#                    else:
#                        reward_item[i,j]=(self.state[i]*self.emission_1029dis[i])/ylprioir_1029[j] #calculate term P(CO2_t|y_{1:t-1}^{1:N},y_t^l,Tem_{1:t},IR_{1:t}^{1:N})
#            for k in range(len(self.obs1029)):
#                term1=reward_item[:,k]
#                term2=np.log(reward_item[:,k])
#                reward_item1[k]=(term1*term2).sum() # calculate P*logP
#           
#            reward=-(reward_item1*ylprioir_1029).sum()
            """
            tran=self.1029tran
            emission=self.1029emission
            prior=self.1029prior
            """
        if action==2:
#            print('[step]: action 2')
            self.beliefCO2_calibrated=self.CO2_1055dis
            ylprioir_1055=np.zeros(len(self.obs1055),dtype=float)
            sensor_index=2 
            self.new_emission_1055dis=np.zeros([len(self.CO2),len(self.obs1055)],dtype=float)
            for j in range (len(self.obs1055)):
                ylvector_1055=self.emission1055[:,j]
                self.new_emission_1055dis[:,j]=self.calculateemission(ylvector_1055,sensor_index)         
            ylprioir_1055=np.dot(self.state,self.new_emission_1055dis)
            reward_item=np.zeros([len(self.CO2),len(self.obs1055)],dtype=float)
            reward_item1=np.zeros(len(self.obs1055),dtype=float)
            for j in range (len(self.obs1055)):
#                if j == (len(self.obs1022) - 1):
#                    print('[step]: action 0-for-loop', i)
                if ylprioir_1055[j]== 0:
                    reward_item[:,j]=np.zeros(len(self.CO2),dtype=float)
                else:
                    reward_item[:,j]=(self.state*self.new_emission_1055dis[:,j])/ylprioir_1055[j]
                    nonzero_idx=np.nonzero(reward_item[:,j])
                    term1=reward_item[:,j].take(nonzero_idx)
                    term2=np.log(term1)
                    reward_item1[j]=(term1*term2).sum()
            reward=(reward_item1*ylprioir_1055).sum()
#            for i in range(len(self.CO2)):
#                for j in range (len(self.obs1055)):
#                    ylvector_1055=self.emission1055[:,j]
#                    self.calculateemission(ylvector_1055,sensor_index)
#                    ylprioir_1055[j]=(self.state*self.emission_1055dis).sum()
##                    print('[step]: action 2-for-loop')
#                    if j == (len(self.obs1055) - 1):
#                        print('[step]: action 0-for-loop', i)
#                    if ylprioir_1055[j]==0:
#                        reward_item[i,j]=0
#                    else:
#                        reward_item[i,j]=(self.state[i]*self.emission_1055dis[i])/ylprioir_1055[j] #calculate term P(CO2_t|y_{1:t-1}^{1:N},y_t^l,Tem_{1:t},IR_{1:t}^{1:N})
#            for k in range(len(self.obs1055)):
#                term1=reward_item[:,k]
#                term2=np.log(reward_item[:,k])
#                reward_item1[k]=(term1*term2).sum() # calculate P*logP
#           
#            reward=-(reward_item1*ylprioir_1055).sum()
            """
            tran=self.1055tran
            emission=self.1055emission
            prior=self.1055prior
            """
        if action==3:
#            print('[step]: action 3')
            self.beliefCO2_calibrated=self.CO2_1098dis
            sensor_index=3 
            ylprioir_1098=np.zeros(len(self.obs1098),dtype=float)
            self.new_emission_1098dis=np.zeros([len(self.CO2),len(self.obs1098)],dtype=float)
            for j in range (len(self.obs1098)):
                ylvector_1098=self.emission1098[:,j]
                self.new_emission_1098dis[:,j]=self.calculateemission(ylvector_1098,sensor_index)         
            ylprioir_1098=np.dot(self.state,self.new_emission_1098dis)
            reward_item=np.zeros([len(self.CO2),len(self.obs1098)],dtype=float)
            reward_item1=np.zeros(len(self.obs1098),dtype=float)
            for j in range (len(self.obs1098)):
#                if j == (len(self.obs1022) - 1):
#                    print('[step]: action 0-for-loop', i)
                if ylprioir_1098[j]== 0:
                    reward_item[:,j]=np.zeros(len(self.CO2),dtype=float)
                else:
                    reward_item[:,j]=(self.state*self.new_emission_1098dis[:,j])/ylprioir_1098[j]
                    nonzero_idx=np.nonzero(reward_item[:,j])
                    term1=reward_item[:,j].take(nonzero_idx)
                    term2=np.log(term1)
                    reward_item1[j]=(term1*term2).sum()
            reward=(reward_item1*ylprioir_1098).sum()
#            for i in range(len(self.CO2)):
#                for j in range (len(self.obs1098)):
#                    ylvector_1098=self.emission1098[:,j]
#                    self.calculateemission(ylvector_1098,sensor_index)
#                    ylprioir_1098[j]=(self.state*self.emission_1098dis).sum()
##                    print('[step]: action 3-for-loop')
#                    if j == (len(self.obs1098) - 1):
#                        print('[step]: action 0-for-loop', i)
#                    if ylprioir_1098[j]==0:
#                        reward_item[i,j]=0
#                    else:
#                        reward_item[i,j]=(self.state[i]*self.emission_1098dis[i])/ylprioir_1098[j] #calculate term P(CO2_t|y_{1:t-1}^{1:N},y_t^l,Tem_{1:t},IR_{1:t}^{1:N})
#            for k in range(len(self.obs1098)):
#                term1=reward_item[:,k]
#                term2=np.log(reward_item[:,k])
#                reward_item1[k]=(term1*term2).sum() # calculate P*logP
#           
#            reward=-(reward_item1*ylprioir_1098).sum()
            """
            tran=self.1098tran
            emission=self.1098emission
            prior=self.1098prior
            """
        if action==4:
#            print('[step]: action 4')
            self.beliefCO2_calibrated=self.CO2_1187dis
            sensor_index=4 
            ylprioir_1187=np.zeros(len(self.obs1187),dtype=float)
            self.new_emission_1187dis=np.zeros([len(self.CO2),len(self.obs1187)],dtype=float)
            for j in range (len(self.obs1187)):
                ylvector_1187=self.emission1187[:,j]
                self.new_emission_1187dis[:,j]=self.calculateemission(ylvector_1187,sensor_index)         
            ylprioir_1187=np.dot(self.state,self.new_emission_1187dis)
            reward_item=np.zeros([len(self.CO2),len(self.obs1187)],dtype=float)
            reward_item1=np.zeros(len(self.obs1187),dtype=float)
            for j in range (len(self.obs1187)):
#                if j == (len(self.obs1022) - 1):
#                    print('[step]: action 0-for-loop', i)
                if ylprioir_1187[j]== 0:
                    reward_item[:,j]=np.zeros(len(self.CO2),dtype=float)
                else:
                    reward_item[:,j]=(self.state*self.new_emission_1187dis[:,j])/ylprioir_1187[j]
                    nonzero_idx=np.nonzero(reward_item[:,j])
                    term1=reward_item[:,j].take(nonzero_idx)
                    term2=np.log(term1)
                    reward_item1[j]=(term1*term2).sum()
            reward=(reward_item1*ylprioir_1187).sum()
#            for i in range(len(self.CO2)):
#                for j in range (len(self.obs1187)):
#                    ylvector_1187=self.emission1187[:,j]
#                    self.calculateemission(ylvector_1187,sensor_index)
#                    ylprioir_1187[j]=(self.state*self.emission_1187dis).sum()
##                    print('[step]: action 4-for-loop')
#                    if j == (len(self.obs1187) - 1):
#                        print('[step]: action 0-for-loop', i)
#                    if ylprioir_1187[j]==0:
#                        reward_item[i,j]=0
#                    else:
#                        reward_item[i,j]=(self.state[i]*self.emission_1187dis[i])/ylprioir_1187[j] #calculate term P(CO2_t|y_{1:t-1}^{1:N},y_t^l,Tem_{1:t},IR_{1:t}^{1:N})
#            for k in range(len(self.obs1187)):
#                term1=reward_item[:,k]
#                term2=np.log(reward_item[:,k])
#                reward_item1[k]=(term1*term2).sum() # calculate P*logP
#           
#            reward=-(reward_item1*ylprioir_1187).sum()
            """
            tran=self.1187tran
            emission=self.1187emission
            prior=self.1187prior
            """
#        print('[step]:imdd')
#        toc = time.time()
#        print('time duration: ', toc-tic)
        self.ReverseAbs()
        self.Calculatezero() # calculate from calibrated belief of CO2 back to belief of zero for each sensor 
        
        self.tem1=random.choice(self.tem) # generate temperature at time t
        self.tem1=round(self.tem1,1)
        self.temvector.append(self.tem1)
        delta_tem = round(self.temvector[-1]-self.temvector[-2],1) # calculate temperature difference
        
        
        flag = True
        idx = 0
        while flag:
            if self.tem1>=-5+idx*1. and self.tem1<-5+(idx+1)*1:
                flag = False
            elif idx <= 28:
                if self.tem1==-5+(idx+1)*1:
                     flag = False
                idx += 1
#           else:
#               print('error')   
#               exit(0)
                
 
        
        IR1022_duration=(43542-38431+1)/29
        IR1022_start = 38431+(28-idx)*IR1022_duration
        IR1022_end = 38431+(28-idx+1)*IR1022_duration
        self.IR1022s = random.randint(math.floor(IR1022_start),math.floor(IR1022_end))
        
        IR1029_duration=(41855-38024+1)/29
        IR1029_start = 38024+(28-idx)*IR1029_duration
        IR1029_end = 38024+(28-idx+1)*IR1029_duration
        self.IR1029s = random.randint(math.floor(IR1029_start),math.floor(IR1029_end))
        
        IR1055_duration=(43052-38162+1)/29
        IR1055_start = 38162+(28-idx)*IR1055_duration
        IR1055_end = 38162+(28-idx+1)*IR1055_duration
        self.IR1055s = random.randint(math.floor(IR1055_start),math.floor(IR1055_end))
        
        IR1098_duration=(43904-38035+1)/29
        IR1098_start = 38035 + (28-idx)*IR1098_duration
        IR1098_end = 38035 + (28-idx+1)*IR1098_duration
        self.IR1098s = random.randint(math.floor(IR1098_start),math.floor(IR1098_end))
        
        IR1187_duration=(43774-38129+1)/29
        IR1187_start = 38129 + (28-idx)*IR1187_duration
        IR1187_end = 38129 + (28-idx+1)*IR1187_duration
        self.IR1187s = random.randint(math.floor(IR1187_start),math.floor(IR1187_end))
        

        
        if delta_tem<=-0.5:  # pick corresponding transition matrix according to temperature change
            tran_1022=self.tran1022[0:1241,:]
            tran_1029=self.tran1029[0:1241,:]
            tran_1055=self.tran1055[0:1241,:] ##########################
            tran_1098=self.tran1098[0:1241,:]
            tran_1187=self.tran1187[0:1241,:]
        elif delta_tem>=0.5:
            tran_1022=self.tran1022[2482:3723,:]
            tran_1029=self.tran1029[2482:3723,:]
            tran_1055=self.tran1055[2482:3723,:]
            tran_1098=self.tran1098[2482:3723,:]
            tran_1187=self.tran1187[2482:3723,:]
        else:
            tran_1022=self.tran1022[1241:2482,:]
            tran_1029=self.tran1029[1241:2482,:]
            tran_1055=self.tran1055[1241:2482,:]
            tran_1098=self.tran1098[1241:2482,:]
            tran_1187=self.tran1187[1241:2482,:]
               
        belief_1022=self.belief1
        belief_1029=self.belief2
        belief_1055=self.belief3
        belief_1098=self.belief4
        belief_1187=self.belief5
        
        
        y1022=random.choice(self.obs1022) # generate observation at time t for each sensor
        y1029=random.choice(self.obs1029)
        y1055=random.choice(self.obs1055)
        y1098=random.choice(self.obs1098)
        y1187=random.choice(self.obs1187)
        
        position_obs1022=y1022-self.obs1022[0] # find the index of the generated observation 
        position_obs1029=y1029-self.obs1029[0]
        position_obs1055=y1055-self.obs1055[0]
        position_obs1098=y1098-self.obs1098[0]
        position_obs1187=y1187-self.obs1187[0]        
        
        yvector_1022=self.emission1022[:,position_obs1022] # pick the corresponding column in emission matrix P(y|zero) for all zero
        yvector_1029=self.emission1029[:,position_obs1029]
        yvector_1055=self.emission1055[:,position_obs1055]
        yvector_1098=self.emission1098[:,position_obs1098]
        yvector_1187=self.emission1187[:,position_obs1187]
        
#        yvector_1022 = np.array(yvector_1022)
#        yvector_1029 = np.array(yvector_1029)
#        yvector_1055 = np.array(yvector_1055)
#        yvector_1098 = np.array(yvector_1098)
#        yvector_1187 = np.array(yvector_1187)
        
        yprioir_1022=(yvector_1022*self.belief1).sum() # calculate marginal probability of the given observation
        yprioir_1029=(yvector_1029*self.belief2).sum()
        yprioir_1055=(yvector_1055*self.belief3).sum()
        yprioir_1098=(yvector_1098*self.belief4).sum()
        yprioir_1187=(yvector_1187*self.belief5).sum()
        
        step1_1022=np.zeros(1241,dtype=float) # P(zero_t|y_{1:t},Tem_{1:t})
        step1_1029=np.zeros(1241,dtype=float)
        step1_1055=np.zeros(1241,dtype=float)
        step1_1098=np.zeros(1241,dtype=float)
        step1_1187=np.zeros(1241,dtype=float)
        
        step2_1022=np.zeros(1241,dtype=float) # P(zero_t+1|y_{1:t},Tem_{1:t+1})
        step2_1029=np.zeros(1241,dtype=float)
        step2_1055=np.zeros(1241,dtype=float)
        step2_1098=np.zeros(1241,dtype=float)
        step2_1187=np.zeros(1241,dtype=float)
        
#        length_step1_1022=len(step1_1022)
#        length_step1_1029=len(step1_1029)
#        length_step1_1055=len(step1_1055)
#        length_step1_1098=len(step1_1098)
#        length_step1_1187=len(step1_1187)
        
# Evolution of the zero belief of individual sensor
        
#改成向量形式
        if yprioir_1022==0:
            self.belief1=belief_1022
        else:
            step1_1022=belief_1022*yvector_1022/yprioir_1022
            step2_1022=np.dot(step1_1022,tran_1022)
            if np.all(step2_1022==0):
                self.belief1=belief_1022
            else:
                self.belief1=step2_1022
#            for i in range(length_step1_1022):
#                step1_1022[i]=(belief_1022[i]*yvector_1022[i])/yprioir_1022
#            step1_1022=np.array(step1_1022)
#            for j in range(length_step1_1022):
#                for k in range(length_step1_1022):
#                    temp_variable=(tran_1022[k,:]*step1_1022).sum()
#                step2_1022[j]=temp_variable
#            step2_1022=np.array(step2_1022)    
          
            
        if yprioir_1029==0:
            self.belief2=belief_1029
        else:
            step1_1029=belief_1029*yvector_1029/yprioir_1029
            step2_1029=np.dot(step1_1029,tran_1029)
            if np.all(step2_1029==0):
                self.belief2=belief_1029
            else:
                self.belief2=step2_1029
#            for i in range(length_step1_1029):
#                step1_1029[i]=(belief_1029[i]*yvector_1029[i])/yprioir_1029
#            for j in range(length_step1_1029):
#                for k in range(length_step1_1029):
#                    temp_variable=(tran_1029[k,:]*step1_1029).sum()
#                step2_1029[j]=temp_variable
#            self.belief2=step2_1029
            
            
        if yprioir_1055==0:
            self.belief3=belief_1055
        else:
            step1_1055=belief_1055*yvector_1055/yprioir_1055
            step2_1055=np.dot(step1_1055,tran_1055)
            if np.all(step2_1055==0):
                self.belief3=belief_1055
            else:
                self.belief3=step2_1055
#            for i in range(length_step1_1055):
#                step1_1055[i]=(belief_1055[i]*yvector_1055[i])/yprioir_1055
#            for j in range(length_step1_1055):
#                for k in range(length_step1_1055):
#                    temp_variable=(tran_1055[k,:]*step1_1055).sum()
#                step2_1055[j]=temp_variable
#            self.belief3=step2_1055
        
        if yprioir_1098==0:
            self.belief4=belief_1098
        else:
            step1_1098=belief_1098*yvector_1098/yprioir_1098
            step2_1098=np.dot(step1_1098,tran_1098)
            if np.all(step2_1098==0):
                self.belief4=belief_1098
            else:
                self.belief4=step2_1098
#            for i in range(length_step1_1098):
#                step1_1098[i]=(belief_1098[i]*yvector_1022[i])/yprioir_1098
#            for j in range(length_step1_1098):
#                for k in range(length_step1_1098):
#                    temp_variable=(tran_1098[k,:]*step1_1098).sum()
#                step2_1098[j]=temp_variable
#            self.belief4=step2_1098
                
        if yprioir_1187==0:
            self.belief5=belief_1187
        else:
            step1_1187=belief_1187*yvector_1187/yprioir_1187
            step2_1187=np.dot(step1_1187,tran_1187)
            if np.all(step2_1187==0):
                self.belief5=belief_1187
            else:
                self.belief5=step2_1187
            
# transfer the belief of zero to belief of CO2    
        self.calculateAbs()
        self.calculateCO2()
# get the fused belief of CO2        
        self.state = self.combinebelief()
    

        return np.array(self.state), reward
    
   
    def calculateemission(self,y,idx):
        ## calculate P(y_t|CO2_t) from P(y_t|zero_t)
#        start=time.time()
        if idx==0:
#            temp_1022_e = 61440 - self.Abs_1022
#            CO2_1022_e = Linterp(self.tablex, self.table1022, temp_1022_e)
#            newCO2_1022_e = self.calculaterange(CO2_1022_e) 
#            list_1022_e=sorted(list_duplicates(newCO2_1022_e)) # Get the CO2 value,  quantize the default CO2 range, and find the duplicated elements
#            newCO2_1022_e=self.newCO2_1022_int
            list_1022_e=self.list_1022
            emission_dis=self.calculatedis1(list_1022_e,y) # Add the duplicated elements together and get a new distribution
        
        if idx==1:    
#            temp_1029_e = 61440 - self.Abs_1029
#            CO2_1029_e = Linterp(self.tablex, self.table1029, temp_1029_e)
#            newCO2_1029_e = self.calculaterange(CO2_1029_e)
#            list_1029_e=sorted(list_duplicates(newCO2_1029_e))
            list_1029_e=self.list_1029
            emission_dis=self.calculatedis1(list_1029_e,y)
            
        if idx==2:    
#            temp_1055_e = 61440 - self.Abs_1055
#            CO2_1055_e = Linterp(self.tablex, self.table1055, temp_1055_e)
#            newCO2_1055_e = self.calculaterange(CO2_1055_e)
#            list_1055_e=sorted(list_duplicates(newCO2_1055_e))
            list_1055_e=self.list_1055
            emission_dis=self.calculatedis1(list_1055_e,y)
        
        if idx==3:
#            temp_1098_e = 61440 - self.Abs_1098
#            CO2_1098_e = Linterp(self.tablex, self.table1098, temp_1098_e)
#            newCO2_1098_e = self.calculaterange(CO2_1098_e)
#            list_1098_e=sorted(list_duplicates(newCO2_1098_e))
            list_1098_e=self.list_1098
            emission_dis=self.calculatedis1(list_1098_e,y)
        
        
        if idx==4:
#            temp_1187_e = 61440 - self.Abs_1187
#            CO2_1187_e = Linterp(self.tablex, self.table1187, temp_1187_e)
#            newCO2_1187_e = self.calculaterange(CO2_1187_e)
#            list_1187_e=sorted(list_duplicates(newCO2_1187_e))
            list_1187_e=self.list_1187
            emission_dis=self.calculatedis1(list_1187_e,y)
#        over=time.time()
#        print('[emi]: dutation: ', over-start)
        return emission_dis
    
    def calculatedis1(self,l,y):
#        star_line = time.time()
        length=len(l)
         
        distribution_new=np.zeros(801,dtype=float)
            
        for idx1 in range(length):
                Probability_length=len(l[idx1][1])
                CO2_value =  l[idx1][0]
                Probability_sum=0.
                for idx2 in range(Probability_length):
                    zero_index = l[idx1][1][idx2]
                    Probability_sum += y[zero_index]
                idx3=int(CO2_value-self.CO2[0])
                distribution_new[idx3]=Probability_sum
#        end_line = time.time()
#        print('[dis]: dutation: ', end_line-star_line)        
        return distribution_new
  
    
    def ReverseAbs(self):
         

        #   print("Table {} is: {} ".format(TableID, Table))

         

        reverse_temp1 = Linterp(self.table1022, self.tablex, self.CO2)
        reverse_temp2 = Linterp(self.table1029, self.tablex, self.CO2)
        reverse_temp3 = Linterp(self.table1055, self.tablex, self.CO2)
        reverse_temp4 = Linterp(self.table1098, self.tablex, self.CO2)
        reverse_temp5 = Linterp(self.table1187, self.tablex, self.CO2)
        
        self.reverse_Abs1022 = 61440 - reverse_temp1
        self.reverse_Abs1029 = 61440 - reverse_temp2
        self.reverse_Abs1055 = 61440 - reverse_temp3
        self.reverse_Abs1098 = 61440 - reverse_temp4
        self.reverse_Abs1187 = 61440 - reverse_temp5  
        
     
        
        #   plot_Tablex_Table(Tablex, Table, CO2, temp, Abs)
  

    def Calculatezero(self):
        """
          calculate zero from abs and generated belief of zero
        """
        DT=(self.tem1- 25)*100
     
        reverseterm1_1022 = self.IR1022s*(1 + DT*self.tz1022/65536/256)
        reverseterm1_1029 = self.IR1029s*(1 + DT*self.tz1029/65536/256)
        reverseterm1_1055 = self.IR1055s*(1 + DT*self.tz1055/65536/256)
        reverseterm1_1098 = self.IR1098s*(1 + DT*self.tz1098/65536/256)
        reverseterm1_1187 = self.IR1187s*(1 + DT*self.tz1187/65536/256)
        
       
        reverseterm2_1022 = (1 + DT*self.ts1022/65536/256)*(self.span1022/4096) #np
        reverseterm2_1029 = (1 + DT*self.ts1029/65536/256)*(self.span1029/4096)
        reverseterm2_1055 = (1 + DT*self.ts1055/65536/256)*(self.span1055/4096)
        reverseterm2_1098 = (1 + DT*self.ts1098/65536/256)*(self.span1098/4096)
        reverseterm2_1187 = (1 + DT*self.ts1187/65536/256)*(self.span1187/4096)
        
        zero_1022 = ((61440 - self.reverse_Abs1022/reverseterm2_1022)*8192)/reverseterm1_1022
        zero_1029 = ((61440 - self.reverse_Abs1029/reverseterm2_1029)*8192)/reverseterm1_1029
        zero_1055 = ((61440 - self.reverse_Abs1055/reverseterm2_1055)*8192)/reverseterm1_1055
        zero_1098 = ((61440 - self.reverse_Abs1098/reverseterm2_1098)*8192)/reverseterm1_1098
        zero_1187 = ((61440 - self.reverse_Abs1187/reverseterm2_1187)*8192)/reverseterm1_1187
        
        newzero_1022_float = self.calculatezerorange(zero_1022)
        newzero_1029_float = self.calculatezerorange(zero_1029)
        newzero_1055_float = self.calculatezerorange(zero_1055)
        newzero_1098_float = self.calculatezerorange(zero_1098)
        newzero_1187_float = self.calculatezerorange(zero_1187)
        
        newzero_1022_int = self.getround(newzero_1022_float)
        newzero_1029_int = self.getround(newzero_1029_float)
        newzero_1055_int = self.getround(newzero_1055_float)
        newzero_1098_int = self.getround(newzero_1098_float)
        newzero_1187_int = self.getround(newzero_1187_float)
        
#        zero_1022 = round(zero_1022)
#        zero_1029 = round(zero_1029)
#        zero_1055 = round(zero_1055)
#        zero_1098 = round(zero_1098)
#        zero_1187 = round(zero_1187)
#        
#        newzero_1022 = self.calculatezerorange(zero_1022)
#        newzero_1029 = self.calculatezerorange(zero_1029)
#        newzero_1055 = self.calculatezerorange(zero_1055)
#        newzero_1098 = self.calculatezerorange(zero_1098)
#        newzero_1187 = self.calculatezerorange(zero_1187)
        
        listzero_1022=sorted(list_duplicates(newzero_1022_int))
        listzero_1029=sorted(list_duplicates(newzero_1029_int))
        listzero_1055=sorted(list_duplicates(newzero_1055_int))
        listzero_1098=sorted(list_duplicates(newzero_1098_int))
        listzero_1187=sorted(list_duplicates(newzero_1187_int))
        
        belief_CO2=self.beliefCO2_calibrated
        
        self.belief1=self.calculatezerodis(listzero_1022,belief_CO2)
        self.belief2=self.calculatezerodis(listzero_1029,belief_CO2)
        self.belief3=self.calculatezerodis(listzero_1055,belief_CO2)
        self.belief4=self.calculatezerodis(listzero_1098,belief_CO2)
        self.belief5=self.calculatezerodis(listzero_1187,belief_CO2)


    def calculatezerodis(self,l,b):
         
        length=len(l)
         
        distributionzero_new=np.zeros(1241,dtype=float)
            
        for idx1 in range(length):
                Probability_length=len(l[idx1][1])
                zero_value =  l[idx1][0]
                Probability_sum=0.
                for idx2 in range(Probability_length):
                    CO2_index = l[idx1][1][idx2]
                    Probability_sum += b[CO2_index]
                idx3=int(zero_value-self.zero[0])
                distributionzero_new[idx3]=Probability_sum
                
        return distributionzero_new
                
    def calculatezerorange(self,c): 
        length=len(c)
        for i in range(length):
            if c[i]>self.zero[-1]:
                c[i] = self.zero[-1]
            elif c[i]<self.zero[0]:
                c[i] = self.zero[0]
        return c
    
    def reset(self):
        
        self.temvector=[]
        self.tem1=random.choice(self.tem)
        self.temvector.append(self.tem1)
        
        flag = True
        idx = 0
        while flag:
            if self.tem1>=-5+idx*1. and self.tem1<-5+(idx+1)*1:
                flag = False
            elif idx <= 28:
                if self.tem1==-5+(idx+1)*1:
                     flag = False
                idx += 1
#           else:
#               print('error')   
#               exit(0)
                
 
        
        IR1022_duration=(43542-38431+1)/29
        IR1022_start = 38431+(28-idx)*IR1022_duration
        IR1022_end = 38431+(28-idx+1)*IR1022_duration
        self.IR1022s = random.randint(math.floor(IR1022_start),math.floor(IR1022_end))
        
        IR1029_duration=(41855-38024+1)/29
        IR1029_start = 38024+(28-idx)*IR1029_duration
        IR1029_end = 38024+(28-idx+1)*IR1029_duration
        self.IR1029s = random.randint(math.floor(IR1029_start),math.floor(IR1029_end))
        
        IR1055_duration=(43052-38162+1)/29
        IR1055_start = 38162+(28-idx)*IR1055_duration
        IR1055_end = 38162+(28-idx+1)*IR1055_duration
        self.IR1055s = random.randint(math.floor(IR1055_start),math.floor(IR1055_end))
        
        IR1098_duration=(43904-38035+1)/29
        IR1098_start = 38035 + (28-idx)*IR1098_duration
        IR1098_end = 38035 + (28-idx+1)*IR1098_duration
        self.IR1098s = random.randint(math.floor(IR1098_start),math.floor(IR1098_end))
        
        IR1187_duration=(43774-38129+1)/29
        IR1187_start = 38129 + (28-idx)*IR1187_duration
        IR1187_end = 38129 + (28-idx+1)*IR1187_duration
        self.IR1187s = random.randint(math.floor(IR1187_start),math.floor(IR1187_end))
        
        self.belief1=self.prior1022
        self.belief2=self.prior1029
        self.belief3=self.prior1055
        self.belief4=self.prior1098
        self.belief5=self.prior1187
          
        self.belief1=np.array(self.belief1)
        self.belief2=np.array(self.belief2)
        self.belief3=np.array(self.belief3)
        self.belief4=np.array(self.belief4)
        self.belief5=np.array(self.belief5)
    
        
        self.calculateAbs()
        self.calculateCO2()
        
        self.state = self.combinebelief()
        self.steps_beyond_done = None
        
    
    def calculateAbs(self):
        """
      Calculate Abs given zero coefficients, IR, DT and other coefficients
      Inputs:
        Zero: an array of real values, shape(n,)
        IR: an array of real values, shape(n,)
        DT: an array of real value, shape(n,)
        coefficients: the coefficients of a sensor [Span, TZ, TZ2, TS, TS2]
      Return:
        Abs: an array of real values, shape(n,)
        """
        
        DT=(self.tem1- 25)*100 
       

        term1_1022 = (1 + DT*self.tz1022/65536/256)*(self.zero/8192) #转成np
        term2_1022 = (1 + DT*self.ts1022/65536/256)*(self.span1022/4096)
        term1_1029 = (1 + DT*self.tz1029/65536/256)*(self.zero/8192)
        term2_1029 = (1 + DT*self.ts1029/65536/256)*(self.span1029/4096)
        term1_1055 = (1 + DT*self.tz1055/65536/256)*(self.zero/8192)
        term2_1055 = (1 + DT*self.ts1055/65536/256)*(self.span1055/4096)
        term1_1098 = (1 + DT*self.tz1098/65536/256)*(self.zero/8192)
        term2_1098 = (1 + DT*self.ts1098/65536/256)*(self.span1098/4096)
        term1_1187 = (1 + DT*self.tz1187/65536/256)*(self.zero/8192)
        term2_1187 = (1 + DT*self.ts1187/65536/256)*(self.span1187/4096)

      # Abs = [61440 - IR*term1]*[term2]
        self.Abs_1022 = (61440 - self.IR1022s*term1_1022)*term2_1022
        self.Abs_1029 = (61440 - self.IR1029s*term1_1029)*term2_1029
        self.Abs_1055 = (61440 - self.IR1055s*term1_1055)*term2_1055
        self.Abs_1098 = (61440 - self.IR1098s*term1_1098)*term2_1098
        self.Abs_1187 = (61440 - self.IR1187s*term1_1187)*term2_1187


    def calculateCO2(self):
        
        temp_1022 = 61440 - self.Abs_1022
        temp_1029 = 61440 - self.Abs_1029
        temp_1055 = 61440 - self.Abs_1055
        temp_1098 = 61440 - self.Abs_1098
        temp_1187 = 61440 - self.Abs_1187

        CO2_1022 = Linterp(self.tablex, self.table1022, temp_1022)
        CO2_1029 = Linterp(self.tablex, self.table1029, temp_1029)
        CO2_1055 = Linterp(self.tablex, self.table1055, temp_1055)
        CO2_1098 = Linterp(self.tablex, self.table1098, temp_1098)
        CO2_1187 = Linterp(self.tablex, self.table1187, temp_1187)



        newCO2_1022_float = self.calculaterange(CO2_1022)
        newCO2_1029_float = self.calculaterange(CO2_1029)
        newCO2_1055_float = self.calculaterange(CO2_1055)
        newCO2_1098_float = self.calculaterange(CO2_1098)
        newCO2_1187_float = self.calculaterange(CO2_1187)
        
        newCO2_1022_int = self.getround(newCO2_1022_float)
        newCO2_1029_int = self.getround(newCO2_1029_float)
        newCO2_1055_int = self.getround(newCO2_1055_float)
        newCO2_1098_int = self.getround(newCO2_1098_float)
        newCO2_1187_int = self.getround(newCO2_1187_float)
        
        self.list_1022=sorted(list_duplicates(newCO2_1022_int))
        self.list_1029=sorted(list_duplicates(newCO2_1029_int))
        self.list_1055=sorted(list_duplicates(newCO2_1055_int))
        self.list_1098=sorted(list_duplicates(newCO2_1098_int))
        self.list_1187=sorted(list_duplicates(newCO2_1187_int))

        self.CO2_1022dis=self.calculatedis(self.list_1022,self.belief1)
        self.CO2_1029dis=self.calculatedis(self.list_1029,self.belief2)
        self.CO2_1055dis=self.calculatedis(self.list_1055,self.belief3)
        self.CO2_1098dis=self.calculatedis(self.list_1098,self.belief4)
        self.CO2_1187dis=self.calculatedis(self.list_1187,self.belief5)

    def getround(self,a):
        length=len(a)
        b=np.zeros(length,dtype=int)
        for idx in range(length):
            b[idx]=round(a[idx])
        return b
            
            
    def calculatedis(self,l,b):
         
        length=len(l)
         
        distribution_new=np.zeros(801,dtype=float)
            
        for idx1 in range(length):
            Probability_length=len(l[idx1][1])
            CO2_value =  l[idx1][0]
            Probability_sum=0.
            for idx2 in range(Probability_length):
                zero_index = l[idx1][1][idx2]
                Probability_sum += b[zero_index]
            idx3=int(CO2_value-self.CO2[0])
            distribution_new[idx3]=Probability_sum
                
        return distribution_new
                
                
    def calculaterange(self,c): 
#        star_line=time.time()
        length=len(c)
        for i in range(length):
            if c[i]>self.CO2[-1]:
                c[i] = self.CO2[-1]
            elif c[i]<self.CO2[0]:
                c[i] = self.CO2[0]
#        end_line = time.time()
#        print('[range]: dutation: ', end_line-star_line)
        return c


    def combinebelief(self):
 
        distance_12=wasserstein_distance(self.CO2, self.CO2, self.CO2_1022dis,self.CO2_1029dis)
        distance_13=wasserstein_distance(self.CO2, self.CO2, self.CO2_1022dis,self.CO2_1055dis)
        distance_14=wasserstein_distance(self.CO2, self.CO2, self.CO2_1022dis,self.CO2_1098dis)
        distance_15=wasserstein_distance(self.CO2, self.CO2, self.CO2_1022dis,self.CO2_1187dis)
        distance_23=wasserstein_distance(self.CO2, self.CO2, self.CO2_1029dis,self.CO2_1055dis)
        distance_24=wasserstein_distance(self.CO2, self.CO2, self.CO2_1029dis,self.CO2_1098dis)
        distance_25=wasserstein_distance(self.CO2, self.CO2, self.CO2_1029dis,self.CO2_1187dis)
        distance_34=wasserstein_distance(self.CO2, self.CO2, self.CO2_1055dis,self.CO2_1098dis)
        distance_35=wasserstein_distance(self.CO2, self.CO2, self.CO2_1055dis,self.CO2_1187dis)
        distance_45=wasserstein_distance(self.CO2, self.CO2, self.CO2_1098dis,self.CO2_1187dis)
        d_sum=distance_12+distance_13+distance_14+distance_15+distance_23+distance_24+distance_25+distance_34+distance_35+distance_45

        d12_norm=distance_12/(d_sum/2.)
        d13_norm=distance_13/(d_sum/2.)
        d14_norm=distance_14/(d_sum/2.)
        d15_norm=distance_15/(d_sum/2.)
        d23_norm=distance_23/(d_sum/2.)
        d24_norm=distance_24/(d_sum/2.)
        d25_norm=distance_25/(d_sum/2.)
        d34_norm=distance_34/(d_sum/2.)
        d35_norm=distance_35/(d_sum/2.)
        d45_norm=distance_45/(d_sum/2.)

        s12=1-d12_norm
        s13=1-d13_norm
        s14=1-d14_norm
        s15=1-d15_norm
        s23=1-d23_norm
        s24=1-d24_norm
        s25=1-d25_norm
        s34=1-d34_norm
        s35=1-d35_norm
        s45=1-d45_norm

        supp1=s12+s13+s14+s15
        supp2=s12+s23+s24+s25
        supp3=s13+s23+s34+s35
        supp4=s14+s23+s34+s45
        supp5=s15+s25+s35+s45

        supp_sum=supp1+supp2+supp3+supp4+supp5

        cred1=supp1/supp_sum
        cred2=supp2/supp_sum
        cred3=supp3/supp_sum
        cred4=supp4/supp_sum
        cred5=supp5/supp_sum

        belief_comb=cred1*self.CO2_1022dis+cred2*self.CO2_1029dis+cred3*self.CO2_1055dis+cred4*self.CO2_1098dis+cred5*self.CO2_1187dis
        temp=belief_comb*belief_comb*belief_comb*belief_comb*belief_comb
        nfactor=temp.sum()
        belief_CO2=temp/nfactor

        return np.array(belief_CO2)


    
    
def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs) >= 1)


def SearchIndex(a, y):
    a_ascend = a[np.argsort(a)] # change into ascending order
    n = a.shape[0]
    m = y.shape[0]
    lo_idx = np.zeros((m,), dtype=int)

    if a[0] > a[-1]:
        for i in range(m):
            target = y[i]
          # binary selection
            lo, hi = 0, n-1
            while lo < hi:
                mid = (lo + hi) // 2
                if a_ascend[mid] > target:
                     hi = mid
                else: lo = mid + 1

            lo_idx[i] = int(n - 1 - lo)
    else:
        for i in range(m):
            target = y[i]
          # binary selection
            lo, hi = 0, n-1
            while lo < hi:
                mid = (lo + hi) // 2
                if a_ascend[mid] > target:
                    hi = mid
                else: lo = mid + 1

            lo_idx[i] = int(lo - 1)

    return lo_idx


def Linterp(Table, Tablex, y):
#    star_line = time.time()
    lo_idx = SearchIndex(Table, y)

    y_lo = Table[lo_idx]
    y_hi = Table[lo_idx+1]
    x_lo = Tablex[lo_idx]
    x_hi = Tablex[lo_idx+1]
    slope = (x_hi - x_lo) / (y_hi - y_lo)[:]
    x_new = slope * (y - y_lo)[:] + x_lo
#    end_line = time.time()
#    print('[Linterp]: dutation: ', end_line-star_line)
    return np.array(x_new)

    #### def close(self):
      #      if self.viewer:
       #         self.viewer.close()
       #         self.viewer = None

class DQN():

    def __init__(self, no_states , no_action,memory_size=100000,model_layers=[64],batch_size=64,gamma=0.99,update_freq=1000):
        '''
        :param no_states: The number of states or observation space. This is simply the number of differnet observations and is
                          the size of the input layer of the neural network
        :param no_action: The number of possible actions the agent can take. The agent does not know the actual value of the action
                        but will return the index for that value.
        :param memory_size: The memory size of action replay. The agent stores previous state,rewards,action, current state and done
                            in the memory and randomly samples it
        :param model_layers: A list of int containing the number of neurons in each hidden layer starting from 1st to last. The Finished
                            network will have 2 more layers for input and ouput.
        :param batch_size: integer Batch size for neural network
        :param gamma: float[0,1]Discount factor for the agent
        '''
        self.states  = no_states
        self.actions = no_action
        self.size = memory_size
        self.mem = []
        self.batch_size=batch_size
        self.update_freq=update_freq
        self.gamma=gamma
        self.rlmod = self.create_model(model_layers)
        self.rlmod.compile(optimizer=Adam(lr=0.001),loss='mse',metrics=['mae','mse'])
        self.rlmod.save('target.h5')
        self.tamod=load_model('target.h5')
        self.rlmod.summary()


    def create_model(self,model_layers):
        '''
        :param model_layers: List of integers containing neurons in each hidden layer starting from 1st
        :return: Keras model object
        '''
        model=Sequential()
        model.add(Dense(units=model_layers[0], activation='relu', input_dim=self.states))
        if len(model_layers)>1:
            for i in model_layers[1:]:
                model.add(Dense(units=i, activation='relu'))

        model.add(Dense(self.actions, activation='linear'))
        return model


    def push(self, val):
        '''tuple of state,reward,action,,next state,done is pushed onto the memory'''
        self.mem.append(val)
        if len(self.mem) > self.size:
            del self.mem[0]

    def sample(self,batch_size):
        '''
        function to randomly sample state,reward,action,next state,done from memory.
        :param batch_size: integer Number of samples to be taken at once
        :return: (State,reward,action,next state done)
        '''
        sample = zip(*random.sample(self.mem,batch_size))
        return sample

    def learn(self,count):
        '''
        Function for training the neural network. The function will update the weights of the newtork and does not return anyting
        '''
        obs, r, a, next_obs = self.sample(self.batch_size) #random samples from memory
        pred_target = self.rlmod.predict(np.array(obs).reshape(self.batch_size, self.states)) #predicted q-values
        next_op1 = self.tamod.predict(np.array(next_obs).reshape(self.batch_size, self.states)) # actual q-values
        for i in range(self.batch_size):
            k = a[i]  # k is the index of action of the action taken
#            if done[i] == False:
            target = r[i] + self.gamma * np.amax(next_op1[i][:]) #For non terminal states
#            else:
#                target = r[i] #For terminal states
            pred_target[i][k] = target
        self.rlmod.fit(np.array(obs).reshape(self.batch_size, self.states), pred_target.reshape(self.batch_size, self.actions), epochs=1, batch_size=1, verbose=0) #training the network to approximate q-values
        if count % self.update_freq == 0:
            self.rlmod.save('target.h5')
            self.tamod=load_model('target.h5')

    def action_select(self,x, eph):
        '''
        A function to select an action based on the Epsilon greedy policy. Epislon percent of times the agent will select a random
        action while 1-Epsilon percent of the time the agent will select the action with the highest Q value as predicted by the
        neural network.
        :param x: list of shape [nuber of observation,]
        :param eph: Float [0,1) value of epsilon
        :return: Integer indicating the index of actual action
        '''
        q_out = self.rlmod.predict_on_batch(np.array(x).reshape(1, self.states))
        action = np.argmax(q_out)
        act = np.array([0, 1, action])
        return np.random.choice(act, p=[(eph / 2), (eph / 2), (1 - eph)])

#Test the agent on Cartpole-v0 problem on openAI

env = sensorbelief()


# get the number of observation and number of actions
actions = env.action_space.shape[0]
states = env.observation_space.shape[0]


#create an instance of the DQN agent
agent = DQN(states, actions)


total_episodes = 2000
eph = 0.9 #ephsilon
eph_min = 0.0001
decay = 0.995 #decay for ephsilon
max_time_steps = 10

count =0
epsisode_rewards=[]

for episodes in range(total_episodes):
    print('Episodes',episodes)
    star_line = time.time()
    env.reset()
    current_state = env.state
#    print('init',type(current_state),current_state.shape) 
    total_reward = 0

    for steps in range(max_time_steps):
        print('Steps',steps)
        if count == 0:
            action = random.choice(env.action_space)
        else:
            action = agent.action_select(current_state,eph)
        next_state , reward  = env.step(action)
#        print('aaa',type(next_state),next_state.shape) 
        total_reward+=reward
        agent.push([current_state, reward, action, next_state]) #push to memory
        current_state = next_state
#        print('bbb',type(current_state),current_state.shape) 
        if count > agent.batch_size:
            agent.learn(count)
        count+=1
        if eph > eph_min:
            eph*=decay
    epsisode_rewards.append(total_reward)
    end_line = time.time()
    print('[Linterp]: dutation: ', end_line-star_line)
    print("Episode: ",episodes,"Reward: ",total_reward,"Average: ", np.mean(epsisode_rewards))
print('solved after ',episodes,' episodes')