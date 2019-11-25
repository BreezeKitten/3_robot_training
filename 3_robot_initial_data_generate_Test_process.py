import tensorflow as tf
import json
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import copy
import file_manger
import state_load as SL


tf.reset_default_graph()
'''
Common parameter
'''
PI = math.pi
resX = 0.1 # resolution of X
resY = 0.1 # resolution of Y
resTH = PI/15


'''
DL Parameter
'''
number_of_state = 15 #for relative coordinate the state variables will reduce 3
layer1_output_number = 150
layer2_output_number = 100
layer3_output_number = 100
layer4_output_number = 50 
training_eposide_num = 5000
training_num = 3000
test_num = 1


'''
Motion Parameter
'''
deltaT = 0.1            #unit:s
V_max = 3               #m/s
W_max = 2               #rad/s
linear_acc_max = 10     #m/s^2
angular_acc_max = 7     #rad/s^2
size_min = 0.1          #unit:m
x_upper_bound = 5       #unit:m
x_lower_bound = -5      #unit:m
y_upper_bound = 5       #unit:m
y_lower_bound = -5      #unit:m
TIME_OUT_FACTOR = 10


agnet2_motion = 'Greedy'
agnet3_motion = 'Greedy'
RL_eposide_num = 1000
RL_epsilon = 0.1
gamma = 0.8

Value_operation = 'nearest'

class State:
    def __init__(self, Px, Py, Pth, V, W, r, gx, gy, gth, rank):
        self.Px = Px
        self.Py = Py
        self.Pth = Pth
        self.V = V
        self.W = W
        self.r = r
        self.gx = gx
        self.gy = gy
        self.gth = gth
        self.rank = rank
    def Set_priority(self, m11, m12, m13):
        self.m11 = m11
        self.m12 = m12
        self.m13 = m13
        
def add_layer(inputs, in_size, out_size, W_name, B_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=W_name)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=B_name)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases
        

def Random_state():
    Px = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    Py = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    Pth = random.random()*2*PI 
    V = (random.random() - 0.5) * V_max
    W = (random.random() - 0.5) * W_max
    r = random.random() + size_min
    gx = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    gy = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    gth = random.random()*2*PI 
    rank = random.randint(1,3)    
    
    agent = State(Px, Py, Pth, V, W, r, gx, gy, gth, rank)
    return agent


def Calculate_distance(x1, y1, x2, y2):
    return np.sqrt(math.pow( (x1-x2) , 2) + math.pow( (y1-y2) , 2))

def angle_correct(angle):
    angle = math.fmod(angle, 2*PI)
    if angle < 0:
        angle = angle + 2*PI
    return angle

def Coordinate_transformation(new_originX, new_originY, new_originTH, x, y, th):
    x_temp = x - new_originX
    y_temp = y - new_originY
    th_new = angle_correct(th - new_originTH)
    x_new = math.cos(new_originTH) * x_temp + math.sin(new_originTH) * y_temp
    y_new = -math.sin(new_originTH) * x_temp + math.cos(new_originTH) * y_temp
    return x_new, y_new, th_new


def Motion_model(Px, Py, Pth, V, W):
    TH = Pth + W * deltaT
    TH = angle_correct(TH)    
    X = Px + V * deltaT * math.cos((Pth+TH)/2)
    Y = Py + V * deltaT * math.sin((Pth+TH)/2)
    
    return X, Y, TH

def Check_collussion(agent1, agent2):
    distance = Calculate_distance(agent1.Px, agent1.Py, agent2.Px, agent2.Py)
    if (distance <= (agent1.r + agent2.r)):
        return True
    else:
        return False
    
    
def Check_Goal(agent, position_tolerance, orientation_tolerance):
    position_error = Calculate_distance(agent.Px, agent.Py, agent.gx, agent.gy)
    orientation_error = abs(agent.Pth - agent.gth)
    if (position_error < position_tolerance) and (orientation_error < orientation_tolerance):
        return True
    else:
        return False
 
def Calculate_value(Path, reward, reward_time):
    for item in Path:
        remain_time_step = (reward_time - Path[item]['time_tag'])/deltaT
        Path[item]['Value'] = reward * math.pow(gamma, remain_time_step)
    return Path

def Observe_state(agent):
    Px_obs = agent.Px + (random.random()-0.5)*0.1
    Py_obs = agent.Py + (random.random()-0.5)*0.1
    Vx_obs = agent.V * math.cos(agent.Pth) + (random.random()-0.5)*0.1
    Vy_obs = agent.V * math.sin(agent.Pth) + (random.random()-0.5)*0.1
    r2_obs = agent.r + (random.random()-0.5)*0.05
    
    return Px_obs, Py_obs, Vx_obs, Vy_obs, r2_obs

def Predict_action_value(agent1, agent2, agent3, V_pred, W_pred):
    dummy = 0
    X_pred,  Y_pred, TH_pred = Motion_model(agent1.Px, agent1.Py, agent1.Pth, V_pred, W_pred)
    Px2, Py2, Vx2, Vy2, r2 = Observe_state(agent2)
    relative_gx, relative_gy, relative_gth = Coordinate_transformation(X_pred, Y_pred, TH_pred, agent1.gx, agent1.gy, agent1.gth)
    relative_Px2, relative_Py2, dummy = Coordinate_transformation(X_pred, Y_pred, TH_pred, Px2, Py2, dummy)
    relative_Vx2, relative_Vy2, dummy = Coordinate_transformation(0,0,TH_pred, Vx2, Vy2, dummy)
    
    Px3, Py3, Vx3, Vy3, r3 = Observe_state(agent3)
    relative_Px3, relative_Py3, dummy = Coordinate_transformation(X_pred, Y_pred, TH_pred, Px3, Py3, dummy)
    relative_Vx3, relative_Vy3, dummy = Coordinate_transformation(0,0,TH_pred, Vx3, Vy3, dummy)
    
    agent1_pred = copy.deepcopy(agent1)
    agent1_pred.Px = X_pred
    agent1_pred.Py = Y_pred
    agent1_pred.Pth = TH_pred

    agent2_pred = copy.deepcopy(agent2)
    agent2_pred.Px = Px2
    agent2_pred.Py = Py2
    
    agent3_pred = copy.deepcopy(agent3)
    agent3_pred.Px = Px3
    agent3_pred.Py = Py3

    R = 0
    if Check_collussion(agent1_pred,agent2_pred) or Check_collussion(agent1_pred,agent3_pred):
        R = -5
    elif Check_Goal(agent1_pred, Calculate_distance(resX, resY, 0, 0), resTH):
        R = 1
    
    state_pred12 = [[V_pred, W_pred, agent1.r, relative_gx, relative_gy, relative_gth, V_max, agent1.m11, agent1.m12, agent1.m13, relative_Px2, relative_Py2, relative_Vx2, relative_Vy2, r2]]
    value_matrix12 = sess.run(predict_value, feed_dict={state: state_pred12})
    action_value12 = R + value_matrix12[0][0]
    
    state_pred13 = [[V_pred, W_pred, agent1.r, relative_gx, relative_gy, relative_gth, V_max, agent1.m11, agent1.m12, agent1.m13, relative_Px3, relative_Py3, relative_Vx3, relative_Vy3, r3]]
    value_matrix13 = sess.run(predict_value, feed_dict={state: state_pred13})
    action_value13 = R + value_matrix13[0][0]
    
    if Value_operation == 'Add':
        action_value = action_value12 + action_value13
    elif Value_operation == 'multi':
        action_value = action_value12 * action_value13
    elif Value_operation == 'nearest':
        if Calculate_distance(agent1.Px, agent1.Py, agent2.Px, agent2.Py) < Calculate_distance(agent1.Px, agent1.Py, agent3.Px, agent3.Py):
            action_value = action_value12
        else:
            action_value = action_value13
    else:
        print('Exception')
        action_value = 0

    return action_value    

    
def Choose_action(agent1, agent2, agent3, epsilon):
    dice = random.random()
    action_value_max = -999999
    if dice < epsilon:
        linear_acc = -linear_acc_max + random.random() * 2 * linear_acc_max
        angular_acc = -angular_acc_max + random.random() * 2 * angular_acc_max
        V_pred = np.clip(agent1.V + linear_acc * deltaT, -V_max, V_max)
        W_pred = np.clip(agent1.W + angular_acc * deltaT, -W_max, W_max)
    else:
        linear_acc_set = np.arange(-linear_acc_max, linear_acc_max, 1)
        angular_acc_set = np.arange(-angular_acc_max, angular_acc_max, 1)
        for linear_acc in linear_acc_set:
            V_pred = np.clip(agent1.V + linear_acc * deltaT, -V_max, V_max)
            for angular_acc in angular_acc_set:
                W_pred = np.clip(agent1.W + angular_acc * deltaT, -W_max, W_max)
                action_value = Predict_action_value(agent1, agent2, agent3, V_pred, W_pred)
                if action_value > action_value_max:
                    action_value_max = action_value
                    action_pair = [V_pred, W_pred]                    
        V_pred = action_pair[0]
        W_pred = action_pair[1]
        
    return V_pred, W_pred

def Update_state(agent, V_next, W_next):
    Px_next, Py_next, Pth_next = Motion_model(agent.Px, agent.Py, agent.Pth, V_next, W_next)    
    agent.Px = Px_next
    agent.Py = Py_next
    agent.Pth = Pth_next
    agent.V = V_next
    agent.W = W_next    

    return agent

def Record_Path(agent1, agent2, agent3, time):
    Vx2 = agent2.V * math.cos(agent2.Pth)
    Vy2 = agent2.V * math.sin(agent2.Pth)
    Vx3 = agent3.V * math.cos(agent3.Pth)
    Vy3 = agent3.V * math.sin(agent3.Pth)
    temp = {}
    temp['Px'] = agent1.Px
    temp['Py'] = agent1.Py
    temp['Pth'] = agent1.Pth
    temp['V'] = agent1.V
    temp['W'] = agent1.W
    temp['r1'] = agent1.r
    temp['gx'] = agent1.gx
    temp['gy'] = agent1.gy
    temp['gth'] = agent1.gth
    temp['Vmax'] = V_max
    temp['m11'] = agent1.m11
    temp['m12'] = agent1.m12
    temp['m13'] = agent1.m13
    temp['Px2'] = agent2.Px
    temp['Py2'] = agent2.Py
    temp['Vx2'] = Vx2
    temp['Vy2'] = Vy2
    temp['r2'] = agent2.r
    temp['Px3'] = agent3.Px
    temp['Py3'] = agent3.Py
    temp['Vx3'] = Vx3
    temp['Vy3'] = Vy3
    temp['r3'] = agent3.r
    temp['time_tag'] = time
    return temp
                
       
def Read_data(file_name):
    data = {}
    file = open(file_name,'r')
    data_line = file.readline()
    count = 0
    while(data_line):
        data[count] = json.loads(data_line)
        data_line = file.readline()
        count = count + 1
    file.close()
    return data

def Record_data(data, file_name):
    file = open(file_name, 'a+')
    for item in data:
        json.dump(data[item],file)
        file.writelines('\n')   
    file.close()
    return
       
       
def Sample_data(data_base, sample_number):
    sampled_data = {}
    sample_array = random.sample(range(0,len(data_base)), sample_number)
    for index in sample_array:
        sampled_data[index] = data_base[index]
    return sampled_data

   

def Divide_state_value(data):
    Start_flag = 1
    for item in data:
        temp_state = [[data[item]['V'],data[item]['W'],data[item]['r1'],data[item]['gx'],data[item]['gy'],data[item]['gth'],data[item]['Vmax'],data[item]['m11'],data[item]['m12'],data[item]['m13'],data[item]['Px2'],data[item]['Py2'],data[item]['Vx2'],data[item]['Vy2'],data[item]['r2']]]
        temp_value = [[data[item]['Value']]]
        if Start_flag:
            state = temp_state
            value = temp_value
            Start_flag = 0
        else:
            state = np.concatenate((state, temp_state), axis=0)
            value = np.concatenate((value, temp_value), axis=0)
    return state, value


def Show_Path(Path, result, final_time, SAVE_PATH):
    L = 0.5
    plt.close('all')
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    ax.cla()
    
    ax.set_xlim((x_lower_bound,x_upper_bound))     #上下限
    ax.set_ylim((x_lower_bound,x_upper_bound))
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    i = 0
    Px_last = Path[0]['Px']
    Py_last = Path[0]['Py']
    Px2_last = Path[0]['Px2']
    Py2_last = Path[0]['Py2']
    Px3_last = Path[0]['Px3']
    Py3_last = Path[0]['Py3']    
    plt.plot(Path[0]['Px'], Path[0]['Py'], 'yo', Path[0]['gx'], Path[0]['gy'], 'mo')
    plt.arrow(Path[0]['gx'], Path[0]['gy'], L*math.cos(Path[0]['gth']), L*math.sin(Path[0]['gth']))
    for item in np.arange(0,final_time,deltaT):
        item = round(item,1)
        if((i%10)==0):
            circle1 = plt.Circle((Path[item]['Px'],Path[item]['Py']), Path[item]['r1'], color = 'b', fill = False)
            circle2 = plt.Circle((Path[item]['Px2'],Path[item]['Py2']), Path[item]['r2'], color = 'r', fill = False)
            circle3 = plt.Circle((Path[item]['Px3'],Path[item]['Py3']), Path[item]['r3'], color = 'g', fill = False)
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            ax.add_artist(circle3)
            plt.arrow(Path[item]['Px'], Path[item]['Py'], L*math.cos(Path[item]['Pth']), L*math.sin(Path[item]['Pth']))
            plt.text(Path[item]['Px']-0.2, Path[item]['Py'], str(round(i*deltaT,1)), bbox=dict(color='blue', alpha=0.5))
            plt.text(Path[item]['Px2']-0.2, Path[item]['Py2'], str(round(i*deltaT,1)), bbox=dict(color='red', alpha=0.5))
            plt.text(Path[item]['Px3']-0.2, Path[item]['Py3'], str(round(i*deltaT,1)), bbox=dict(color='green', alpha=0.5))
        if(i>0):
            plt.plot([Px_last, Path[item]['Px']], [Py_last, Path[item]['Py']], 'b-')
            plt.plot([Px2_last, Path[item]['Px2']], [Py2_last, Path[item]['Py2']], 'r-')
            plt.plot([Px3_last, Path[item]['Px3']], [Py3_last, Path[item]['Py3']], 'g-')
        i = i+1
        Px_last = Path[item]['Px']
        Py_last = Path[item]['Py']
        Px2_last = Path[item]['Px2']
        Py2_last = Path[item]['Py2']
        Px3_last = Path[item]['Px3']
        Py3_last = Path[item]['Py3']
    
    NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig(SAVE_PATH +'/image/'+ NOW + result +'.png')
    #plt.show()
    
    return

def Transform_data_to_relative_coordinate(read_file_name, save_file_name):
    data = Read_data(read_file_name)
    relative_data = {}
    dummy = 0
    for item in data:
        relative_data[item] = {}
        relative_gx, relative_gy, relative_gth = Coordinate_transformation(data[item]['Px'],data[item]['Py'],data[item]['Pth'],data[item]['gx'],data[item]['gy'],data[item]['gth'])
        relative_Px2, relative_Py2, dummy = Coordinate_transformation(data[item]['Px'],data[item]['Py'],data[item]['Pth'],data[item]['Px2'],data[item]['Py2'], dummy)
        relative_Vx2, relative_Vy2, dummy = Coordinate_transformation(0,0,data[item]['Pth'],data[item]['Vx2'],data[item]['Vy2'], dummy)
        relative_Px3, relative_Py3, dummy = Coordinate_transformation(data[item]['Px'],data[item]['Py'],data[item]['Pth'],data[item]['Px3'],data[item]['Py3'], dummy)
        relative_Vx3, relative_Vy3, dummy = Coordinate_transformation(0,0,data[item]['Pth'],data[item]['Vx3'],data[item]['Vy3'], dummy)
        relative_data[item]['V'] = data[item]['V']
        relative_data[item]['W'] = data[item]['W']
        relative_data[item]['r1'] = data[item]['r1']
        relative_data[item]['gx'] = relative_gx
        relative_data[item]['gy'] = relative_gy
        relative_data[item]['gth'] = relative_gth 
        relative_data[item]['Vmax'] = data[item]['Vmax']
        relative_data[item]['m11'] = data[item]['m11']
        relative_data[item]['m12'] = data[item]['m12']
        relative_data[item]['m13'] = data[item]['m13']
        relative_data[item]['Px2'] = relative_Px2
        relative_data[item]['Py2'] = relative_Py2
        relative_data[item]['Vx2'] = relative_Vx2
        relative_data[item]['Vy2'] = relative_Vy2
        relative_data[item]['r2'] = data[item]['r2']
        relative_data[item]['Px3'] = relative_Px3
        relative_data[item]['Py3'] = relative_Py3
        relative_data[item]['Vx3'] = relative_Vx3
        relative_data[item]['Vy3'] = relative_Vy3
        relative_data[item]['r3'] = data[item]['r3']
        relative_data[item]['Value'] = data[item]['Value']
        
    Record_data(relative_data, save_file_name)    
    return
    
        
        
        
def DL_process(DL_database):
    data = Read_data(DL_database)
    #test_data = Sample_data(data, test_num)
    #test_state, test_value = Divide_state_value(test_data)
    #test_predict = []
    
    for training_eposide in range(training_eposide_num):
        training_data = Sample_data(data, training_num)
        training_state, training_value = Divide_state_value(training_data)
        sess.run(train_step, feed_dict={state: training_state, value: training_value})
        
        #test_predict.append(sess.run(predict_value, feed_dict={state: test_state}))
        if training_eposide%100 == 0:
            rs = sess.run(loss_record, feed_dict = {state: training_state, value: training_value})
            writer.add_summary(rs, training_eposide)
            print('record', training_eposide)
        #print('eposide: ',training_eposide, 'test error: ', test_value-test_predict[-1][0][0])
    saver.save(sess,'relative_network/test.ckpt')    
    return
        
def RL_process(eposide_num, epsilon, RL_SAVE_PATH):      
    for eposide in range(eposide_num):
        
        if eposide%20 == 0:
            print(eposide)
            
        agent1 = Random_state()
        agent2 = Random_state()
        agent3 = Random_state()
        
        agent1.Set_priority(0,0,1)
        agent2.Set_priority(0,0,1)
        agent3.Set_priority(0,0,1)
        
        
        
        time = 0
        Path = {}
        result = 'Finish'
        if Check_collussion(agent1, agent2) or Check_collussion(agent1, agent3):
            continue
        if Check_Goal(agent1, Calculate_distance(resX, resY, 0, 0), resTH):
            continue
        TIME_OUT = Calculate_distance(agent1.Px, agent1.Py, agent1.gx, agent1.gy) * TIME_OUT_FACTOR
        Path[round(time,1)] = Record_Path(agent1, agent2, agent3, time)
        
        agent1_init_state = [agent1.Px, agent1.Py, agent1.Pth, agent1.V, agent1.W, agent1.r, agent1.gx, agent1.gy, agent1.gth, agent1.rank, agent1.m11, agent1.m12, agent1.m13, V_max]
        agent2_init_state = [agent2.Px, agent2.Py, agent2.Pth, agent2.V, agent2.W, agent2.r, agent2.gx, agent2.gy, agent2.gth, agent2.rank, agent2.m11, agent2.m12, agent2.m13, V_max]
        agent3_init_state = [agent3.Px, agent3.Py, agent3.Pth, agent3.V, agent3.W, agent3.r, agent3.gx, agent3.gy, agent3.gth, agent3.rank, agent3.m11, agent3.m12, agent3.m13, V_max]

        
        while(not Check_Goal(agent1, Calculate_distance(resX, resY, 0, 0), resTH)):
            if time > TIME_OUT:
                result = 'TIME_OUT'
                break
            elif Check_collussion(agent1, agent2) or Check_collussion(agent1, agent3):
                result = 'Collussion'
                break
            else:
                V1_next, W1_next = Choose_action(agent1, agent2, agent3, epsilon)
                agent1 = Update_state(agent1, V1_next, W1_next)
                
                if agnet2_motion == 'Static':
                    V2_next = 0
                    W2_next = 0
                elif agnet2_motion == 'Greedy':
                    V2_next, W2_next = Choose_action(agent2, agent1, agent3, 0)
                elif agnet2_motion == 'RL':
                    V2_next, W2_next = Choose_action(agent2, agent1, agent3, epsilon)
                else:
                    V2_next = agent2.V + random.random() - 0.5
                    W2_next = agent2.W + random.random() - 0.5
                agent2 = Update_state(agent2, V2_next, W2_next)
                
                if agnet3_motion == 'Static':
                    V3_next = 0
                    W3_next = 0
                elif agnet3_motion == 'Greedy':
                    V3_next, W3_next = Choose_action(agent3, agent1, agent2, 0)
                elif agnet3_motion == 'RL':
                    V3_next, W3_next = Choose_action(agent3, agent1, agent2, epsilon)
                else:
                    V3_next = agent3.V + random.random() - 0.5
                    W3_next = agent3.W + random.random() - 0.5
                agent3 = Update_state(agent3, V3_next, W3_next)
                
            time = time + deltaT
            Path[round(time,1)] = Record_Path(agent1, agent2, agent3, time)
            
        lines =  str(agent1_init_state) + ';' + str(agent2_init_state) + ';' + str(agent3_init_state) + '\n'   
        if result == 'Finish':
            Path = Calculate_value(Path, 1, time)
            f = open(RL_SAVE_PATH + '/Finish.json', 'a+')
            f.writelines(lines)
            f.close()
        elif result == 'TIME_OUT':
            Path = Calculate_value(Path, -1, time)
            f = open(RL_SAVE_PATH + '/TimeOut.json', 'a+')
            f.writelines(lines)
            f.close()
        elif result == 'Collussion':
            Path = Calculate_value(Path, -5, time)
            f = open(RL_SAVE_PATH + '/Collussion.json', 'a+')
            f.writelines(lines)
            f.close()
        else:
            print('Unexpected result: ', result)
            f = open(RL_SAVE_PATH + '/Unexpected.json', 'a+')
            f.writelines(lines)
            f.close()
            
        Record_data(Path, RL_SAVE_PATH +'/RL_Path.json')
        
        
        Show_Path(Path, result, time, RL_SAVE_PATH)

        
        Path.clear() 
    return
    
 
def Test_process(State_file, TEST_SAVE_PATH):    
    state_data = open(State_file, 'r')
    agent_set = SL.load_state(3,state_data)
    epsilon = 0
    while agent_set != 'file_over':        
        agent1 = agent_set[0]
        agent2 = agent_set[1]
        agent3 = agent_set[1]
        
        
        
        time = 0
        Path = {}
        result = 'Finish'
        if Check_collussion(agent1, agent2) or Check_collussion(agent1, agent3):
            continue
        if Check_Goal(agent1, Calculate_distance(resX, resY, 0, 0), resTH):
            continue
        TIME_OUT = Calculate_distance(agent1.Px, agent1.Py, agent1.gx, agent1.gy) * TIME_OUT_FACTOR
        Path[round(time,1)] = Record_Path(agent1, agent2, agent3, time)
        
        agent1_init_state = [agent1.Px, agent1.Py, agent1.Pth, agent1.V, agent1.W, agent1.r, agent1.gx, agent1.gy, agent1.gth, agent1.rank, agent1.m11, agent1.m12, agent1.m13, V_max]
        agent2_init_state = [agent2.Px, agent2.Py, agent2.Pth, agent2.V, agent2.W, agent2.r, agent2.gx, agent2.gy, agent2.gth, agent2.rank, agent2.m11, agent2.m12, agent2.m13, V_max]
        agent3_init_state = [agent3.Px, agent3.Py, agent3.Pth, agent3.V, agent3.W, agent3.r, agent3.gx, agent3.gy, agent3.gth, agent3.rank, agent3.m11, agent3.m12, agent3.m13, V_max]

        
        while(not Check_Goal(agent1, Calculate_distance(resX, resY, 0, 0), resTH)):
            if time > TIME_OUT:
                result = 'TIME_OUT'
                break
            elif Check_collussion(agent1, agent2) or Check_collussion(agent1, agent3):
                result = 'Collussion'
                break
            else:
                V1_next, W1_next = Choose_action(agent1, agent2, agent3, epsilon)
                agent1 = Update_state(agent1, V1_next, W1_next)
                
                if agnet2_motion == 'Static':
                    V2_next = 0
                    W2_next = 0
                elif agnet2_motion == 'Greedy':
                    V2_next, W2_next = Choose_action(agent2, agent1, agent3, 0)
                elif agnet2_motion == 'RL':
                    V2_next, W2_next = Choose_action(agent2, agent1, agent3, epsilon)
                else:
                    V2_next = agent2.V + random.random() - 0.5
                    W2_next = agent2.W + random.random() - 0.5
                agent2 = Update_state(agent2, V2_next, W2_next)
                
                if agnet3_motion == 'Static':
                    V3_next = 0
                    W3_next = 0
                elif agnet3_motion == 'Greedy':
                    V3_next, W3_next = Choose_action(agent3, agent1, agent2, 0)
                elif agnet3_motion == 'RL':
                    V3_next, W3_next = Choose_action(agent3, agent1, agent2, epsilon)
                else:
                    V3_next = agent3.V + random.random() - 0.5
                    W3_next = agent3.W + random.random() - 0.5
                agent3 = Update_state(agent3, V3_next, W3_next)
                
            time = time + deltaT
            Path[round(time,1)] = Record_Path(agent1, agent2, agent3, time)
            
        lines =  str(agent1_init_state) + ';' + str(agent2_init_state) + ';' + str(agent3_init_state) + '\n'   
        if result == 'Finish':
            Path = Calculate_value(Path, 1, time)
            f = open(TEST_SAVE_PATH + '/Finish.json', 'a+')
            f.writelines(lines)
            f.close()
        elif result == 'TIME_OUT':
            Path = Calculate_value(Path, -1, time)
            f = open(TEST_SAVE_PATH + '/TimeOut.json', 'a+')
            f.writelines(lines)
            f.close()
        elif result == 'Collussion':
            Path = Calculate_value(Path, -5, time)
            f = open(TEST_SAVE_PATH + '/Collussion.json', 'a+')
            f.writelines(lines)
            f.close()
        else:
            print('Unexpected result: ', result)
            f = open(TEST_SAVE_PATH + '/Unexpected.json', 'a+')
            f.writelines(lines)
            f.close()
            
        Record_data(Path, TEST_SAVE_PATH +'/TEST_Path.json')      
        Show_Path(Path, result, time, TEST_SAVE_PATH)

        
        Path.clear() 
        agent_set = SL.load_state(3,state_data)
    state_data.close()
    return
        
                   
                
    
    

if __name__ == '__main__':
    
    NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '_initial_data'
    FM = file_manger.file_manger('logs',NOW)
    SAVE_DIR = FM.log_path
    FM.create_dir()
    
    state = tf.placeholder(tf.float32, [None, number_of_state])
    value = tf.placeholder(tf.float32, [None, 1])
    
    H1, W1, B1 = add_layer(state, number_of_state, layer1_output_number, 'W1', 'B1', activation_function=tf.nn.relu)
    H2, W2, B2 = add_layer(H1, layer1_output_number, layer2_output_number, 'W2', 'B2', activation_function=tf.nn.relu)
    H3, W3, B3 = add_layer(H2, layer2_output_number, layer3_output_number, 'W3', 'B3', activation_function=tf.nn.relu)
    H4, W4, B4 = add_layer(H3, layer3_output_number, layer4_output_number, 'W4', 'B4', activation_function=tf.nn.sigmoid)
    predict_value, Wf, Bf = add_layer(H4, layer4_output_number, 1, 'Wf', 'Bf', activation_function=tf.nn.sigmoid)
    
    cost = tf.losses.mean_squared_error(predict_value, value)
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    loss = cost + 0.0001* regularizers
    
    loss_record = tf.summary.scalar('loss',loss)
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SAVE_DIR+'/training_record/DL_logs/', sess.graph)
    
    
    init = tf.global_variables_initializer()
    sess.run(init)       
    
    RL_SAVE_PATH = SAVE_DIR+'/training_record'
    TEST_SAVE_PATH = SAVE_DIR+'/test_result'
    DL_database = 'record/initial_data.json'
    
    saver.restore(sess,'2_robot_network/test.ckpt')
    
    '''
    for i in range(1):
        print('Start Process',i)
        RL_process(RL_eposide_num, RL_epsilon, RL_SAVE_PATH)
        print('Finish RL',i)
        DL_process(DL_database)
        print('Finish DL',i)
    '''
    '''
    FM.Network_copy('relative_network', False)
    print('start RL')
    RL_process(RL_eposide_num, RL_epsilon, RL_SAVE_PATH)
    Transform_data_to_relative_coordinate(RL_SAVE_PATH +'/RL_Path.json', DL_database)
    print('start DL')
    DL_process(DL_database)
    FM.Network_copy('relative_network', True)
    print('start TEST')
    RL_process(50, 0, TEST_SAVE_PATH)
    print('Finish')
    '''
    RL_process(2000,0, TEST_SAVE_PATH)
    Transform_data_to_relative_coordinate(TEST_SAVE_PATH +'/RL_Path.json', DL_database)

   