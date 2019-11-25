

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
        
def load_state(agent_num, state_file):
    data_line = state_file.readline()
    if not data_line:
        print('EOF')
        return 'file_over'
    else:
        data_line = data_line.split(';')
        agent_set = []
        for i in range(agent_num):
            data = data_line[i].replace('[','').replace(']','')
            data = data.split(',')
            temp = State(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),float(data[8]),float(data[9]))
            temp.Set_priority(float(data[10]),float(data[11]),float(data[12]))
            agent_set.append(temp)
        return agent_set
        
if __name__ == '__main__':
    f = open('nothing.txt','r')
    test = load_state(2,f)
    test2 = load_state(2,f)
    f.close()