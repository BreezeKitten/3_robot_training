import os
import shutil

class file_manger():
    def __init__(self, log_path, time):
        self.log_path = log_path + '/' + time
        
    def create_dir(self):
        os.makedirs(self.log_path)
        os.makedirs(self.log_path + '/training_record')
        os.makedirs(self.log_path + '/training_record/image')
        os.makedirs(self.log_path + '/training_record/DL_logs')
        os.makedirs(self.log_path + '/test_result')
        os.makedirs(self.log_path + '/test_result/image')
        
    def Network_copy(self, Network_path, New_flag):
        if New_flag:
            shutil.copytree(Network_path, self.log_path + '/New_Network')
        else:
            shutil.copytree(Network_path, self.log_path + '/Network_backup')
            
if __name__ == '__main__':
    FM = file_manger('log_test', '1124')
            