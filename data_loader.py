import numpy as np
import pandas as pd
import torch
import pickle
import random
import math

class load_data:
    def __init__(self,args):
        super(load_data, self).__init__()
        data_file = args.datapath
        train_set, valid_set = self.get_data(data_file,args)
        self.train_set = train_set
        self.valid_set = valid_set

    # def make_batch(self,data,args):
    #     batch_data, batch_labels = [], []
    #     t_data = data[0]
    #     labels = data[1]
    #     num_batch = len(t_data)//args.batch_size
        
    #     for i in range(num_batch):
    #         batch_data.append(torch.FloatTensor(t_data[i*args.batch_size:(i+1)*args.batch_size]))
    #         batch_labels.append(torch.FloatTensor(labels[i*args.batch_size:(i+1)*args.batch_size]))
    #     batch_data.append(torch.FloatTensor(t_data[(i+1)*args.batch_size:]))
    #     batch_labels.append(torch.FloatTensor(labels[(i+1)*args.batch_size:]))
        
    #     return batch_data, batch_labels
    
    def get_data(self, path, args):
        df = pd.read_csv(path)
        t_interval = args.time_interval
        t_window = args.time_window

        if 'core' in args.datapath :
            speed_data = df.drop(columns=['Link_ID_1'	,'Link_ID_2',	'Center_Point_1',	'Center_Point_2',	'Limit'	,'Length',	'Direction']).to_numpy()
        else :
            speed_data = df.drop(columns=['Link_ID_1','Start_1','Start_2','End_1','End_2','Limit','-']).to_numpy()
            
        dev_sample_index = -1 * int(args.split_percentage * float(speed_data.shape[1]))
        train_set, valid_set = speed_data[:,:dev_sample_index], speed_data[:,dev_sample_index:]
        # data = []
        # labels = []
        # for i in range(train_set.shape[1]-args.time_window):
        #     data.append(train_set[:,i:i+args.time_window])
        #     labels.append(train_set[:,i+args.time_window])
        # batch_data, batch_labels = self.make_batch([data,labels],args)
        
        return torch.FloatTensor(train_set), torch.FloatTensor(valid_set)
    
    def get_tasks(self, time_window, time_interval, batch_size=100):
        timestep = int(time_interval/5)
        min_timestep_length = 2 * (time_window) * timestep + 12 # next 60분 하려면 최소 12 띄워야함

        train_tasks = []

        # train tasks sampling
        available_starting_points = list(range(self.train_set[:,:-(min_timestep_length)].shape[1]))
        # sampled_starting_points = random.choices(available_starting_points, k=batch_size)

        batch_num = math.ceil(len(available_starting_points)/batch_size)

        for batch in range(batch_num):
            if len(available_starting_points) >= batch_size :
                train_batch = np.random.choice(available_starting_points, size=batch_size, replace=False).tolist()
                for d in train_batch:
                    available_starting_points.remove(d)
            else :
                train_batch = available_starting_points

            train_tasks_batch = []
            for start_point in train_batch:
                sample_col_idx = [start_point]
                for i in range(time_window-1):
                    sample_col_idx.append(start_point+(i+1)*timestep)
                train_task = self.train_set[:,sample_col_idx]
                next_col_idx_for_time_window = []
                for i in range(time_window):
                    next_col_idx_for_time_window.append(sample_col_idx[-1]+(i+1)*timestep)
                y = self.train_set[:,next_col_idx_for_time_window]
                y_next_15 = self.train_set[:,sample_col_idx[-1]+3]
                y_next_30 = self.train_set[:,sample_col_idx[-1]+6]
                y_next_60 = self.train_set[:,sample_col_idx[-1]+12]
                train_tasks_batch.append((train_task, y, y_next_15, y_next_30, y_next_60))
            
            train_tasks.append(train_tasks_batch)

        # test tasks sampling
        available_starting_points = list(range(self.valid_set[:,:-(min_timestep_length)].shape[1]))
        sampled_starting_points = random.choices(available_starting_points, k=batch_size)
        valid_tasks = []
        for start_point in sampled_starting_points:
            sample_col_idx = [start_point]
            for i in range(time_window-1):
                sample_col_idx.append(start_point+(i+1)*timestep)
            next_col_idx_for_time_window = []
            for i in range(time_window):
                    next_col_idx_for_time_window.append(sample_col_idx[-1]+(i+1)*timestep)
            valid_task = self.valid_set[:,sample_col_idx]
            y = self.valid_set[:,next_col_idx_for_time_window]
            y_next_15 = self.valid_set[:,sample_col_idx[-1]+3]
            y_next_30 = self.valid_set[:,sample_col_idx[-1]+6]
            y_next_60 = self.valid_set[:,sample_col_idx[-1]+12]
            valid_tasks.append((valid_task, y, y_next_15, y_next_30, y_next_60))

        return train_tasks, valid_tasks