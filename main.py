import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import argparse
import random
from data_loader import load_data

from utils import *

import time
from tqdm import tqdm

np.random.seed(10)


def main():
    parser = argparse.ArgumentParser()
    timestr = time.strftime("%m%d-%H%M")

    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model', type=str, default='LSTM_correlation_multi_attention')
    parser.add_argument('--savepath', type=str, default='./saved_results')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--time_interval', type=float,
                        default=5)  # min단위 / 5분 단위로 설정
    parser.add_argument('--time_window', type=int, default=5)
    parser.add_argument('--task_num', type=int, default=100)
    parser.add_argument('--datapath', type=str, default='./SeoulData/urban-core_v2.csv',
                        help='data path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lstm_n_layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument("--summary", type=str, default=timestr)
    parser.add_argument("--save_model", action = 'store_true', default=False) 

    args = parser.parse_args()

    timestr = time.strftime("%m%d-%H%M")
    config_str = config2string(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data = load_data(args)
    # train_tasks = [[batch],...,[batch]], [batch] = [task1, task2, ..., task batch_size], task1 = (speed_data, y for next time_window, y_15, y_30, y_60)
    train_tasks, valid_tasks = data.get_tasks(
        time_window=args.time_window, time_interval=args.time_interval, batch_size=args.batch_size)
    num_point = valid_tasks[0][0].shape[0]

    device = torch.device("cuda:" + str(args.device)
                          if torch.cuda.is_available() else "cpu")

    if args.model == 'LSTM':
        from LSTM import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
    elif args.model == 'LSTM_correlation':
        from LSTM_correlation import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
    elif args.model == 'LSTM_correlation_v2':
        from LSTM_correlation_v2 import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
    elif args.model == 'LSTM_correlation_v3':
        from LSTM_correlation_v3 import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
    elif args.model == 'LSTM_correlation_attention':
        from LSTM_correlation_attention import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
    elif args.model == 'LSTM_correlation_multi_attention':
        from LSTM_correlation_attention import LSTM
        lstm_model = LSTM(args.time_window, num_point,
                          num_layers=args.lstm_n_layers, output_size=1).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        lstm_model.parameters(), lr=args.lr, weight_decay=0)

    count = 0
    count2 = 0
    best_acc = 100000
    lstm_model.train()
    for epoch in tqdm(range(args.epochs)):
        num_tasks = 0
        train_epoch_loss = 0
        for batch in tqdm(train_tasks):
            train_batch_loss = 0
            for task in batch:
                optimizer.zero_grad()
                sequence = task[0].to(device)
                y = task[1].to(device)
                y_15 = task[2].to(device)
                y_30 = task[3].to(device)
                y_60 = task[4].to(device)
                if args.model == 'LSTM_correlation_v3':
                    pred_x, spatial_loss = lstm_model(sequence)
                    loss = criterion(pred_x.squeeze(1).float(), y[:, 0].view(-1))
                    loss += spatial_loss
                else:
                    pred_x = lstm_model(sequence)
                    loss = criterion(pred_x.squeeze(1).float(), y[:, 0].view(-1))
                # loss = criterion(pred_x.squeeze(1).float(), y_15.view(-1))
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                train_batch_loss += loss.item()
            # tqdm.write('Batch Loss: %.3f' % (
            #     train_batch_loss / len(batch)))
            num_tasks += len(batch)
        tqdm.write('Epoch Loss: %.3f' % (
            train_epoch_loss / num_tasks))

        current_acc = train_epoch_loss / num_tasks

        if best_acc > current_acc:
            if args.save_model :
                if 'core' in args.datapath :
                    torch.save(lstm_model,f'./model_state_dict/core/{args.model}.pt')
                else :
                    torch.save(lstm_model,f'./model_state_dict/mix/{args.model}.pt')

            best_acc = current_acc
            count = 0
        else:
            count += 1
            tqdm.write(f'count : {count}/{args.patience}')

        if count >= args.patience:
            break

        if current_acc > 100 :
            count2 += 1
        if count2 >= 10:
            break

    lstm_model.eval()
    loss_15 = []
    loss_30 = []
    loss_60 = []
    num_tasks = 0

    INFO_LOG("-------------------------------------------------------Test")
    with torch.no_grad():
        start = time.time()
        if args.save_model :
            if 'core' in args.datapath :
                lstm_model = torch.load(f'./model_state_dict/core/{args.model}.pt')
            else :
                lstm_model = torch.load(f'./model_state_dict/mix/{args.model}.pt')

        for task in valid_tasks:

            for i in range(12):
                sequence = task[0].to(device)
                y_15 = task[2].to(device)
                y_30 = task[3].to(device)
                y_60 = task[4].to(device)
                if args.model == 'LSTM_correlation_v3':
                    pred_x, spatial_loss = lstm_model(sequence)
                else:
                    pred_x = lstm_model(sequence)
                sequence = torch.concat((sequence[:, 1:], pred_x), 1)
                if i == 2:  # 15
                    loss_15.append(
                        criterion(pred_x.squeeze(1).float(), y_15.view(-1)))
                elif i == 5:  # 30
                    loss_30.append(
                        criterion(pred_x.squeeze(1).float(), y_30.view(-1)))
                elif i == 11:  # 60
                    loss_60.append(
                        criterion(pred_x.squeeze(1).float(), y_60.view(-1)))

    num_tasks = len(valid_tasks)

    f = open(f'{args.savepath}/results.txt', 'a')
    f.write(f"======================================\n")
    f.write(f"# Final Result : {config_str}\n")
    f.write(f"# Summary : {args.summary}\n")
    f.write('15 min loss: ' + str((sum(loss_15)/len(valid_tasks)).item())+'\n')
    f.write('30 min loss: ' + str((sum(loss_30)/len(valid_tasks)).item())+'\n')
    f.write('60 min loss: ' + str((sum(loss_60)/len(valid_tasks)).item())+'\n\n')
    f.close()
    print("15 min loss:", (sum(loss_15)/len(valid_tasks)).item())
    print("30 min loss:", (sum(loss_30)/len(valid_tasks)).item())
    print("60 min loss:", (sum(loss_60)/len(valid_tasks)).item())


if __name__ == '__main__':
    main()
