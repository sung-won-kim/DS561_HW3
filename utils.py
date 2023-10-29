import numpy as np
import time
import math
import torch
import copy
import torch.nn as nn

def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))


def getBatch(data, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]

    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield batch



def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ""
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in [
            "savepath",
            'seed',
            'datapath'
        ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]