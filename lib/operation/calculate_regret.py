import numpy as np

def cal_reg(res:dict, offline, grid):
    for alg in res.values():
        alg["regret"] = alg["value"][grid] - offline