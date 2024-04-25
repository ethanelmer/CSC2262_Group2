import numpy as np
import matplotlib.pyplot as plt

import json
with open('config.json', 'r') as f:
    config = json.load(f)
v_r = config['v_r']
v_thr = config['v_thr']
v_spike = config['v_spike']
v_rev = config['v_rev']
tao_m = config['tao_m']
tao_syn = config['tao_syn']
c_m = config['c_m']
g_bar = config['g_bar']
t_r = config['t_r']
w = config['w']
dt = config['dt']
# initialize other variables
t = 0

def t0():
    t_spike_train = np.arrange(0, 100, (100 - 0) / hertz )
    for i in t_spike_train:
        t0 = t - t_spike_train[i]
        if(t0>0):
            return t0

duration = 100

def S(t, t_s):
    return 0 if t - t_s - t_r <= 0 \
        else 1

def LIF_neuron_model(v_m, t, I_syn):
    return ((-(v_m-v_r)/tao_m) + (I_syn/c_m)) * S(t, t_s)







