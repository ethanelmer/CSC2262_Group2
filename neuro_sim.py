import argparse
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

if __name__ == "__main__":

    # Add arguments

    parser = argparse.ArgumentParser(description='Simulate single LIF neuron alpha synapse using numerical methods.')

    # Define group for '-m' which will require '--spike_rate' and '--current' if either mode is selected
    group = parser.add_mutually_exclusive_group(required=True)

    # '-m' argument for the mode of simulation
    group.add_argument('-m', type=str, help='Simulation mode (spike or current)', choices=['spike', 'current'],)

    # '-s' argument for how long to run the simulation in milliseconds
    parser.add_argument('-s', type=float, help='Simulation run-time (milliseconds)', required=True,)

    # '--spike_rate' argument required if 'spike' is chosen
    parser.add_argument('--spike_rate', type=int, help='Input spike rate in Hz (Only required for "spike" mode)',)

    # '--current' argument required if 'current' is chosen
    parser.add_argument('--current', type=float, help='Specify input current in nA (Only required for "current" mode)',)

    args = parser.parse_args()

    # Error check
    if args.m == 'spike' and not args.spike_rate:
        parser.error('--spike_rate is required when mode is "spike"')
    if args.m == 'current' and not args.current:
        parser.error('--current is required when mode is "current"')

    duration = 100

    #GET VM USING EULER'S METHOD
    while t < duration:
        t = t + dt  # Increase step size
        v_m=0
        t0=0
        I_syn = 1 * ((v_rev - v_m) * ((t - t0()) / tao_syn) * (np.exp(-(t - t0()) / tao_syn)))
        v_m = (v_m+dt) * LIF_neuron_model(v_m, t, I_syn)