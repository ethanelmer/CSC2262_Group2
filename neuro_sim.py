import argparse
import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork_functions import LIF_neuron_model
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# initialize other variables
#t = 0

def main():

    # Add argument
    parser = argparse.ArgumentParser(description='Simulate single LIF neuron alpha synapse using numerical methods.')

    # '-m' mode of the experiment (current or spike)
    parser.add_argument("m", choices=["current", "spike"], help="Simulation mode: 'current' or 'spike'")

    # '-s' runtime of simulation (milliseconds)
    parser.add_argument("s", type=float, help="Simulation time in milliseconds.")

    # '--spike_rate' and '--current'
    parser.add_argument("--spike_rate", type=int, help="Spike rate in Hz (required for 'spike' mode).")
    parser.add_argument("--current", type=float, help="Input current in nanoamps (required for 'current' mode).")
    args = parser.parse_args()

    # Error check
    if args.m == 'spike' and not args.spike_rate:
        parser.error('--spike_rate is required when mode is "spike"')
    elif args.m == 'current' and not args.current:
        parser.error('--current is required when mode is "current"')



    '''
        duration = 100
    
    #GET VM USING EULER'S METHOD
    while t < duration:
        t = t + dt  # Increase step size
        v_m=0
        t0=0
        I_syn = 1 * ((v_rev - v_m) * ((t - t0(args.spike_rate())) / tao_syn) * (np.exp(-(t - t0()) / tao_syn)))
        v_m = (v_m+dt) * LIF_neuron_model(v_m, t, I_syn)
    '''

if __name__ == "__main__":
    main()