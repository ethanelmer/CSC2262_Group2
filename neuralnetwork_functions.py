import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def t0(hertz): #Equation 1
    t_spike_train = np.arrange(0, 100, (100 - 0) / hertz )
    for i in t_spike_train:
        t0 = t - t_spike_train[i]
        if(t0>0):
            return t0

duration = 100

def S(t, t_s):
    return 0 if t - t_s - t_r <= 0 else 1 #Equation 2 piecewise

def isyn(v_m, t, t0, w):
    isyn = g_bar * ((v_rev - v_m) * ((t - t0) /tao_syn)) * np.exp(-(t - t0)/tao_syn)
    #isyn = g * ((v_r-v_m)*((t-t0)/tao_syn))**(-(t-t0)/tao_syn) #This should be right for equation 3, but there may be
    return isyn                                      #a difference. Not sure how the exponent is supposed to work.
def LIF_model(mode, t, spike_rate=None, current=None):
    steps = int(t/dt)
    vm = np.zeros(steps)
    time = np.linspace(0,t,steps)
    for i in range (1,steps):
        if(vm[i-1]>=v_thr):
            vm[i]=v_r
        else:
            if mode == "current":
                input_current = current
            elif mode == "spike":
                input_current = isyn(
                    vm[i - 1], time[i], time[0], 1
                )
                input_current *= spike_rate / 1000  # Convert Hz to kHz
            else:
                raise ValueError("Invalid mode. Mode must be 'current' or 'spike'.")

            vm[i] = (
                vm[i - 1]
                + (input_current / c_m) * dt
                - (vm[i - 1] - v_r)
                * dt
                / t_r)
    plt.plot(time, vm)
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Membrane Voltage vs Time")
    plt.show()



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

    LIF_model(args.m, args.s, args.spike_rate)



    #GET VM USING EULER'S METHOD
    while t < duration:
        t = t + dt  # Increase step size
        v_m=0
        t0=0
        I_syn = 1 * ((v_rev - v_m) * ((t - t0(args.spike_rate())) / tao_syn) * (np.exp(-(t - t0()) / tao_syn)))
        v_m = (v_m+dt) * LIF_neuron_model(v_m, t, I_syn)


if __name__ == "__main__":
    main()