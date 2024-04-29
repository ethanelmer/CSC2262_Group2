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

def synaptic_current_term(v_m, input_current):
    return -(v_m - v_r) / tau_m + input_current / c_m
def S(t, t_s):
    return 0 if t - t_s - t_r <= 0 else 1  # Equation 2 piecewise
def isyn(v_m, t, t0, local_w):
    synaptic_current = local_w * g_bar * (v_rev - v_m) * ((t - t0) / tao_syn) * np.exp(-(t - t0) / tao_syn)
    # isyn = g * ((v_r-v_m)*((t-t0)/tao_syn))**(-(t-t0)/tao_syn) #This should be right for equation 3, but there may be
    return synaptic_current  # a difference. Not sure how the exponent is supposed to work.
def dvm_dt(t, v_m, v_r, tau_m, input_current, c_m, t_s, t_r):
    return synaptic_current_term(v_m, input_current) * S(t, t_s)
def euler_vm(v_m, input_current, c_m, t_r, dt, t, v_r, tau_m):
    dv_dt = dvm_dt(t, v_m, v_r, tau_m, input_current, c_m, t_s, t_r)
    v_m_next = v_m + dvm_dt * dt
    return v_m_next

def LIF_model(mode, t, spike_rate=None, current=None):
    steps = int(t / dt)
    v_m = np.zeros(steps)
    time = np.linspace(0, t, steps)
    local_w = w
    for i in range(1, steps):
        if v_m[i - 1] >= v_thr:
            v_m[i] = v_r
        else:
            if mode == "current":
                input_current = current
            elif mode == "spike":
                input_current = isyn(
                    v_m[i - 1], time[i], time[0], 1
                )
                input_current *= spike_rate / 1000  # Convert Hz to kHz
                v_m[i] = euler_vm(v_m[i - 1], input_current, c_m, t_r, dt, time[i], v_r, tao_m)
            else:
                raise ValueError("Invalid mode. Mode must be 'current' or 'spike'.")

    plt.plot(time, v_m)
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

    LIF_model(args.m, args.s, args.spike_rate, args.current)

    # GET VM USING EULER'S METHOD
    '''
    while t < duration:
        t = t + dt  # Increase step size
        v_m=0
        t0=0
        I_syn = 1 * ((v_rev - v_m) * ((t - t0(args.spike_rate())) / tao_syn) * (np.exp(-(t - t0()) / tao_syn)))
        v_m = (v_m+dt) * LIF_model(v_m, t, I_syn)
    '''


if __name__ == "__main__":
    main()
