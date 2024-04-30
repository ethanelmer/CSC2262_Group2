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
    return -(v_m - v_r) / tao_m + input_current / c_m
def S(t, t_s):
    return 0 if t - t_s - t_r <= 0 else 1  # Equation 2 piecewise
# Function for alpha synapse
def isyn(v_m, t, t0):
    # Ensure t0 is not negative infinity and t - t0 is positive
    if t0 == -np.inf or t - t0 < 0:
        return 0
    # Calculate the synaptic current using the alpha function
    time_since_spike = (t - t0)
    if time_since_spike < 0:
        return 0  # This ensures no synaptic input is calculated before the first spike
    synaptic_current = w * g_bar * (v_rev - v_m) * (time_since_spike / tao_syn) * np.exp(-(time_since_spike) / tao_syn)
    return synaptic_current

#Function to define ODE (multiplies S() and synaptic_current_term() functions
def dvm_dt(t, v_m, input_current, t_s):
    return synaptic_current_term(v_m, input_current) * S(t, t_s)


def LIF_model(mode, t, spike_rate, current=None):
    steps = int(t / dt)
    v_m = np.full(steps, v_r)  # Initialize with the resting potential
    time = np.linspace(0, t, steps)


    # Variables for tracking spikes and input currents
    last_spike_time = -np.inf  # Time of the last neuron's spike
    last_input_spike_time = 0  # Time of the last input spike from presynaptic neuron

    for i in range(1, steps):
        # Get the last input spike time
        if mode == "spike" and time[i] >= last_input_spike_time + 1 / spike_rate:
            presynaptic_spike_train = np.arange(0, t, dt)
            last_input_spike_time = presynaptic_spike_train[presynaptic_spike_train <= time[i]].max()

        # Calculate the synaptic current based on the most recent input spike
        if mode == "spike":
            input_current = isyn(v_m[i - 1], time[i], last_input_spike_time)

        # For "current" mode, just set the input current to the constant value
        elif mode == "current" and current is not None:
            input_current = current / 1e9  # Convert nA to A

        # Calculate dv/dt using Euler's method
        if time[i] - last_spike_time >= t_r:
            dv_dt = dvm_dt(time[i], v_m[i - 1], input_current, last_spike_time)
            v_m[i] = v_m[i - 1] + dv_dt * dt

        # Check if the neuron fires
        if v_m[i] >= v_thr:
            v_m[i] = v_spike  # Visualize spike
            last_spike_time = time[i]  # Update the time of the last spike
            if i < steps - 1:
                v_m[i + 1] = v_r  # Reset to resting potential after the spike

    # Plot the results
    time_ms = time * 1000  # Convert time to milliseconds for plotting
    plt.plot(time_ms, v_m, label='Membrane Potential')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Membrane Voltage vs Time")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simulate single LIF neuron alpha synapse using numerical methods.')

    # Positional arguments for mode and simulation time
    parser.add_argument("mode", choices=["current", "spike"], help="Simulation mode: 'current' or 'spike'")
    parser.add_argument("sim_time", type=float, help="Simulation time in milliseconds.")

    # Optional arguments
    parser.add_argument("--spike_rate", type=int, help="Spike rate in Hz (required for 'spike' mode).")
    parser.add_argument("--current", type=float, help="Input current in nanoamps (required for 'current' mode).")

    args = parser.parse_args()

    # Convert simulation time from milliseconds to seconds
    sim_time_seconds = args.sim_time / 1000

    # Error checks
    if args.mode == 'spike' and args.spike_rate is None:
        parser.error('--spike_rate is required when mode is "spike"')
    elif args.mode == 'current' and args.current is None:
        parser.error('--current is required when mode is "current"')

    # Run the model
    LIF_model(args.mode, sim_time_seconds, args.spike_rate, args.current)


if __name__ == "__main__":
    main()