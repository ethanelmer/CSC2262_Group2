import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define constants
# You can load these from the config.json file later
RESTING_POTENTIAL = -70e-3  # -70 mV in volts
SPIKE_THRESHOLD = -50e-3  # -50 mV in volts
SPIKE_VALUE = 40e-3  # 40 mV in volts
MEMBRANE_CAPACITANCE = 1e-12  # 1 pF in farads
REFRACTORY_PERIOD = 9.37e-3  # 9.37 ms in seconds
DECAY_TIME_CONSTANT = 0.3e-3  # 0.3 ms in seconds
MAX_CONDUCTANCE = 100e-9  # 100 nS in Siemens
SPIKE_SYNAPSE_DECAY = 3e-3  # 3 ms in seconds
DELTA_T = 0.001e-3  # 0.001 ms in seconds


def alpha_synapse_model(membrane_voltage, spike_time, current_time, weight):
    reversal_potential = 0
    decay_time_constant = 0.1
    max_conductance = 1e-9

    return max_conductance * (reversal_potential - membrane_voltage) * np.exp(
        -(current_time - spike_time) / decay_time_constant
    )


def leaky_integrate_and_fire(mode, simulation_time, spike_rate=None, current=None):
    # Initialize variables
    time_steps = int(simulation_time / DELTA_T)
    membrane_voltage = np.zeros(time_steps)
    time = np.linspace(0, simulation_time, time_steps)

    # Run simulation
    for i in range(1, time_steps):
        # Implement leaky integrate and fire dynamics
        if membrane_voltage[i - 1] >= SPIKE_THRESHOLD:
            membrane_voltage[i] = RESTING_POTENTIAL
        else:
            if mode == "current":
                input_current = current
            elif mode == "spike":
                input_current = alpha_synapse_model(
                    membrane_voltage[i - 1], time[0], time[i], weight=1
                )
                input_current *= spike_rate / 1000  # Convert Hz to kHz
            else:
                raise ValueError("Invalid mode. Mode must be 'current' or 'spike'.")

            membrane_voltage[i] = (
                membrane_voltage[i - 1]
                + (input_current / MEMBRANE_CAPACITANCE) * DELTA_T
                - (membrane_voltage[i - 1] - RESTING_POTENTIAL)
                * DELTA_T
                / REFRACTORY_PERIOD
            )

    # Plot results
    plt.plot(time, membrane_voltage)
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Membrane Voltage vs Time")
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simulate a LIF neuron.")
    parser.add_argument(
        "mode", choices=["current", "spike"], help="Simulation mode: 'current' or 'spike'"
    )
    parser.add_argument(
        "sim_time",
        type=float,
        help="Simulation time in milliseconds.",
    )
    parser.add_argument(
        "--spike_rate",
        type=int,
        help="Spike rate in Hz (required for 'spike' mode).",
    )
    parser.add_argument(
        "--current",
        type=float,
        help="Input current in nanoamps (required for 'current' mode).",
    )
    args = parser.parse_args()

    # Run simulation
    if args.mode == "current":
        leaky_integrate_and_fire(args.mode, args.sim_time / 1000, current=args.current)
    elif args.mode == "spike":
        leaky_integrate_and_fire(args.mode, args.sim_time / 1000, spike_rate=args.spike_rate)


if __name__ == "__main__":
    main()