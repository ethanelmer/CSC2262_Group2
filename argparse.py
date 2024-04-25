import argparse

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

