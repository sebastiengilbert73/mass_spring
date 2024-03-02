import logging
import argparse
import os
import sys
sys.path.append("../src/mass_spring")
import simulation
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    mass,
    gamma,
    k,
    x0,
    v0,
    duration,
    numberOfTimesteps,
    zeroThreshold
):
    logging.info("display_xva.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    mass_spring = simulation.MassSpring(
        mass=mass,
        gamma=gamma,
        k=k,
        x0=x0,
        v0=v0,
        zero_threshold=zeroThreshold
    )

    timestep = duration/numberOfTimesteps
    ts = np.arange(0, duration + timestep, timestep).tolist()
    xs = [mass_spring.evaluate(t) for t in ts]
    vs = [mass_spring.evaluate_velocity(t) for t in ts]
    accs = [mass_spring.evaluate_acceleration(t) for t in ts]

    # Plot the three signals
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.suptitle(f"mass={mass}; gamma={gamma}; k={k}; x0={x0}; v0={v0}")

    axs[0].plot(ts, xs, label="Position (m)")
    axs[0].set_ylabel("x (m)")
    axs[0].grid(True)
    plt.legend(loc='upper right')
    axs[1].plot(ts, vs, label="Velocity (m/s)")
    axs[1].set_ylabel("v (m/s)")
    axs[1].grid(True)
    plt.legend(loc='upper right')
    axs[2].plot(ts, accs, label="Acceleration (m/s^2)")
    axs[2].set_ylabel("a (m/s^2)")
    axs[2].set_xlabel("t (s)")
    axs[2].grid(True)

    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_display_xva'", default='./output_display_xva')
    parser.add_argument('--mass', help="The mass, in kg. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--gamma', help="The friction coefficient, in Ns/m. Default: 0.3", type=float, default=0.3)
    parser.add_argument('--k', help="The spring coefficient, in N/m. Default: 5.0", type=float, default=5.0)
    parser.add_argument('--x0', help="The initial position, in m. Default: 0.3", type=float, default=0.3)
    parser.add_argument('--v0', help="The initial velocity. Default: 2.0", type=float, default=2.0)
    parser.add_argument('--duration', help="The duration of the simulation, in seconds. Default: 3.0", type=float, default=3.0)
    parser.add_argument('--numberOfTimesteps', help="The number of timesteps. Default: 300", type=int, default=300)
    parser.add_argument('--zeroThreshold', help="The x zero threshold. Default: 1.0e-6", type=float, default=1.0e-6)
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.mass,
        args.gamma,
        args.k,
        args.x0,
        args.v0,
        args.duration,
        args.numberOfTimesteps,
        args.zeroThreshold
    )