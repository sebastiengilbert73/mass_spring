import logging
import argparse
import ast
import random
import os
import numpy as np
import sys
sys.path.append("../src/mass_spring")
import simulation

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    randomSeed,
    massRange,
    gammaRange,
    kRange,
    x0Range,
    v0Range,
    duration,
    numberOfTimesteps,
    zeroThreshold,
    noiseSigma,
    numberOfSignals
):
    logging.info("generate_simulations.main()")

    random.seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    timestep = duration/numberOfTimesteps
    ts = np.arange(0, duration + timestep, timestep).tolist()  # [0, 0.01, ..., 3.0]; len(ts) = 301

    with open(os.path.join(outputDirectory, "signal_params.csv"), 'w') as output_file:
        for t_ndx in range(len(ts)):
            output_file.write(f"x_{t_ndx},")
        output_file.write(f"mass,gamma,k,x0,v0\n")

        for example_ndx in range(numberOfSignals):
            mass = random.uniform(*massRange)
            gamma = random.uniform(*gammaRange)
            k = random.uniform(*kRange)
            x0 = random.uniform(*x0Range)
            v0 = random.uniform(*v0Range)
            mass_spring = simulation.MassSpring(
                mass=mass, gamma=gamma, k=k, x0=x0, v0=v0, zero_threshold=zeroThreshold
            )
            xs = [mass_spring.evaluate(t) for t in ts]

            # Add noise
            noise = np.random.normal(scale=noiseSigma, size=len(xs)).tolist()
            xs = [xs[i] + noise[i] for i in range(len(xs))]

            xs_str = [str(x) for x in xs]
            output_file.write(f"{','.join(xs_str)},")
            output_file.write(f"{mass},{gamma},{k},{x0},{v0}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_generate_simulations'",
                        default='./output_generate_simulations')
    parser.add_argument('--randomSeed', help="The random seed. Default: 1", type=int, default=1)
    parser.add_argument('--massRange', help="The range for mass. Default: '[0.01, 1.0]'", default='[0.01, 1.0]')
    parser.add_argument('--gammaRange', help="The range for friction coefficient. Default: '[0.03, 3.0]'", default='[0.03, 3.0]')
    parser.add_argument('--kRange', help="The range for spring coefficient. Default: '[0.5, 50.0]'", default='[0.5, 50.0]')
    parser.add_argument('--x0Range', help="The range for initial position. Default: '[-3.0, 3.0]'", default='[-3.0, 3.0]')
    parser.add_argument('--v0Range', help="The range for initial velocity. Default: '[-20.0, 20.0]'", default='[-20.0, 20.0]')
    parser.add_argument('--duration', help="The duration of the simulation, in seconds. Default: 3.0", type=float, default=3.0)
    parser.add_argument('--numberOfTimesteps', help="The number of timesteps. Default: 300", type=int, default=300)
    parser.add_argument('--zeroThreshold', help="The x zero threshold. Default: 1.0e-6", type=float, default=1.0e-6)
    parser.add_argument('--noiseSigma', help="The additive noise standard deviation. Default: 0.02", type=float, default=0.02)
    parser.add_argument('--numberOfSignals', help="The number of signals to generate. Default: 3000", type=int, default=3000)
    args = parser.parse_args()

    args.massRange = ast.literal_eval(args.massRange)
    args.gammaRange = ast.literal_eval(args.gammaRange)
    args.kRange = ast.literal_eval(args.kRange)
    args.x0Range = ast.literal_eval(args.x0Range)
    args.v0Range = ast.literal_eval(args.v0Range)

    main(
        args.outputDirectory,
        args.randomSeed,
        args.massRange,
        args.gammaRange,
        args.kRange,
        args.x0Range,
        args.v0Range,
        args.duration,
        args.numberOfTimesteps,
        args.zeroThreshold,
        args.noiseSigma,
        args.numberOfSignals
    )