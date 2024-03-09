import logging
import argparse
import random
import os
import sys
sys.path.append("../src/mass_spring")
import simulation
import numpy as np
import torch
import architectures
import pickle
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    neuralNetworkFilepath,
    XStandardScalerFilepath,
    yStandardScalerFilepath,
    outputDirectory,
    randomSeed,
    mass,
    gamma,
    k,
    x0,
    v0,
    duration,
    numberOfTimesteps,
    zeroThreshold,
    noiseSigma
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    logging.info(f"test_neural_network_predictions.main(); device = {device}")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    np.random.seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)


    # Generate the signal
    timestep = duration / numberOfTimesteps
    ts = np.arange(0, duration + timestep, timestep).tolist()  # [0, 0.01, ..., 3.0]; len(ts) = 301

    mass_spring = simulation.MassSpring(
        mass=mass, gamma=gamma, k=k, x0=x0, v0=v0, zero_threshold=zeroThreshold
    )
    xs = [mass_spring.evaluate(t) for t in ts]

    # Add noise
    noise = np.random.normal(scale=noiseSigma, size=len(xs)).tolist()
    xs = [xs[i] + noise[i] for i in range(len(xs))]

    # Load the neural network
    neural_net = None
    architecture = os.path.splitext(os.path.basename(neuralNetworkFilepath))[0]
    architecture_tokens = architecture.split('_')
    if architecture.startswith('Cantilever'):
        neural_net = architectures.Cantilever(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            hidden_size=int(architecture_tokens[3]),
            dropout_ratio=float(architecture_tokens[4])
        )
    elif architecture.startswith('Balance'):
        neural_net = architectures.Balance(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            hidden_size=int(architecture_tokens[3]),
            dropout_ratio=float(architecture_tokens[4])
        )
    elif architecture.startswith('Coil'):
        neural_net = architectures.Coil(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            hidden_size=int(architecture_tokens[3]),
            dropout_ratio=float(architecture_tokens[4])
        )
    elif architecture.startswith('Leaf'):
        neural_net = architectures.Leaf(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            hidden_size=int(architecture_tokens[3]),
            dropout_ratio=float(architecture_tokens[4])
        )
    elif architecture.startswith('Volute_'):
        neural_net = architectures.Volute(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            hidden_size=int(architecture_tokens[3]),
            dropout_ratio=float(architecture_tokens[4])
        )
    elif architecture.startswith('Vspring_'):
        neural_net = architectures.Vspring(
            number_of_convs1=int(architecture_tokens[1]),
            number_of_convs2=int(architecture_tokens[2]),
            number_of_convs3=int(architecture_tokens[3]),
            number_of_convs4=int(architecture_tokens[4]),
            hidden_size=int(architecture_tokens[5]),
            dropout_ratio=float(architecture_tokens[6]),
        )
    else:
        raise NotImplementedError(f"train.main(): Architecture '{architecture}' is not implemented.")
    neural_net.load_state_dict(torch.load(neuralNetworkFilepath))
    neural_net.to(device)
    neural_net.eval()

    # Scalers
    X_std_scaler = None
    with open(XStandardScalerFilepath, 'rb') as f:
        X_std_scaler = pickle.load(f)
    y_std_scaler = None
    with open(yStandardScalerFilepath, 'rb') as f:
        y_std_scaler = pickle.load(f)

    X = np.expand_dims(xs, axis=0)
    logging.info(f"X.shape = {X.shape}")
    X_std = X_std_scaler.transform(X)

    X_std_tsr = torch.from_numpy(X_std).float().unsqueeze(0).to(device)
    logging.info(f"X_std_tsr.shape = {X_std_tsr.shape}")
    params_pred = neural_net(X_std_tsr)
    # Un-standardize the output
    params_pred = y_std_scaler.inverse_transform(params_pred.cpu().detach().numpy())
    logging.info(f"params_pred = {params_pred}")

    # Simulations
    found_mass_spring = simulation.MassSpring(
        mass=params_pred[0, 0],
        gamma=params_pred[0, 1],
        k=params_pred[0, 2],
        x0=params_pred[0, 3],
        v0=params_pred[0, 4],
        zero_threshold=1e-6
    )
    x_analytical = [found_mass_spring.evaluate(t) for t in ts]

    fig, ax = plt.subplots()
    ax.plot(ts, xs, label="Given signal")
    ax.plot(ts, x_analytical, label='Simulation with the found parameters')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (m)")
    ax.grid(True)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('XStandardScalerFilepath', help="The filepath to the standard scaler for the features")
    parser.add_argument('yStandardScalerFilepath', help="The filepath to the standard scaler for the output values")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_test_neural_network_predictions'",
                        default='./output_test_neural_network_predictions')
    parser.add_argument('--randomSeed', help="The random seed. Default: 1", type=int, default=1)
    parser.add_argument('--mass', help="The mass, in kg. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--gamma', help="The friction coefficient, in Ns/m. Default: 0.3", type=float, default=0.3)
    parser.add_argument('--k', help="The spring coefficient, in N/m. Default: 5.0", type=float, default=5.0)
    parser.add_argument('--x0', help="The initial position, in m. Default: 0.3", type=float, default=0.3)
    parser.add_argument('--v0', help="The initial velocity. Default: 2.0", type=float, default=2.0)
    parser.add_argument('--duration', help="The duration of the simulation, in seconds. Default: 3.0", type=float,
                        default=3.0)
    parser.add_argument('--numberOfTimesteps', help="The number of timesteps. Default: 300", type=int, default=300)
    parser.add_argument('--zeroThreshold', help="The x zero threshold. Default: 1.0e-6", type=float, default=1.0e-6)
    parser.add_argument('--noiseSigma', help="The additive noise standard deviation. Default: 0.02", type=float,
                        default=0.02)

    args = parser.parse_args()
    main(
        args.neuralNetworkFilepath,
        args.XStandardScalerFilepath,
        args.yStandardScalerFilepath,
        args.outputDirectory,
        args.randomSeed,
        args.mass,
        args.gamma,
        args.k,
        args.x0,
        args.v0,
        args.duration,
        args.numberOfTimesteps,
        args.zeroThreshold,
        args.noiseSigma
    )