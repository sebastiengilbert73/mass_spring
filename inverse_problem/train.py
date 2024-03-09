import logging
import argparse
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
from torch.utils.data import DataLoader, TensorDataset
import architectures
from statistics import mean
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    datasetFilepath,
    outputDirectory,
    randomSeed,
    batchSize,
    architecture,
    learningRate,
    weightDecay,
    numberOfEpochs,
    displayResults
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    logging.info(f"train.main(); device = {device}; architecture = {architecture}")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the dataset
    dataset_df = pd.read_csv(datasetFilepath)
    signal_length = len(dataset_df.iloc[0]) - 5  # There are 5 output values
    attribute_names = [f"x_{i}" for i in range(signal_length)]
    output_names = ['mass', 'gamma', 'k', 'x0', 'v0']

    X = dataset_df[attribute_names].values  # (N, 301)
    y = dataset_df[output_names].values  # (N, 5)

    # Train-validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=randomSeed)

    # Standardize
    X_std_scaler = StandardScaler()
    X_train_std = X_std_scaler.fit_transform(X_train)
    X_validation_std = X_std_scaler.transform(X_validation)
    dump(X_std_scaler, open(os.path.join(outputDirectory, "X_std_scaler.pkl"), 'wb'))

    y_std_scaler = StandardScaler()
    y_train_std = y_std_scaler.fit_transform(y_train)
    y_validation_std = y_std_scaler.transform(y_validation)
    dump(y_std_scaler, open(os.path.join(outputDirectory, "y_std_scaler.pkl"), 'wb'))

    # Create datasets and dataloaders
    X_train_std_tsr = torch.from_numpy(X_train_std).float()
    y_train_tsr = torch.from_numpy(y_train_std).float()
    train_ds = TensorDataset(X_train_std_tsr, y_train_tsr)

    X_validation_std_tsr = torch.from_numpy(X_validation_std).float()
    y_validation_tsr = torch.from_numpy(y_validation_std).float()
    validation_ds = TensorDataset(X_validation_std_tsr, y_validation_tsr)

    train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=len(validation_ds))

    # Define the neural network
    neural_net = None
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
        raise NotImplemented(f"train.main(): Architecture '{architecture}' is not implemented.")

    neural_net.to(device)

    # Optimization objects
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learningRate, weight_decay=weightDecay)

    training_losses = []
    validation_losses = []
    champion_epoch = None
    champion_loss = 1.0e9
    with open(os.path.join(outputDirectory, 'epochLoss.csv'), 'w') as epoch_loss_file:
        epoch_loss_file.write("epoch,train_loss,validation_loss,is_champion\n")
        for epoch in range(1, numberOfEpochs + 1):
            neural_net.train()
            epoch_losses = []
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(device)
                X_batch = X_batch.unsqueeze(1)  # (B, 1, 301)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                y_pred = neural_net(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            training_loss = mean(epoch_losses)
            training_losses.append(training_loss)

            # Validation
            neural_net.eval()
            epoch_validation_losses = []
            for X_valid, y_valid in validation_dl:  # Should be a single batch
                X_valid = X_valid.to(device)
                X_valid = X_valid.unsqueeze(1)  # (B, 1, 301)
                y_valid = y_valid.to(device)
                y_valid_pred = neural_net(X_valid)
                loss_validation = loss_fn(y_valid_pred, y_valid)
                epoch_validation_losses.append(loss_validation.item())
            validation_loss = mean(epoch_validation_losses)
            validation_losses.append(validation_loss)

            logging.info(f"Epoch {epoch}: training_loss = {training_loss}; validation_loss = {validation_loss}")
            is_champion = False
            if validation_loss < champion_loss:
                champion_loss = validation_loss
                champion_epoch = epoch
                is_champion = True
                logging.info(f" *** Champion validation loss! ***")
                champion_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(neural_net.state_dict(), champion_filepath)

            epoch_loss_file.write(f"{epoch},{training_loss},{validation_loss},{is_champion}\n")

        if displayResults:
            fig, ax = plt.subplots()
            ax.plot(range(1, numberOfEpochs + 1), training_losses, label='Training loss')
            ax.plot(range(1, numberOfEpochs + 1), validation_losses, label='Validation loss')
            plt.annotate('champion', xy=(champion_epoch, champion_loss), xytext=(-50, 50), xycoords='data',
                         arrowprops=dict(arrowstyle="->"), textcoords='offset points')
            ax.legend(loc='upper right')
            plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetFilepath', help="The filepath to the csv file containing the dataset. Default: '../utilities/output_generate_simulations/signal_params.csv'",
                        default="../utilities/output_generate_simulations/signal_params.csv")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train'",
                        default='./output_train')
    parser.add_argument('--randomSeed', help="The random seed. Default: 1", type=int, default=1)
    parser.add_argument('--batchSize', help="The training batch size. Default: 64", type=int, default=64)
    parser.add_argument('--architecture', help="The neural network architecture. Default: Cantilever_32_64_128_0.5", default='Cantilever_32_64_128_0.5')
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.001", type=float, default=0.001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.00001", type=float, default=0.00001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 100", type=int, default=100)
    parser.add_argument('--displayResults', help="Display the loss evolution", action='store_true')
    args = parser.parse_args()

    main(
        args.datasetFilepath,
        args.outputDirectory,
        args.randomSeed,
        args.batchSize,
        args.architecture,
        args.learningRate,
        args.weightDecay,
        args.numberOfEpochs,
        args.displayResults
    )