import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.afilm import get_afilm
from models.tfilm import get_tfilm 
from utils import load_h5, save_checkpoint

def make_parser():
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument('--model', default='afilm',
        choices=('afilm', 'tfilm'),
        help='model to train')
    train_parser.add_argument('--train', required=True,
        help='path to h5 archive of training patches')
    train_parser.add_argument('--val', required=True,
        help='path to h5 archive of validation set patches')
    train_parser.add_argument('-e', '--epochs', type=int, default=20,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=16,
        help='training batch size')
    train_parser.add_argument('--logname', default='tmp-run',
        help='folder where logs will be stored')
    train_parser.add_argument('--layers', default=4, type=int,
        help='number of layers in each of the D and U halves of the network')
    train_parser.add_argument('--alg', default='adam',
        help='optimization algorithm')
    train_parser.add_argument('--lr', default=3e-4, type=float,
        help='learning rate')
    train_parser.add_argument('--save_path', default="model.pth",
        help='path to save the model')
    train_parser.add_argument('--r', type=int, default=4, help='upscaling factor')
    train_parser.add_argument('--pool_size', type=int, default=4, help='size of pooling window')
    train_parser.add_argument('--strides', type=int, default=4, help='pooling stide')
    return train_parser


class CustomCheckpoint:
    def __init__(self, file_path):
        self.file_path = file_path

    def on_epoch_end(self, model, optimizer, epoch, logs=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': logs.get('loss', None) if logs else None,
        }, self.file_path)


def train(args):
    X_train, Y_train = load_h5(args.train)
    X_val, Y_val = load_h5(args.val)

    model = get_model(args)

    # Équivalent PyTorch de keras.optimizers.Adam
    if args.alg.lower() == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Optimizer {args.alg} not implemented')

    model_checkpoint_callback = CustomCheckpoint(file_path=args.save_path)

    # Équivalent de model.compile() - définir loss et métriques
    criterion = nn.MSELoss()

    # Équivalent de model.fit() mais avec une boucle d'entraînement manuelle
    # pour rester fidèle à la structure simple de l'original
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Créer les DataLoaders (équivalent des tensors TF)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rmse = 0.0
        num_batches = 0

        for batch_data, batch_target in train_loader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            opt.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            opt.step()

            # Calcul RMSE (équivalent de tf.keras.metrics.RootMeanSquaredError)
            with torch.no_grad():
                rmse = torch.sqrt(criterion(output, batch_target))
                epoch_rmse += rmse.item()

            epoch_loss += loss.item()
            num_batches += 1

        # Logs similaires à TensorFlow
        logs = {
            'loss': epoch_loss / num_batches,
            'root_mean_squared_error': epoch_rmse / num_batches
        }

        # Callback équivalent
        model_checkpoint_callback.on_epoch_end(model, opt, epoch, logs)

        print(f'Epoch {epoch+1}/{args.epochs} - loss: {logs["loss"]:.4f} - rmse: {logs["root_mean_squared_error"]:.4f}')


def get_model(args):
    if args.model == 'tfilm':
        model = get_tfilm(n_layers=args.layers, scale=args.r)
    elif args.model == 'afilm':
        model = get_afilm(n_layers=args.layers, scale=args.r)
    else:
        raise ValueError('Invalid model')
    return model



def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()





