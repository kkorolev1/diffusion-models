import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser

from model.datasets import get_MNIST
from model.ddpm import DDPM
from model.unet import UNet
from model.training import train, sample


def main(args):
    batch_size = args.batch_size

    print('Loading MNIST...')
    dataloader = get_MNIST(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDPM(UNet().to(device), device=device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.command == 'train':
        print('Training the model...')
        loss_log = train(model, dataloader, optimizer, criterion, device, n_epochs=n_epochs, model_path=args.path)
        print("loss: ", loss_log)
    elif args.command == 'sample':
        print('Sampling...')
        model.load_state_dict(torch.load(args.path))
        img_shape = tuple(next(iter(dataloader))[0].shape[1:])
        sample(model, args.samples, img_shape, device, step=args.step)
    else:
        print(f'Unknown command: {args.command}')


if __name__ == "__main__":
    parser = ArgumentParser(description='DDPM on MNIST')
    parser.add_argument('command', help='train or sample')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('-lr', type=float, help='Adam learning rate', default=1e-3)
    parser.add_argument('-n', '--samples', type=int, help='Number of samples to generate', default=16)
    parser.add_argument('-s', '--step', type=int, help='Step of time during sampling', default=100)
    parser.add_argument('-p', '--path', help='Path to load/save the model', default='ddpm.pt')

    args = parser.parse_args()

    main(args)