#!/home/kakorolev/.conda/envs/kkorolev/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
import os

from model.datasets import get_MNIST
from model.ddpm import DDPM
from model.v1.unet import Unet
from model.training import train, sample
from model.utils import SaveBestModel, load_model

def main(args):
    batch_size = args.batch_size

    print('Loading MNIST...')
    dataloader = get_MNIST(batch_size=batch_size)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    n_epochs = args.epochs
    model = DDPM(Unet().to(device), device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model_saver = SaveBestModel()
    start_epoch = 0

    if os.path.exists(args.path):
        model, optimizer, start_epoch, loss = load_model(model, optimizer, None, args.path)
        model_saver = SaveBestModel(loss)

    if args.command == 'train':
        print('Training the model...')
        loss_log = train(model, dataloader, optimizer, criterion, device, n_epochs=n_epochs, start_epoch=start_epoch, model_path=args.path, model_saver=model_saver)
    elif args.command == 'sample':
        print('Sampling...')
        img_shape = tuple(next(iter(dataloader))[0].shape[1:])
        sample(model, args.samples, img_shape, device, step=args.step, output_dir=args.output)
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
    parser.add_argument('-p', '--path', help='Path to load/save the model', default='bin/ddpm.pth')
    parser.add_argument('-o', '--output', help='Output directory', default='out')
    parser.add_argument('-c', '--cuda', type=int, help='CUDA device', default=0)

    args = parser.parse_args()

    main(args)