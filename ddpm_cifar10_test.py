#!/home/kakorolev/.conda/envs/kkorolev/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
import os

from model.datasets import get_CIFAR10
from model.ddpm import DDPM
from model.v2.unet import Unet
from model.training import train, sample
from model.utils import SaveBestModel, load_model, plot_images
from model.metrics import fid_score

import gc


def main(args):
    batch_size = args.batch_size

    print(f'Batch size {batch_size}')

    print('Loading CIFAR10...')
    dataloader = get_CIFAR10(batch_size=batch_size)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    n_epochs = args.epochs
    
    unet = Unet(T=1000, n_channels=128, channels_mul=[1, 2, 2, 2], n_residual_blocks=2, dropout=0.1).to(device)
    model = DDPM(unet, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()

    model_saver = SaveBestModel()
    start_epoch = 0

    # Empty cache
    torch.cuda.empty_cache()
    print("Collected: {}".format(gc.collect()))

    if os.path.exists(args.path):
        loss = float("inf")
        try:
            model, optimizer, scheduler, start_epoch, loss = load_model(model, optimizer, scheduler, args.path)
        except Exception as exc:
            print(f"Cannot load model from {args.path}")
            print(exc)
        model_saver = SaveBestModel(loss)

    if args.command == 'train':
        print('Training the model...')
        loss_log = train(model, dataloader, optimizer, criterion, device, scheduler=None, n_epochs=start_epoch + n_epochs, start_epoch=start_epoch, model_path=args.path, model_saver=model_saver)
    elif args.command == 'sample':
        num_samples = max(args.samples, batch_size)
        print(f'Sampling {num_samples} images...')
        
        img_shape = dataloader.dataset[0][0].shape
        sampled_images = sample(model, num_samples, img_shape, args.batch_size, device, step=args.step)
        
        if not os.path.exists(args.output):
            os.mkdir(args.output)

        plot_images(sampled_images, "Sampled images", output_filename=os.path.join(args.output, f"sampled.png"))

    elif args.command == 'metrics':
        print('Calculating metrics...')
        real_data = []

        for images, _ in dataloader:
            real_data.append(images)
        
        print('Real data is collected')
        real_data = torch.cat(real_data, dim=0)
        fake_data = real_data #sample(models, )
        fid = fid_score(real_data, fake_data)

        print('FID {:.5f}'.format(fid))
    else:
        print(f'Unknown command: {args.command}')


if __name__ == "__main__":
    parser = ArgumentParser(description='DDPM on CIFAR10')
    parser.add_argument('command', help='train,sample,metrics')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('-lr', type=float, help='Adam learning rate', default=2e-4)
    parser.add_argument('-n', '--samples', type=int, help='Number of samples to generate', default=16)
    parser.add_argument('-s', '--step', type=int, help='Step of time during sampling', default=100)
    parser.add_argument('-p', '--path', help='Path to load/save the model', default='bin/ddpm.pth')
    parser.add_argument('-o', '--output', help='Output directory', default='out')
    parser.add_argument('-c', '--cuda', type=int, help='CUDA device', default=0)

    args = parser.parse_args()

    main(args)