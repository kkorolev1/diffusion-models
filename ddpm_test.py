#!~/.conda/envs/kakorolev/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import logging
import os

from model.datasets import get_loaders

from model.training import DDPM
from model.utils import plot_images
from model.metrics import fid_score, inception_score

import gc
from tqdm import tqdm

from argparse import ArgumentParser


wandb.config = {
    "dataset": "cifar10",
    "learning_rate": 2e-5,
    "epochs": 2000,
    "batch_size": 128,
    "ema_decay": 0.9999,
    "grad_clip": 1,
    "warmup": 1000,
    "model_path": "bin/cifar10_.pth",
    "epochs_per_save": 15,
    "epochs_per_sample": 15,
    "num_workers": 8
}

logging.basicConfig(
    handlers=[logging.FileHandler("ddpm.log", mode='w'), logging.StreamHandler()],
    level=logging.INFO, 
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
 )

def main(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)
    
    ddpm = DDPM(device)
    ddpm.load("bin/cifar10_1740.pth")
    
    if args.command == 'train':
        wandb.init(project="diffusion", entity="kkorolev", config=wandb.config)
        
        logging.info('Training the model...')
        
        train_loader = get_loaders(wandb.config['dataset'], batch_size=wandb.config['batch_size'], val_ratio=None, train=True)
        
        ddpm.train(train_loader, None)
        
    elif args.command == 'sample':        
        train_loader = get_loaders(wandb.config['dataset'], batch_size=wandb.config['batch_size'], val_ratio=None, train=True)
        num_samples = len(train_loader.dataset)
        logging.info(f'Sampling {num_samples} images...')
            
        batch_images = ddpm.sample(ddpm.ema_sampler, num_samples) 
        torch.save(batch_images, 'fake_data_1740.pt') 
        
        if not os.path.exists(args.output):
            os.mkdir(args.output)
            
        plot_images(batch_images[:128,...], "Sampled images", output_filename=os.path.join(args.output, f"sampled.png"))
    else:
        logging.info(f'Unknown command: {args.command}')


if __name__ == "__main__":
    parser = ArgumentParser(description='DDPM on CIFAR10')
    parser.add_argument('command', help='train,sample,metrics')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument('-lr', type=float, help='Adam learning rate', default=2e-4)
    parser.add_argument('-n', '--samples', type=int, help='Number of samples to generate', default=16)
    parser.add_argument('-p', '--path', help='Path to load/save the model', default='bin/ddpm.pth')
    parser.add_argument('-o', '--output', help='Output directory', default='out')
    parser.add_argument('-c', '--cuda', type=int, help='CUDA device', default=0)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name', default='cifar10')

    args = parser.parse_args()

    main(args)