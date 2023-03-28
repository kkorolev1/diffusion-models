import torch
import torch.optim as optim

from tqdm.notebook import tqdm
import wandb
import copy
import logging

from model.unet import Unet
from model.ddpm import DiffusionTrainer, DiffusionSampler
from model.checkpoint import ModelCheckpoint

from model.utils import plot_images


def lambda_lr(step):
    return 1 #min(step, wandb.config['warmup']) / wandb.config['warmup']


class DDPM:
    def __init__(self, device):
        self.current_epoch = 0
        self.epochs = wandb.config['epochs']
        
        self.unet = Unet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1).to(device)
        self.ema_unet = copy.deepcopy(self.unet).to(device)
        
        self.trainer = torch.nn.DataParallel(DiffusionTrainer(self.unet).to(device))
        self.sampler = torch.nn.DataParallel(DiffusionSampler(self.unet).to(device))
        self.ema_sampler = torch.nn.DataParallel(DiffusionSampler(self.ema_unet).to(device))
        
        self.optimizer = optim.Adam(self.trainer.parameters(), lr=wandb.config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        
        self.device = device
        
        self.model_checkpoint = ModelCheckpoint(wandb.config['model_path'])
    
    def load(self, model_path=None):
        unet, ema_unet, optimizer, scheduler, epoch, loss = self.model_checkpoint.load(
            self.unet, self.ema_unet, self.optimizer, self.scheduler, model_path
        )
        
        self.unet = unet
        self.ema_unet = ema_unet
        
        self.trainer.model = self.unet
        self.sampler.model = self.unet
        self.ema_sampler.model = self.ema_unet
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_epoch = epoch

        self.optimizer.param_groups[0]['lr'] = wandb.config["learning_rate"]
    
    def ema(self):
        decay = wandb.config["ema_decay"]
        source_dict = self.unet.state_dict()
        target_dict = self.ema_unet.state_dict()
        
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))
    
    def train_epoch(self, dataloader, tqdm_desc):
        self.trainer.train()

        epoch_loss = 0.0

        for img, _ in tqdm(dataloader, desc=tqdm_desc):
            img = img.to(self.device)

            self.optimizer.zero_grad()

            loss = self.trainer(img).mean()
            epoch_loss += loss.item() * img.shape[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), wandb.config["grad_clip"])
            
            self.optimizer.step()
            #self.scheduler.step()
            
            self.ema()

        epoch_loss /= len(dataloader.dataset)
        return epoch_loss

    @torch.no_grad()
    def validate_epoch(self, dataloader, tqdm_desc):
        self.trainer.eval()

        epoch_loss = 0.0

        for img, _ in tqdm(dataloader, desc=tqdm_desc):
            img = img.to(self.device)
            loss = self.trainer(img).mean()
            epoch_loss += loss.item() * img.shape[0]
            
        epoch_loss /= len(dataloader.dataset)
        return epoch_loss
    
    def train(self, train_loader, val_loader, continue_training=False):
        if continue_training:
            self.load_for_training()
            
        for epoch in range(self.current_epoch, self.epochs):
            self.optimizer.param_groups[0]['lr'] = wandb.config["learning_rate"]
            train_loss = self.train_epoch(train_loader, f'Training epoch {epoch+1}/{self.epochs}')
            #val_loss = self.validate_epoch(val_loader, f'Validating epoch {epoch+1}/{self.epochs}')

            lr = self.optimizer.param_groups[0]['lr']
            #lr = self.scheduler._last_lr[0]
            
            logging.info('epoch: {} loss: {:.5f} lr: {}'.format(epoch + 1, train_loss, lr))
            wandb.log({'loss': train_loss})
            
            if (epoch + 1) % wandb.config["epochs_per_save"] == 0:
                self.model_checkpoint.save(train_loss, epoch + 1, self.unet, self.ema_unet, self.optimizer, self.scheduler)
            
            if (epoch + 1) % wandb.config["epochs_per_sample"] == 0:
                sampled_images = self.sample(self.ema_sampler, wandb.config["batch_size"])
                plot_images(sampled_images, f"sampled_epoch_{epoch + 1}", output_filename=f"out/sampled_epoch_{epoch + 1}.png")
                
    @torch.no_grad()
    def sample(self, sampler, n_samples):
        sampler.eval()
        
        img_shape = (3, 32, 32)
        
        sampled_images = []
        batch_size = wandb.config['batch_size']
        
        logging.info(f'Going to sample {n_samples // batch_size} batches')

        for i in range(0, n_samples, batch_size):
            logging.info(f'Sampling #{i // batch_size + 1} batch...')
            size = min(batch_size, n_samples - i)
            x_T = torch.randn(size, *img_shape).to(self.device)
            x_0 = self.sampler(x_T).cpu()
            sampled_images.append(x_0)

            torch.save(torch.cat(sampled_images, dim=0), 'tmp_fake_data.pt')
        
        sampled_images = torch.cat(sampled_images, dim=0)
        return sampled_images