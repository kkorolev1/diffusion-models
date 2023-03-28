import torch
import numpy as np
import os
import logging


class ModelCheckpoint:
    def __init__(self, model_path, best_val_loss=np.inf):
        self.model_path = model_path
        self.best_val_loss = best_val_loss
    
    def __call__(self, val_loss, epoch, model, ema_model, optimizer, scheduler):
        if val_loss < self.best_val_loss:                
            self.best_val_loss = val_loss
            self.save(val_loss, epoch, model, ema_model, optimizer, scheduler)
    
    def save(self, val_loss, epoch, model, ema_model, optimizer, scheduler):
        model_path, ext = os.path.splitext(self.model_path)
        model_path = f"{model_path}{epoch}{ext}"
            
        if not os.path.exists(os.path.dirname(model_path)):
            os.mkdir(os.path.dirname(model_path))
            
        torch.save({
            'val_loss': val_loss,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, model_path
        )

        logging.info(f'Model is saved in {model_path}')
        logging.info('epoch {} loss {:.5f}'.format(epoch, val_loss))
    
    def load(self, model, ema_model, optimizer, scheduler, model_path=None):
        if model_path is None:
            model_path = self.model_path
        
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']

        logging.info(f'Loaded model from {model_path} after {epoch} epochs')
        print(f'Loaded model from {model_path} after {epoch} epochs')
        
        return model, ema_model, optimizer, scheduler, epoch, val_loss