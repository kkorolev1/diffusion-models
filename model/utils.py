import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_images(images, title, output_filename=None):
    images = images * 0.5 + 0.5
    np_images = images.detach().cpu().numpy()
    np_images = np.clip(np_images, 0, 1)

    fig = plt.figure(figsize=(10,10))

    n_cols = int(np.sqrt(len(images)))
    n_rows = len(images) // n_cols
    index = 0

    for i in range(n_rows):
        for j in range(n_cols):
            fig.add_subplot(n_rows, n_cols, index + 1)
            plt.axis('off')
            plt.imshow(np.transpose(np_images[index], (1, 2, 0)), cmap='gray')
            index += 1
    
    fig.suptitle(title, fontsize=20)

    if output_filename is not None:
        plt.savefig(output_filename)

    plt.show()

class SaveBestModel:
    def __init__(self, best_val_loss=np.inf):
        self.best_val_loss = best_val_loss
        
    def __call__(self, val_loss, epoch, model, optimizer, scheduler=None, model_path='bin/best_model.pth'):
        if val_loss < self.best_val_loss:
            if not os.path.exists(os.path.dirname(model_path)):
                os.mkdir(os.path.dirname(model_path))
            self.best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else {}
                }, model_path
            )
            print('New best model with loss {:.5f} is saved'.format(val_loss))
    
def load_model(model, optimizer, scheduler=None, model_path='bin/best_model.pth'):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Loaded model from {model_path}')

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return model, optimizer, scheduler, epoch, loss
    
    return model, optimizer, epoch, loss