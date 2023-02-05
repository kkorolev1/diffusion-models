import torch
from tqdm import tqdm


@torch.no_grad()
def sample(model, n_samples, img_shape, batch_size, device):
    model.eval()
    sampled_images = []

    print(f'Going to sample {n_samples // batch_size} batches')
    
    for i in range(0, n_samples, batch_size):
        print(f'Sampling #{i // batch_size + 1} batch...')
        size = min(batch_size, n_samples - i)
        x_T = torch.randn(size, *img_shape).to(device)
        x_0 = model(x_T).cpu()
        sampled_images.append(x_0)
        
        torch.save(torch.cat(sampled_images, dim=0), 'tmp_fake_data.pt')
        
    sampled_images = torch.cat(sampled_images, dim=0)
    return sampled_images

def train_epoch(model, optimizer, dataloader, device, tqdm_desc):
    model.train()

    epoch_loss = 0.0

    for img, _ in tqdm(dataloader, desc=tqdm_desc):
        img = img.to(device)

        optimizer.zero_grad()

        loss = model(img).mean()
        epoch_loss += loss.item() * img.shape[0]

        loss.backward()
        optimizer.step()

    epoch_loss /= len(dataloader.dataset)
    return epoch_loss

@torch.no_grad()
def validate_epoch(model, dataloader, device, tqdm_desc):
    model.eval()

    epoch_loss = 0.0

    for img, _ in tqdm(dataloader, desc=tqdm_desc):
        img = img.to(device)
        loss = model(img).mean()
        epoch_loss += loss.item() * img.shape[0]

    epoch_loss /= len(dataloader.dataset)
    return epoch_loss

def train(model, optimizer, scheduler, train_loader, val_loader, device, n_epochs, start_epoch, model_saver, model_path, writer=None):

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, device, f'Training epoch {epoch+1}/{n_epochs}')
        val_loss = validate_epoch(model, val_loader, device, f'Validating epoch {epoch+1}/{n_epochs}')
        
        if scheduler is not None:
            scheduler.step(val_loss)

        print('train loss: {:.5f}'.format(train_loss))
        print('val loss: {:.5f}'.format(val_loss))

        if writer is not None:
            writer.add_scalar('Train loss', train_loss, epoch + 1)
            writer.add_scalar('Val loss', val_loss, epoch + 1)
            writer.flush()

        model_saver(val_loss, epoch + 1, model, optimizer, scheduler, model_path)