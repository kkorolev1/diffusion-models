import torch
from tqdm import tqdm


@torch.no_grad()
def sample(model, n_samples, img_shape, batch_size, device):
    model.eval()
    slicer = range(0, n_samples, batch_size)
    sampled_images = []

    print(f'Going to sample {n_samples // batch_size} batches')
    
    for i in slicer:
        print(f'Sampling #{i // batch_size + 1} batch...')

        size = min(batch_size, n_samples - i)
        x = torch.randn(size, *img_shape).to(device)

        #if output_dir is not None:
        #    plot_images(x, f"t={model.T}", output_filename=os.path.join(output_dir, f"sampled_{model.T}.png"))

        for t in tqdm(range(model.T - 1, -1, -1)):
            t_batch = torch.full((size,), t).to(device).long()
            eps_batch = model.backward(x, t_batch)

            alpha_t = model.alphas[t]
            alpha_prod = model.alpha_prods[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_prod).sqrt() * eps_batch)

            if t > 0:
                noise = torch.randn_like(x).to(device)

                # option 1 from the article for sigma_t
                # both versions from the article give the same result
                sigma_t = model.betas[t].sqrt()

                x = x + sigma_t * noise

            #if t % step == 0 and output_dir is not None:
            #    plot_images(x, f"t={t}", output_filename=os.path.join(output_dir, f"sampled_{t}.png"))
        sampled_images.append(x)
    
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