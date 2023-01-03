import torch
from tqdm import tqdm
import os

from model.utils import plot_images

def sample(model, n_samples, img_shape, device, step, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with torch.no_grad():
        x = torch.randn(n_samples, *img_shape).to(device)
        plot_images(x, f"t={model.T}", output_filename=os.path.join(output_dir, f"sampled_{model.T}.png"))

        for t in tqdm(range(model.T - 1, -1, -1)):
            t_batch = (torch.ones(n_samples, 1) * t).to(device).long()
            eps_batch = model.backward(x, t_batch)

            alpha_t = model.alphas[t]
            alpha_prod = model.alpha_prods[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_prod).sqrt() * eps_batch)

            if t > 0:
                noise = torch.randn(n_samples, *img_shape).to(device)

                # option 1 from the article for sigma_t
                # both versions from the article give the same result
                sigma_t = model.betas[t].sqrt()

                x = x + sigma_t * noise

            if t % step == 0:
                plot_images(x, f"t={t}", output_filename=os.path.join(output_dir, f"sampled_{t}.png"))


def train(model, dataloader, optimizer, criterion, device, scheduler, n_epochs, start_epoch, model_saver, model_path):
    loss_log = []

    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0

        for img, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            img = img.to(device)

            optimizer.zero_grad()

            eps = torch.randn_like(img).to(device)
            t = torch.randint(model.T, (img.shape[0],)).to(device)
            destroyed_img = model(img, t, eps)

            eps_pred = model.backward(destroyed_img, t)

            loss = criterion(eps, eps_pred)
            epoch_loss += loss.item() * img.shape[0]

            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(epoch_loss)

        print('loss: {:.5f}'.format(epoch_loss))
        model_saver(epoch_loss, epoch + 1, model, optimizer, None, model_path)
        loss_log.append(epoch_loss)

    return loss_log