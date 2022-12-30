import torch
from tqdm import tqdm
import os

from model.utils import plot_images

def sample(model, n_samples, img_shape, device, step):
    if not os.path.exists("results"):
        os.mkdir("results")

    with torch.no_grad():
        x = torch.randn(n_samples, *img_shape).to(device)

        for t in reversed(range(model.T)):
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
                plot_images(x, f"sampled[t={t}]", saved=True)


def train(model, dataloader, optimizer, criterion, device, n_epochs, model_path):
    loss_log = []
    best_loss = float("inf")

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0

        for img, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs} loss {loss_log[-1] if len(loss_log) > 0 else 0}'):
            img = img.to(device)

            optimizer.zero_grad()

            eps = torch.randn_like(img).to(device)
            t = torch.randint(model.T, (img.shape[0],)).to(device)
            destroyed_img = model(img, t, eps)

            eps_pred = model.backward(destroyed_img, t)

            loss = criterion(eps, eps_pred)
            epoch_loss += loss.item() * img.shape[0] / len(dataloader.dataset)

            loss.backward()
            optimizer.step()

            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)

        loss_log.append(epoch_loss)

    return loss_log