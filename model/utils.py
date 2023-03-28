import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_images(images, title, output_filename=None):
    images = images * 0.5 + 0.5
    np_images = np.clip(images.numpy(), 0, 1)

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