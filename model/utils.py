import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_images(images, title, saved=False):
    images = images * 0.3081 + 0.1307
    np_images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(10,10))

    n_cols = int(np.sqrt(len(images)))
    n_rows = len(images) // n_cols
    index = 0

    for i in range(n_rows):
        for j in range(n_cols):
            fig.add_subplot(n_rows, n_cols, index + 1)
            plt.axis('off')
            plt.imshow(np.transpose(np_images[index], (1, 2, 0)), cmap='gray')
            if saved:
                plt.savefig(f"results/{title}.png")
            index += 1
    fig.suptitle(title, fontsize=20)
    plt.show()