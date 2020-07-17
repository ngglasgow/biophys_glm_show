import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.palplot(sns.cubehelix_palette(10, dark=0.1, light=0.7, reverse=True))

def colors(cmap_name):
    palette = sns.color_palette(cmap_name, 10)
    flip = 1
    x = np.linspace(0, 14, 100)

    sns.palplot(palette)
    sns.set_palette(palette)
    fig, axs = plt.subplots()
    for i in range(10):
        axs.plot(x, np.sin(x + i * 0.5) * (11 - i) * flip)
    fig.suptitle(cmap_name)

# colors('viridis')

bluered10 = ['#08509b', '#2070b4', '#4191c6', '#6aaed6', '#9dcae1', 'k', '#fca082', '#fb694a', '#e32f27', '#b11218']
