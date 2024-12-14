import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
SEED = 42

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


# These functions are taken for Pehlevan's GitHub:
# https://github.com/Pehlevan-Group/sample_efficient_pop_codes

def sorted_spectral_decomp(resp, imgs=None, plot=False):
    K = 1/resp.shape[1] * resp @ resp.T
    #inds_0 = [i for i in range(len(class_stim)) if class_stim[i] == 0 ]
    #inds_1 = [i for i in range(len(class_stim)) if class_stim[i] == 1 ]
    #inds_sort = inds_0 + inds_1
    #k = K[inds_sort,:]
    if plot:
        plt.imshow(K)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r'image 1', fontsize=20)
        plt.ylabel(r'image 2', fontsize=20)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(fig_dir+ 'kernel_matrix_natural_images.pdf')
        plt.show()
    print(K.shape)
    s,v = np.linalg.eigh(K)
    print(s.shape)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]
    print(s.shape)
    return K,s,v

import seaborn as sns



def plot_top_2_eigenvectors(v, y, fig_name='', title_postfix='', fig_dir='./figures/'):

    plt.rcParams.update({'font.family': 'serif'})
    # Define indices for labels
    pos = y == 1  # Indices where label is +1 (mouse)
    neg = y == -1  # Indices where label is -1 (bird)
    # Create the plot

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=v[pos, 0],
        y=v[pos, 1],
        s=100,
        color=sns.palettes.color_palette('rocket_r')[0],
        label='mouse',
        alpha=0.8
    )
    sns.scatterplot(
        x=v[neg, 0],
        y=v[neg, 1],
        s=100,
        color=sns.palettes.color_palette('rocket_r')[-2],
        label='bird',
        alpha=0.8,
    )

    # Labels and title
    plt.xlabel(r'$\sqrt{\lambda_1} \psi_1$', fontsize=20, fontfamily='serif')
    plt.ylabel(r'$\sqrt{\lambda_2} \psi_2$', fontsize=20, fontfamily='serif')
    plt.title('')

    # Ticks and legend
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=20)

    # Tight layout and save the figure
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_dir + fig_name, format='pdf')
    plt.show()

def plot_power(v, y, fig_name='', title_postfix='', fig_dir='./figures/'):

    coeffs = (v.T @ y)**2

    ck = np.cumsum(coeffs)/np.sum(coeffs)

    # Set plot style
    sns.set(style="white", context="notebook")  # Clean style without grid

    # Plot settings
    plt.figure(figsize=(8, 6))
    plt.plot(
        ck,
        color=sns.color_palette("deep")[0],  # Blue from deep palette
        linewidth=2,  # Line width
    )

    # Axis labels and title
    plt.xlabel(r'$k$', fontsize=20, fontfamily='serif')
    plt.ylabel(r'$C(k)$', fontsize=20, fontfamily='serif')
    plt.title('', fontsize=14, fontfamily='serif')

    # Remove ticks for a cleaner look
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Adjust layout
    plt.tight_layout()

    # Save and display the plot
    if fig_name:
        plt.savefig(fig_dir + fig_name, format='pdf')
    plt.show()

def plot_3d_projected_response(response, fig_name='', title_postfix='', fig_dir='./figures/'):
    np.random.seed(SEED)
    # Assumes response is (N x P)

    # Generate a random 3 x N projection matrix
    W = np.random.randn(3, response.shape[0])

    # Project the responses to 3D space
    Y = W @ response  # Y is a 3 x P matrix

    # Plot the resulting responses as a 3D line plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the coordinates
    x_coords = Y[0, :]
    y_coords = Y[1, :]
    z_coords = Y[2, :]

    # Create the line plot
    ax.plot(x_coords, y_coords, z_coords, linewidth=2, alpha=1)
    
    ax.grid(False)

    # Remove axis labels and title
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    plt.title('', fontsize=16, fontfamily='serif')


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Save and display the plot
    if fig_name:
        plt.savefig(fig_dir + fig_name, format='pdf')

    plt.show()

from matplotlib.colors import LinearSegmentedColormap


def plot_kernel(K, fig_name='', title_postfix='', fig_dir='./figures/'):

    # Set the style and font
    sns.set_theme(style="white", font="serif")

    # Create a truncated colormap for the last 0.8 fraction of the Blues palette
    full_palette = sns.color_palette("Blues_r", as_cmap=True)
    palette = LinearSegmentedColormap.from_list(
        "truncated_blues", full_palette(np.linspace(0, 0.8, 256))
    )

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(K / K.max(), cmap=palette, annot=False, cbar=True)

    # Customize the plot aesthetics
    plt.title('')

    plt.xticks([], fontsize=12)
    plt.yticks([], fontsize=12)

    # Show the plot
    plt.tight_layout()
    # Save and display the plot
    if fig_name:
        plt.savefig(fig_dir + fig_name, format='pdf')
    plt.show()

