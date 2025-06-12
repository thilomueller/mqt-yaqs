import matplotlib.pyplot as plt
import numpy as np
import operator
import pickle

from matplotlib.colors import LogNorm

def plot_bond_heatmaps(
    method1="TEBD",
    method2="TDVP",
    cmap="viridis",
    log_scale=False,
    figsize=(7.0, 2.5)
) -> None:
    """
    Create a Nature Physics style 1×3 figure row with:
      (a) Observable expectation vs. Trotter steps for both TEBD and TDVP
      (b) Combined heatmap (top = TEBD, bottom = TDVP) with annotations
      (c) Memory/compression plot (placeholder)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.titlepad': 2,
        'axes.labelpad': 2,
        'xtick.major.pad': 1,
        'ytick.major.pad': 1
    })

    with open("heisenberg_bonds.pickle", 'rb') as f:
        results = pickle.load(f)['results']

    def extract_obs(method):
        data = [(t, exp_val) for (t, bonds, exp_val) in results[method]]
        data.sort(key=lambda x: x[0])
        return (np.array(x) for x in zip(*data)) if data else (np.array([]), np.array([]))

    ts_tebd, obs_tebd = extract_obs(method1)
    ts_tdvp, obs_tdvp = extract_obs(method2)

    def extract_matrix(method):
        filtered = [e for e in results[method]]
        filtered.sort(key=operator.itemgetter(0))
        return np.vstack([e[1] for e in filtered]) if filtered else np.empty((0, 0))

    mat1 = extract_matrix(method1)
    mat2 = extract_matrix(method2)

    # Shared heatmap scale
    if mat1.size and mat2.size:
        vmin = np.nanmin([np.nanmin(mat1), np.nanmin(mat2)])
        vmax = np.nanmax([np.nanmax(mat1), np.nanmax(mat2)])
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    else:
        norm = None
        vmin = vmax = None

    # ----- Figure setup -----
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.4})
    labels = ['(a)', '(b)', '(c)']

    # Panel (a): expectation vs steps
    ax = axes[0]
    if ts_tebd.size:
        ax.plot(ts_tebd, obs_tebd, marker='^', linestyle='-', label=method1)
    if ts_tdvp.size:
        ax.plot(ts_tdvp, obs_tdvp, marker='o', linestyle='--', label=method2)
    ax.set_xlabel('Trotter steps')
    ax.set_ylabel('Expectation value')
    if log_scale:
        ax.set_yscale('log')
    ax.legend(frameon=False, loc='upper left')
    ax.text(-0.15, 1.05, labels[0], transform=ax.transAxes,
            fontsize=9, fontweight='bold')

    # Panel (b): Combined heatmap
    ax = axes[1]
    mat1_time_x = np.array(mat1.T)
    mat2_time_x = np.array(mat2.T)
    combined = np.vstack([mat2_time_x, mat1_time_x]) if mat1.size and mat2.size else np.empty((0, 0))
    im = ax.imshow(combined, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.axhline(y=mat1_time_x.shape[0] - 0.5, color='white', linewidth=1)
    ax.set_ylabel('Bond index')
    ax.set_xlabel('Trotter steps')
    ax.text(-0.15, 1.05, labels[1], transform=ax.transAxes,
            fontsize=9, fontweight='bold')
    ax.text(0.0125, 0.95, method1, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontweight='bold', color='white')
    ax.text(0.0125, 0.4, method2, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', fontweight='bold', color='white')
    ax.set_xlim(0, 40)
    # Shared colorbar
    cbar_ax = fig.add_axes([0.615, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Bond dim')

    # Panel (c): Memory/Compression placeholder
    ax = axes[2]
    tebd_total_bond = []
    tdvp_total_bond = []
    for i in range(mat1.shape[0]):
        tebd_total_bond.append(np.sum(mat1[i, :]))
        tdvp_total_bond.append(np.sum(mat2[i, :]))
    ax.plot(ts_tebd, tebd_total_bond)
    ax.plot(ts_tdvp, tdvp_total_bond)
    ax.set_xlabel('Trotter steps')
    ax.set_ylabel('Ratio')
    ax.text(-0.15, 1.05, labels[2], transform=ax.transAxes,
            fontsize=9, fontweight='bold')
    ax.text(0.01, 0.95, 'Memory', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontweight='bold')
    ax.set_xticks([])
    # ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.6, 1])
    plt.show()

if __name__ == "__main__":
    plot_bond_heatmaps()