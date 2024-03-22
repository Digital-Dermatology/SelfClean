from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from ..ssl_library.src.utils.logging import create_subtitle, denormalize_image


def plot_inspection_result(
    pred_dups_indices: np.ndarray,
    pred_oods_indices: np.ndarray,
    dataset: Dataset,
    plot_top_N: int,
    pred_lbl_errs_indices: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
):
    rows = 4 if pred_lbl_errs_indices is not None else 3
    fig, ax = plt.subplots(rows, plot_top_N, figsize=figsize)
    for i, (idx1, idx2) in enumerate(pred_dups_indices[:plot_top_N]):
        ax[0, i].imshow(
            transforms.ToPILImage()(denormalize_image(dataset[int(idx1)][0]))
        )
        ax[1, i].imshow(
            transforms.ToPILImage()(denormalize_image(dataset[int(idx2)][0]))
        )
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[0, i].set_title(f"Ranking: {i+1}, Idx: {int(idx1)}", fontsize=6)
        ax[1, i].set_title(f"Idx: {int(idx2)}", fontsize=6)

    for i, idx in enumerate(pred_oods_indices[:plot_top_N]):
        ax[2, i].imshow(
            transforms.ToPILImage()(denormalize_image(dataset[int(idx)][0]))
        )
        ax[2, i].set_title(f"Ranking: {i+1}, Idx: {int(idx)}", fontsize=6)
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])

    if pred_lbl_errs_indices is not None:
        for i, idx in enumerate(pred_lbl_errs_indices[:plot_top_N]):
            class_label = labels[idx] if labels is not None else None
            ax[3, i].imshow(
                transforms.ToPILImage()(denormalize_image(dataset[int(idx)][0]))
            )
            ax[3, i].set_title(
                f"Ranking: {i+1}\nIdx: {int(idx)}\nLbl: {class_label}",
                fontsize=6,
            )
            ax[3, i].set_xticks([])
            ax[3, i].set_yticks([])

    grid = plt.GridSpec(rows, plot_top_N)
    create_subtitle(fig, grid[0, ::], "Near-Duplicate Ranking", fontsize=12)
    create_subtitle(fig, grid[2, ::], "Irrelevant Samples Ranking", fontsize=12)
    if pred_lbl_errs_indices is not None:
        create_subtitle(fig, grid[3, ::], "Label Error Ranking", fontsize=12)
    fig.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_frac_cut(dist, logit_scores, bins, q1, q2, cutoff, loc, scale, path):
    with plt.style.context(["science", "std-colors", "grid"]):
        dist_name = dist.__class__.__name__.split("_")[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        subplot_frac_cut(
            ax,
            logit_scores,
            bins,
            q1,
            q2,
            cutoff,
            dist,
            loc,
            scale,
        )
        ax.legend()

        plt.title(dist_name)
        if path is not None:
            plt.savefig(path, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        plt.figure().clear()
        plt.close("all")
        plt.close()
        plt.cla()
        plt.clf()


def subplot_frac_cut(ax, logit_scores, bins, q1, q2, cutoff, dist, loc, scale):
    ax.axvline(
        x=q1,
        color="green",
        linestyle=":",
        linewidth=1.4,
        label="left-tail range",
    )
    ax.axvline(
        x=q2,
        color="green",
        linestyle=":",
        linewidth=1.4,
    )
    ax.axvspan(q1, q2, alpha=0.5, color="green")
    ax.hist(
        logit_scores,
        bins=bins,
        histtype="step",
        density=True,
        log=True,
        label="scores",
        linewidth=1.4,
    )
    x_grid = np.linspace(cutoff, q2, 101)
    y_grid = dist.pdf((x_grid - loc) / scale) / scale
    ax.plot(x_grid, y_grid, label="distribution fit", color="orange")
    ax.axvline(
        x=cutoff,
        color="orange",
        label="outlier cutoff",
        linestyle="--",
        linewidth=1.4,
    )
    ax.set_ylabel("Probability Density", fontsize=18)
    ax.set_xlabel(r"$\tilde{s}$", fontsize=18)


def plot_sensitivity(result, ylabel: str, xlabel: str):
    with plt.style.context(["science", "std-colors", "grid"]):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        subplot_sensitivity(ax, result, ylabel, xlabel)
        plt.show()
        plt.close(fig)
        plt.figure().clear()
        plt.close("all")
        plt.close()
        plt.cla()
        plt.clf()


def subplot_sensitivity(ax, result, ylabel: str, xlabel: str):
    ax.plot(result[:, 0], result[:, 1], marker="o")
    ax.plot(result[:, 0], result[:, 0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
