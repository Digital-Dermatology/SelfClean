from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy
import scipy.stats
from loguru import logger

from ..cleaner.issue_manager import IssueManager
from ..utils.plotting import (
    plot_frac_cut,
    plot_sensitivity,
    subplot_frac_cut,
    subplot_sensitivity,
)


class AutoCleaningMixin:
    def __init__(
        self,
        auto_cleaning: bool = False,
        irrelevant_cut_off: float = 0.01,
        near_duplicate_cut_off: float = 0.01,
        label_error_cut_off: float = 0.01,
        significance_level: float = 0.05,
        cleaner_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.auto_cleaning = auto_cleaning
        self.irrelevant_cut_off = irrelevant_cut_off
        self.near_duplicate_cut_off = near_duplicate_cut_off
        self.label_error_cut_off = label_error_cut_off
        self.significance_level = significance_level
        self.cleaner_kwargs = cleaner_kwargs

    def perform_auto_cleaning(
        self,
        issue_manger: IssueManager,
        return_dict: dict,
        output_path: Optional[Union[str, Path]] = None,
    ):
        if self.auto_cleaning:
            # make sure the significance level is correctly set
            self.cleaner_kwargs["q"] = self.significance_level

            # Near Duplicates
            near_duplicate_issues = issue_manger["near_duplicates"]
            if near_duplicate_issues is not None:
                if output_path is not None:
                    self.cleaner_kwargs["path"] = (
                        f"{output_path.stem}_auto_dups{output_path.suffix}"
                    )
                self.cleaner_kwargs["alpha"] = self.near_duplicate_cut_off
                issues_dup = self.fraction_cut(
                    scores=near_duplicate_issues["scores"],
                    **self.cleaner_kwargs,
                )
                return_dict["near_duplicates"]["auto_issues"] = issues_dup

            # Irrelevant Samples
            irrelevant_issues = issue_manger["irrelevants"]
            if irrelevant_issues is not None:
                if output_path is not None:
                    self.cleaner_kwargs["path"] = (
                        f"{output_path.stem}_auto_oods{output_path.suffix}"
                    )
                self.cleaner_kwargs["alpha"] = self.irrelevant_cut_off
                issues_ood = self.fraction_cut(
                    scores=irrelevant_issues["scores"],
                    **self.cleaner_kwargs,
                )
                return_dict["irrelevants"]["auto_issues"] = issues_ood

            # Label Errors
            label_error_issues = issue_manger["label_errors"]
            if label_error_issues is not None:
                if output_path is not None:
                    self.cleaner_kwargs["path"] = (
                        f"{output_path.stem}_auto_lbls{output_path.suffix}"
                    )
                self.cleaner_kwargs["alpha"] = self.label_error_cut_off
                issues_lbl = self.fraction_cut(
                    scores=label_error_issues["scores"],
                    **self.cleaner_kwargs,
                )
                return_dict["label_errors"]["auto_issues"] = issues_lbl

        return return_dict

    def fraction_cut(
        self,
        scores: np.ndarray,
        alpha: float = 0.01,
        q: float = 0.05,
        dist=scipy.stats.logistic,
        plot_result: bool = False,
        ax=None,
        bins="sqrt",
        path: Optional[str] = None,
    ):
        M = len(scores)
        if M == self.condensed_size:
            # scale alpha for duplicates
            alpha = alpha**2
        # only consider the point in range [0,1]
        _scores = scores[(scores > 0) & (scores < 1)]
        # logit transform
        logit_scores = np.log(_scores / (1 - _scores))

        # calculate the quantiles
        p = alpha
        prob = q * p * self.N / M
        q1 = np.quantile(logit_scores, p)
        q2 = np.quantile(logit_scores, (0.5 * p) ** 0.5)

        # calculate the cut-off
        scale, loc = AutoCleaningMixin.get_scale_loc(
            dist, logit_scores, p, (0.5 * p) ** 0.5
        )
        cutoff = dist.ppf(prob) * scale + loc

        # Exclude the scores below probability threshold
        exclude = logit_scores < cutoff
        n = exclude.sum()
        logger.debug(f"{n} outliers ({n/self.N:.1%})")

        if plot_result:
            if ax is not None:
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
            else:
                plot_frac_cut(
                    dist,
                    logit_scores,
                    bins,
                    q1,
                    q2,
                    cutoff,
                    loc,
                    scale,
                    path,
                )

        return np.where(exclude)[0]

    def threshold_sensitivity(self, scores: np.ndarray, ax=None):
        thresholds = 2 ** np.linspace(-10, -2, 17)
        result = np.array(
            [
                (
                    q,
                    self.fraction_cut(
                        scores=scores,
                        alpha=0.1,
                        q=q,
                        plot_result=False,
                    ).shape[0],
                )
                for q in thresholds
            ]
        )
        result[:, 1] = result[:, 1] / self.N
        if ax is not None:
            subplot_sensitivity(
                ax,
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Significance level $q$",
            )
        else:
            plot_sensitivity(
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Significance level $q$",
            )
        return result

    def alpha_sensitivity(self, scores: np.ndarray, ax=None):
        alphas = 2 ** np.linspace(-10, -2, 17)
        result = np.array(
            [
                (
                    a,
                    self.fraction_cut(
                        scores=scores,
                        alpha=a,
                        plot_result=False,
                    ).shape[0],
                )
                for a in alphas
            ]
        )
        result[:, 1] = result[:, 1] / self.N
        if ax is not None:
            subplot_sensitivity(
                ax,
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Contamination rate guess $\alpha$",
            )
        else:
            plot_sensitivity(
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Contamination rate guess $\alpha$",
            )
        return result

    @staticmethod
    def get_scale_loc(dist, x, q1, q2):
        x1 = np.quantile(x, q1)
        x2 = np.quantile(x, q2)
        y1 = dist.ppf(q1)
        y2 = dist.ppf(q2)
        scale = (x1 - x2) / (y1 - y2)
        loc = (y1 * x2 - y2 * x1) / (y1 - y2)
        return scale, loc
