from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.patches import Rectangle
from scipy.cluster import hierarchy
from tqdm.auto import tqdm


class LAD:
    def __init__(
        self,
        merged_linkage_matrix: Optional[List[dict]] = None,
        plot_scores: bool = False,
    ):
        self.plot_scores = plot_scores
        if self.plot_scores:
            assert (
                merged_linkage_matrix is not None
            ), "For plotting the result the merged_linkage_matrix is needed."
            n_needed_colors = len(
                [True for x in merged_linkage_matrix if 1 in x["leaves"]]
            )
            self.colors = plt.get_cmap("plasma", n_needed_colors)

    def calc_scores(
        self,
        linkage_matrix: list,
        global_leaves: bool = False,
        save_fig_path: Optional[str] = None,
    ) -> List[Tuple[float, int]]:
        if self.plot_scores:
            plt.figure(figsize=(5, 5))

        # TODO: In need of refactoring
        color_idx = 0
        n_same_text = 1
        leaf_end = None
        leaf_dist = None
        white_spaces = []
        scores = []
        progress_bar = tqdm(desc="Processing possible irrelevant samples")
        rootnode = hierarchy.to_tree(linkage_matrix)

        NodeElement = namedtuple(
            "NodeElement",
            ["node", "start", "end", "dist_parent", "n_leaves"],
        )
        node = NodeElement(
            rootnode,
            start=0.0,
            end=1.0,
            dist_parent=1.0,
            n_leaves=None,
        )
        queue: List[NodeElement] = [node]
        scoring: List[Tuple[float, int]] = []

        while len(queue) > 0:
            # get all the element info from the queue
            node_element = queue.pop(0)
            node = node_element.node
            start = node_element.start
            end = node_element.end
            dist_parent = node_element.dist_parent
            n_leaves = node_element.n_leaves

            (
                white_spaces,
                scores,
                leaf_end,
                leaf_dist,
            ) = self._check_for_unallocated_squares(
                white_spaces=white_spaces,
                scores=scores,
                node=node,
                start=start,
                end=end,
                dist_parent=dist_parent,
                leaf_end=leaf_end,
                leaf_dist=leaf_dist,
                color_idx=color_idx,
            )

            scores, square = self._calculate_area_of_square(
                scores=scores,
                node=node,
                start=start,
                end=end,
                dist_parent=dist_parent,
                color_idx=color_idx,
            )
            if not node.is_leaf():
                # reset the counter
                n_same_text = 1

                # needed for global vs local leaf normalization
                if n_leaves is None:
                    n_leaves = node.count

                # sort according to the min number of leaves
                if node.left.count > node.right.count:
                    _left_node = node.left
                    node.left = node.right
                    node.right = _left_node
                    del _left_node

                n_left_leaves = node.left.count
                n_right_leaves = node.right.count

                # calculate the ratios
                p_left = n_left_leaves / n_leaves
                p_right = n_right_leaves / n_leaves

                # formula: start + (end - start) * p
                w_left = start + (end - start) * p_left
                w_right = start + (end - start) * p_right

                if self.plot_scores:
                    plt.axhline(
                        y=node.dist,
                        xmin=w_left,
                        xmax=w_right,
                        color="dimgray",
                    )
                    plt.axvline(
                        x=w_left,
                        ymin=node.dist,
                        ymax=0,
                        color="dimgray",
                        linestyle="dotted",
                    )
                    plt.axvline(
                        x=w_right,
                        ymin=node.dist,
                        ymax=0,
                        color="dimgray",
                        linestyle="dotted",
                    )

                logger.debug(
                    f"ID: {node.id}, #leaves: {n_leaves}, scores: {len(scores)}, "
                    f"dist: {round(node.dist, 2)}, square: {round(square, 2)}, "
                    f"start: {round(start, 2)}, end: {round(end, 2)}, "
                    f"p_left: {round(p_left, 2)}, p_right: {round(p_right, 2)}, "
                    f"w_left: {round(w_left, 2)}, w_right: {round(w_right, 2)}"
                )

                node_right = NodeElement(
                    node=node.right,
                    start=w_left,
                    end=w_right,
                    dist_parent=node.dist,
                    n_leaves=n_leaves if global_leaves else None,
                )
                queue.insert(0, node_right)

                node_left = NodeElement(
                    node=node.left,
                    start=start,
                    end=w_left,
                    dist_parent=node.dist,
                    n_leaves=n_leaves if global_leaves else None,
                )
                queue.insert(0, node_left)
            else:
                logger.debug(
                    f"Leaf ({node.id}), "
                    f"score: {round(sum(scores), 2)}, square: {round(square, 2)}, "
                    f"start: {round(start, 2)}, end: {round(end, 2)}"
                )

                if self.plot_scores:
                    plt.gca().text(
                        end,
                        -0.05 * n_same_text,
                        f"$w_{node.id}$",
                        color="dimgray",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
                scoring.append((sum(scores), node.id))

                if n_same_text == 1:
                    color_idx += 1
                n_same_text += 1
                leaf_end = end
                leaf_dist = dist_parent

            # END
            progress_bar.update(1)

        if self.plot_scores:
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.ylabel("Distance (d)")
            plt.xlabel("Weight (w)")
            plt.tick_params(axis="x", length=30)
            if save_fig_path is not None:
                plt.savefig(
                    Path(save_fig_path) / "LAD_Visualisation.pdf", bbox_inches="tight"
                )
            plt.show()

        return scoring

    def _check_for_unallocated_squares(
        self,
        white_spaces: list,
        scores: list,
        node: hierarchy.ClusterNode,
        start: float,
        end: float,
        dist_parent: float,
        leaf_end: Optional[float],
        leaf_dist: Optional[float],
        color_idx: int,
    ):
        # add the info for the right possible unused rectangle
        if node.dist > 0.0:
            white_spaces.append(
                {
                    "id": node.id,
                    "start": end,
                    "y_end": dist_parent,
                    "y_start": node.dist,
                }
            )

        # check if there is a white square that needs to be added
        if leaf_end != start and leaf_end is not None:
            square = (start - leaf_end) * leaf_dist
            if square > 0.0:
                scores.append(square)
                if self.plot_scores:
                    plt.gca().add_patch(
                        Rectangle(
                            xy=(leaf_end, 0),
                            width=start - leaf_end,
                            height=leaf_dist,
                            linewidth=0,
                            facecolor=self.colors(color_idx),
                            alpha=0.5,
                        )
                    )
                    plt.gca().text(
                        leaf_end + ((start - leaf_end) / 2),
                        ((leaf_dist) / 2),
                        len(scores),
                        fontsize=6,
                        ha="center",
                        va="center",
                    )
            leaf_end = None
            leaf_dist = None
            # check if there are more white squares
            add_squares = [x for x in white_spaces if x["start"] < start]
            if len(add_squares) > 0:
                for square_info in add_squares:
                    x_start = square_info["start"]
                    y_start = square_info["y_start"]
                    y_end = square_info["y_end"]
                    square = (start - x_start) * (y_end - y_start)
                    if square > 0.0:
                        scores.append(square)
                        if self.plot_scores:
                            plt.gca().add_patch(
                                Rectangle(
                                    xy=(x_start, y_start),
                                    width=start - x_start,
                                    height=y_end - y_start,
                                    linewidth=0,
                                    facecolor=self.colors(color_idx),
                                    alpha=0.5,
                                )
                            )
                            plt.gca().text(
                                x_start + ((start - x_start) / 2),
                                y_start + ((y_end - y_start) / 2),
                                len(scores),
                                fontsize=6,
                                ha="center",
                                va="center",
                            )
            # reset the white square list
            white_spaces = []
        return white_spaces, scores, leaf_end, leaf_dist

    def _calculate_area_of_square(
        self,
        scores: list,
        node: hierarchy.ClusterNode,
        start: float,
        end: float,
        dist_parent: float,
        color_idx: int,
    ):
        # area square: (end - start) * (d_parent - d_node)
        square = (end - start) * (dist_parent - node.dist)
        # update score with the area of the square
        if square > 0.0:
            scores.append(square)
            if self.plot_scores:
                plt.axhline(
                    y=node.dist,
                    xmin=start,
                    xmax=end,
                    color="dimgray",
                    linestyle="dotted",
                )
                plt.gca().add_patch(
                    Rectangle(
                        xy=(start, node.dist),
                        width=end - start,
                        height=dist_parent - node.dist,
                        linewidth=0,
                        facecolor=self.colors(color_idx),
                        alpha=0.5,
                    )
                )
                plt.gca().text(
                    start + ((end - start) / 2),
                    node.dist + ((dist_parent - node.dist) / 2),
                    len(scores),
                    fontsize=6,
                    ha="center",
                    va="center",
                )
        return scores, square
