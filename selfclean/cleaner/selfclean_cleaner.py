import math
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import scienceplots  # noqa: F401
import sklearn  # noqa: F401
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..cleaner.auto_cleaning_mixin import AutoCleaningMixin
from ..cleaner.base_cleaner import BaseCleaner
from ..cleaner.issue_manager import IssueManager, IssueTypes
from ..cleaner.label_errors.intra_extra_distance_mixin import (
    IntraExtraDistanceLabelErrorMixin,
)
from ..cleaner.near_duplicates.embedding_distance_mixin import EmbeddingDistanceMixin
from ..cleaner.off_topic_samples.lad_mixin import LADOffTopicMixin
from ..distances import *  # noqa: F401, F403
from ..distances.projective_distance import *  # noqa: F401, F403
from ..core.src.utils.logging import set_log_level
from ..core.src.utils.utils import fix_random_seeds
from ..utils.plotting import plot_inspection_result
from ..utils.utils import triu_indices_memmap


class SelfCleanCleaner(
    BaseCleaner,
    LADOffTopicMixin,
    EmbeddingDistanceMixin,
    IntraExtraDistanceLabelErrorMixin,
    AutoCleaningMixin,
):
    def __init__(
        self,
        # distance calculation
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 10_000,
        precision_type_distance: type = np.float32,
        # memory management
        memmap: bool = True,
        memmap_path: Union[Path, str, None] = None,
        approximate_nn: bool = False,
        # plotting
        plot_distribution: bool = False,
        plot_top_N: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        # utils
        random_seed: int = 42,
        # logging
        log_level: str = "INFO",
        **kwargs,
    ):
        self.log_level = log_level
        set_log_level(min_log_level=self.log_level)
        fix_random_seeds(seed=random_seed)

        self.memmap = memmap
        self.approximate_nn = approximate_nn
        self.chunk_size = chunk_size
        self.precision_type_distance = precision_type_distance

        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = Path(self.output_path)

        if memmap_path is None:
            self.memmap_path = Path(tempfile.mkdtemp())
        else:
            self.memmap_path = Path(memmap_path)
            self.memmap_path.parent.mkdir(parents=True, exist_ok=True)

        self.distance_function_name = distance_function_name
        self.distance_function: Callable = eval(
            f"{distance_function_path}{self.distance_function_name}"
        )

        self.plot_distribution = plot_distribution
        self.plot_top_N = plot_top_N
        self.figsize = figsize
        self.is_fitted = False
        super().__init__(**kwargs)

    def fit(
        self,
        emb_space: np.ndarray,
        labels: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        dataset: Optional[Dataset] = None,
        class_labels: Optional[list] = None,
    ):
        self.emb_space = emb_space
        self.labels = labels
        self.dataset = dataset
        self.paths = paths
        self.class_labels = class_labels
        self.N, self.D = emb_space.shape
        self.condensed_size = int(self.N * ((self.N - 1) / 2))
        # Default to "no exact distance arrays" — the approximate path skips
        # them and keeps these as None so downstream code can branch on
        # whether they exist (they're written by the exact path below).
        self.distance_matrix = None
        self.p_distances = None
        logger.info(f"Fitting cleaner on representation space: {emb_space.shape}")

        if self.approximate_nn:
            # KNN-only mode: skip the O(N²) `distance_matrix` and `p_distances`
            # allocations entirely. Build an Annoy index once and cache the
            # K-nearest-neighbour graph + a single-linkage dendrogram derived
            # from its MST. Both stay in the same [0, 1] cosine distance
            # scale the exact path uses, so downstream mixins and AutoCleaning
            # thresholds operate in compatible units.
            self._build_knn_index()
            self.is_fitted = True
            return

        if self.memmap:
            dist_file = self.memmap_path / "dist_matrix.dat"
            if dist_file.exists():
                dist_file.unlink()
            self.distance_matrix = np.memmap(
                str(dist_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.N, self.N),
            )
        else:
            self.distance_matrix = np.zeros(
                shape=(self.N, self.N),
                dtype=self.precision_type_distance,
            )

        # create the distance matrix in chunks
        n_chunks = math.ceil(self.N / self.chunk_size)
        iterator = range(n_chunks)
        for i in (
            tqdm(
                iterator,
                desc="Creating distance matrix",
                total=n_chunks,
                position=0,
                leave=True,
            )
            if self.log_level == "DEBUG"
            else iterator
        ):
            chunk_slice = slice(i * self.chunk_size, (i + 1) * self.chunk_size, 1)
            X_emb = emb_space[chunk_slice]
            distance_row = self.distance_function(
                X=X_emb,
                Y=emb_space,
            )
            distance_row = np.squeeze(distance_row)
            if self.distance_function_name == "cosine_similarity":
                # normalize and invert the cosine similarity to obtain distance
                distance_row = 1 - ((distance_row + 1) / 2)
            self.distance_matrix[chunk_slice, :] = distance_row
            del distance_row
        # clip the values to range [0, 1]
        # could be outside because of floating point inaccuracy
        np.clip(self.distance_matrix, 0.0, 1.0, out=self.distance_matrix)
        # create the condensed matrix
        if self.memmap:
            p_dist_file = self.memmap_path / "p_distances.dat"
            if p_dist_file.exists():
                p_dist_file.unlink()
            self.p_distances = np.memmap(
                str(p_dist_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.condensed_size,),
            )
            triu_indices = triu_indices_memmap(
                str(self.memmap_path / "triu_indices"),
                N=self.N,
                k=1,
            )
        else:
            self.p_distances = np.zeros(
                shape=(self.condensed_size,),
                dtype=self.precision_type_distance,
            )
            triu_indices = np.triu_indices(self.N, k=1)
        # create the upper triangular matrix of the distance matrix
        for start_idx in range(0, len(triu_indices[0]), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(triu_indices[0]))
            self.p_distances[start_idx:end_idx] = self.distance_matrix[
                triu_indices[0][start_idx:end_idx], triu_indices[1][start_idx:end_idx]
            ]
        self.is_fitted = True
        del triu_indices

    def _build_knn_index(self):
        """Build an Annoy index over `self.emb_space` and cache the K-nearest-
        neighbour graph.

        Sets:
            self.knn_index       — the AnnoyIndex (kept around for re-queries)
            self.knn_indices     — int32  array, shape (N, K)
            self.knn_distances   — float  array, shape (N, K), in the SAME
                                   [0, 1] cosine-distance scale as the exact
                                   path's `distance_matrix`.

        Self is excluded from the K neighbours. K = `self.approx_no_neighbors`.
        Annoy's "angular" distance assumes L2-normalised inputs, so we
        normalise here once: for unit vectors, angular² / 4 equals the same
        cosine distance the exact path uses ((1 − cos_sim) / 2).
        """
        from annoy import AnnoyIndex

        K = max(1, int(self.approx_no_neighbors))
        norms = np.linalg.norm(self.emb_space, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        emb_unit = (self.emb_space / norms).astype(np.float32)

        self.knn_index = AnnoyIndex(self.D, "angular")
        for i, x in enumerate(emb_unit):
            self.knn_index.add_item(i, x)
        self.knn_index.build(self.tree_size, n_jobs=-1)

        self.knn_indices = np.zeros((self.N, K), dtype=np.int32)
        self.knn_distances = np.zeros(
            (self.N, K), dtype=self.precision_type_distance
        )
        for i in range(self.N):
            ids, dists = self.knn_index.get_nns_by_item(
                i, K + 1, include_distances=True, search_k=-1
            )
            # Drop self if Annoy returned it (it normally does, with dist≈0).
            if ids and ids[0] == i:
                ids, dists = ids[1:], dists[1:]
            n = min(K, len(ids))
            self.knn_indices[i, :n] = ids[:n]
            # Annoy "angular" -> SelfClean [0, 1] cosine distance.
            self.knn_distances[i, :n] = (
                np.asarray(dists[:n], dtype=np.float32) ** 2
            ) / 4.0
            if n < K:
                # Degenerate (Annoy returned fewer neighbours than requested).
                # Pad with self/dist=1 so downstream rankings stay valid.
                self.knn_indices[i, n:] = i
                self.knn_distances[i, n:] = 1.0
        np.clip(self.knn_distances, 0.0, 1.0, out=self.knn_distances)

        # The LAD off-topic detector needs a single-linkage dendrogram. In
        # exact mode it is `scipy.cluster.hierarchy.single(self.p_distances)`,
        # which we cannot afford here. Build the equivalent dendrogram from
        # the MST of the KNN graph (Kruskal over MST edges = single linkage).
        # For densely-clustered points whose MST edges are all KNN edges the
        # dendrogram is identical to the exact one.
        self._build_knn_linkage()

    def _build_knn_linkage(self):
        """Build a `scipy.cluster.hierarchy`-format linkage matrix from the
        MST of the cached KNN graph.

        Single-linkage clustering is equivalent to Kruskal's algorithm over
        the MST: process edges by ascending weight, union-find merge the two
        endpoints, record one linkage row per merge. Same algorithm LAD
        consumes from `single(self.p_distances)` in the exact path — only
        the input source differs.

        Sets:
            self.knn_linkage_matrix — (N − 1, 4) float64 array.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import (
            connected_components,
            minimum_spanning_tree,
        )

        N, K = self.knn_indices.shape
        rows = np.repeat(np.arange(N, dtype=np.int64), K)
        cols = self.knn_indices.reshape(-1).astype(np.int64)
        data = self.knn_distances.reshape(-1).astype(np.float64)
        valid = rows != cols
        rows, cols, data = rows[valid], cols[valid], data[valid]

        # +epsilon so the sparse encoding does not drop zero-weight edges
        # (planted exact duplicates have cosine distance 0).
        EPS = 1e-12
        adj = coo_matrix((data + EPS, (rows, cols)), shape=(N, N)).tocsr()
        # Symmetrise: cosine is symmetric so both directions agree where
        # both KNN edges exist; max() preserves edges that go only one way.
        adj = adj.maximum(adj.T)

        mst = minimum_spanning_tree(adj).tocoo()
        edges = list(
            zip(
                mst.row.tolist(),
                mst.col.tolist(),
                (mst.data - EPS).tolist(),
            )
        )

        # If the KNN graph is disconnected (components don't reach each
        # other within K hops), bridge them with phantom edges at the max
        # cosine distance (1.0). LAD interprets large merge distances as
        # "isolated", which is the correct semantic for a true component
        # outlier — same approximation HDBSCAN uses for sparse regions.
        n_components, comp_labels = connected_components(adj, directed=False)
        if n_components > 1:
            comp_sizes = np.bincount(comp_labels)
            reps: dict = {}
            for i, c in enumerate(comp_labels):
                reps.setdefault(int(c), int(i))
            comps_sorted = sorted(reps.keys(), key=lambda c: -int(comp_sizes[c]))
            trunk = reps[comps_sorted[0]]
            for c in comps_sorted[1:]:
                edges.append((trunk, reps[c], 1.0))

        edges.sort(key=lambda e: e[2])
        self.knn_linkage_matrix = self._mst_to_linkage(edges, N)

    @staticmethod
    def _mst_to_linkage(edges_sorted_asc, N: int) -> np.ndarray:
        """Kruskal/union-find conversion of MST edges (ascending weight) to a
        scipy `linkage_matrix` (rows: [cluster_a, cluster_b, distance,
        merged_size])."""
        parent = list(range(N))
        rank = [0] * N
        size = [1] * N
        cluster_label = list(range(N))
        next_label = N

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        rows = []
        for i, j, d in edges_sorted_asc:
            ri, rj = find(int(i)), find(int(j))
            if ri == rj:
                continue
            new_size = size[ri] + size[rj]
            rows.append([cluster_label[ri], cluster_label[rj], float(d), new_size])
            if rank[ri] < rank[rj]:
                parent[ri] = rj
                cluster_label[rj] = next_label
                size[rj] = new_size
            else:
                parent[rj] = ri
                cluster_label[ri] = next_label
                size[ri] = new_size
                if rank[ri] == rank[rj]:
                    rank[ri] += 1
            next_label += 1
        return np.asarray(rows, dtype=np.float64)

    def predict(
        self,
        issues_to_detect: List[IssueTypes] = [
            IssueTypes.NEAR_DUPLICATES,
            IssueTypes.OFF_TOPIC_SAMPLES,
            IssueTypes.LABEL_ERRORS,
        ],
    ) -> IssueManager:
        return_dict = {}
        if IssueTypes.NEAR_DUPLICATES in issues_to_detect:
            # Both exact and approximate paths return `(scores, indices)` of
            # the same shape — `get_near_duplicate_ranking` dispatches based on
            # `self.approximate_nn`. The output is always emitted under the
            # standard `near_duplicates` key. (Earlier versions emitted the
            # approximate result under `approx_near_duplicates` as a wide
            # DataFrame; that divergent API has been removed.)
            pred_nd_scores, pred_nd_indices = self.get_near_duplicate_ranking()
            return_dict["near_duplicates"] = {
                "indices": pred_nd_indices,
                "scores": pred_nd_scores,
            }
        if IssueTypes.OFF_TOPIC_SAMPLES in issues_to_detect:
            # LAD runs in both modes — `LADOffTopicMixin.get_off_topic_ranking`
            # picks the right linkage-matrix source (exact: condensed
            # `p_distances`; approximate: MST built over the cached KNN graph,
            # also a valid single-linkage dendrogram).
            pred_ot_scores, pred_ot_indices = self.get_off_topic_ranking()
            return_dict["off_topic_samples"] = {
                "indices": pred_ot_indices,
                "scores": pred_ot_scores,
            }
        if IssueTypes.LABEL_ERRORS in issues_to_detect:
            pred_lbl_errs_scores, pred_lbl_errs_indices = self.get_label_error_ranking()
            if pred_lbl_errs_scores is not None and pred_lbl_errs_indices is not None:
                return_dict["label_errors"] = {
                    "indices": pred_lbl_errs_indices,
                    "scores": pred_lbl_errs_scores,
                }

        if self.labels is not None:
            # transform labels using class names if given
            labels = [
                self.class_labels[x] if self.class_labels is not None else x
                for x in self.labels
            ]
        else:
            labels = self.labels
        # create the manger for the issues to pass to plotting and return
        issue_manager = IssueManager(
            issue_dict=return_dict,
            meta_data_dict={
                "path": self.paths,
                "label": labels,
            },
        )

        if self.plot_top_N is not None and self.dataset is not None:
            plot_inspection_result(
                issue_manger=issue_manager,
                dataset=self.dataset,
                labels=labels,
                plot_top_N=self.plot_top_N,
                output_path=self.output_path,
                figsize=self.figsize,
            )
        return_dict = self.perform_auto_cleaning(
            issue_manger=issue_manager,
            return_dict=return_dict,
            output_path=self.output_path,
        )
        return issue_manager
