import math
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

from ...cleaner.near_duplicates.base_near_duplicate_mixin import BaseNearDuplicateMixin
from ...ssl_library.src.utils.logging import plot_dist
from ...utils.utils import condensed_to_square


class EmbeddingDistanceMixin(BaseNearDuplicateMixin):
    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.memmap:
            score_file = self.memmap_path / "near_duplicate_scores.dat"
            # make sure the files do not exist already
            if score_file.exists():
                score_file.unlink()
            scores_near_dup = np.memmap(
                str(score_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.condensed_size,),
            )
        else:
            scores_near_dup = np.zeros(
                shape=(self.condensed_size,),
                dtype=self.precision_type_distance,
            )

        # sort the values in the condensed matrix
        sorting = self.p_distances.argsort()
        scores_near_dup[:] = np.take(self.p_distances, indices=sorting, axis=0)
        if self.memmap:
            scores_near_dup.flush()
        # vectorize the mapping function
        vec_index_mapping = np.vectorize(condensed_to_square)
        # here the chunk size is x**2 since we have quadratically more
        chunk_size = self.chunk_size**2
        # chunk the sorted values for memory optimization
        n_chunks = math.ceil(self.condensed_size / chunk_size)
        if self.memmap:
            indices_file = self.memmap_path / "near_duplicate_indices.dat"
            # make sure the files do not exist already
            if indices_file.exists():
                indices_file.unlink()
            indices_near_dup = np.memmap(
                str(indices_file),
                dtype=np.int32,
                mode="w+",
                shape=(self.condensed_size, 2),
            )
        else:
            indices_near_dup = np.zeros(
                shape=(self.condensed_size, 2),
                dtype=np.int32,
            )
        for i in tqdm(
            range(n_chunks),
            desc="Processing possible near duplicates",
            total=n_chunks,
            position=0,
            leave=True,
        ):
            chunk_slice = slice(i * chunk_size, (i + 1) * chunk_size, 1)
            chunk_sorting = sorting[chunk_slice]
            # map the indices from the condensed to the redundant distance matrix
            mapping_row = np.asarray(vec_index_mapping(chunk_sorting, self.N)).T
            indices_near_dup[chunk_slice, :] = mapping_row
            if self.memmap:
                indices_near_dup.flush()
            del mapping_row

        if self.plot_distribution:
            plot_dist(
                scores=scores_near_dup,
                title="Distribution of near-duplicates",
            )
        return scores_near_dup, indices_near_dup
