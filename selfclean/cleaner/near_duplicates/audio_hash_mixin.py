from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from ...cleaner.near_duplicates.base_near_duplicate_mixin import BaseNearDuplicateMixin
from ...core.src.utils.plotting import plot_dist


class AudioHashNearDuplicateMixin(BaseNearDuplicateMixin):
    """
    Near duplicate detection using audio fingerprinting and hashing.

    This implementation creates perceptual hashes of audio files and compares
    them to find near duplicates. It's more robust than exact matching but
    simpler than complex neural approaches.
    """

    def __init__(
        self,
        hash_method: str = "mfcc",
        hash_size: int = 32,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_mfcc: int = 13,
        segment_duration: float = 30.0,
        similarity_threshold: float = 0.8,
        **kwargs,
    ):
        """
        Initialize Audio Hash near duplicate mixin.

        Args:
            hash_method (str): Hashing method ("mfcc", "chroma", "spectral").
            hash_size (int): Size of the hash vector.
            sample_rate (int): Target sample rate for audio processing.
            hop_length (int): Hop length for feature extraction.
            n_mfcc (int): Number of MFCC coefficients.
            segment_duration (float): Duration of audio segment to analyze (seconds).
            similarity_threshold (float): Threshold for considering files similar.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.hash_method = hash_method
        self.hash_size = hash_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.segment_duration = segment_duration
        self.similarity_threshold = similarity_threshold

    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect near duplicates using audio hashing.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) where lower scores
            indicate more likely near duplicates. Indices are pairs [i, j].
        """
        # Check if we have audio paths available
        if not hasattr(self, "paths") or self.paths is None:
            print(
                "Warning: No audio paths available, falling back to embedding similarity"
            )
            return self._embedding_similarity_fallback()

        # Extract audio hashes for all files
        audio_hashes = self._extract_audio_hashes()

        if len(audio_hashes) == 0:
            print("Warning: No valid audio hashes extracted")
            return np.array([]), np.array([]).reshape(0, 2)

        # Compute pairwise similarities
        similarity_matrix = self._compute_similarity_matrix(audio_hashes)

        # Build full ranking (no thresholding): distance = 1 - similarity for all pairs
        n_files = similarity_matrix.shape[0]
        triu_indices = np.triu_indices(n_files, k=1)
        distances = 1.0 - similarity_matrix[triu_indices]
        order = np.argsort(distances)
        scores = distances[order]
        indices = np.column_stack(
            [
                triu_indices[0][order],
                triu_indices[1][order],
            ]
        )

        if self.plot_distribution and len(scores) > 0:
            plot_dist(
                scores=scores,
                title="Distribution of near-duplicates (Audio Hash)",
            )

        return scores, indices

    def _extract_audio_hashes(self) -> np.ndarray:
        """
        Extract perceptual hashes from all audio files.

        Returns:
            numpy array of shape (n_files, hash_size)
        """
        hashes = []

        for idx, audio_path in enumerate(self.paths):
            try:
                audio_path = Path(audio_path)

                # Handle duplicate dataset filenames
                if not audio_path.exists() and "_duplicate_" in str(audio_path):
                    # Extract original filename from duplicate path
                    # e.g., "5-221950-A-22.wav_duplicate_0" -> "5-221950-A-22.wav"
                    original_name = str(audio_path.name).split("_duplicate_")[0]
                    original_path = audio_path.parent / original_name

                    if original_path.exists():
                        audio_path = original_path
                        print(
                            f"Info: Using original file {original_path} for duplicate {self.paths[idx]}"
                        )
                    else:
                        print(
                            f"Warning: Neither duplicate nor original file found: {audio_path}"
                        )
                        hashes.append(np.zeros(self.hash_size))
                        continue
                elif not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    hashes.append(np.zeros(self.hash_size))
                    continue

                # Load and process audio
                audio_hash = self._compute_audio_hash(audio_path)
                hashes.append(audio_hash)

            except Exception as e:
                print(f"Warning: Could not process audio {audio_path}: {e}")
                # Add zero hash for failed files
                hashes.append(np.zeros(self.hash_size))
                continue

        return np.array(hashes)

    def _compute_audio_hash(self, audio_path: Path) -> np.ndarray:
        """
        Compute perceptual hash for a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Hash vector of size hash_size
        """
        try:
            # Load audio with librosa for better format support
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

            # Take a segment from the middle of the audio
            segment_samples = int(self.segment_duration * self.sample_rate)
            if len(y) > segment_samples:
                start_idx = (len(y) - segment_samples) // 2
                y = y[start_idx : start_idx + segment_samples]

            # Extract features based on method
            if self.hash_method == "mfcc":
                features = librosa.feature.mfcc(
                    y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
                )
            elif self.hash_method == "chroma":
                features = librosa.feature.chroma(
                    y=y, sr=sr, hop_length=self.hop_length
                )
            elif self.hash_method == "spectral":
                features = np.vstack(
                    [
                        librosa.feature.spectral_centroid(
                            y=y, sr=sr, hop_length=self.hop_length
                        ),
                        librosa.feature.spectral_rolloff(
                            y=y, sr=sr, hop_length=self.hop_length
                        ),
                        librosa.feature.zero_crossing_rate(
                            y, hop_length=self.hop_length
                        ),
                    ]
                )
            else:
                raise ValueError(f"Unknown hash method: {self.hash_method}")

            # Create hash by taking statistics across time
            feature_stats = np.concatenate(
                [
                    np.mean(features, axis=1),
                    np.std(features, axis=1),
                    np.median(features, axis=1),
                ]
            )

            # Reduce to target hash size
            if len(feature_stats) > self.hash_size:
                # Use PCA-like dimensionality reduction
                feature_stats = feature_stats[: self.hash_size]
            elif len(feature_stats) < self.hash_size:
                # Pad with zeros
                feature_stats = np.pad(
                    feature_stats, (0, self.hash_size - len(feature_stats))
                )

            # Normalize
            feature_stats = feature_stats / (np.linalg.norm(feature_stats) + 1e-8)

            return feature_stats

        except Exception as e:
            print(f"Warning: Could not compute hash for {audio_path}: {e}")
            return np.zeros(self.hash_size)

    def _compute_similarity_matrix(self, hashes: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for hashes.

        Args:
            hashes: Array of shape (n_files, hash_size)

        Returns:
            Similarity matrix of shape (n_files, n_files)
        """
        # Use cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(hashes)

    def _embedding_similarity_fallback(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to embedding-based similarity when audio paths are not available."""
        if not hasattr(self, "emb_space") or self.emb_space is None:
            return np.array([]), np.array([]).reshape(0, 2)

        # Use cosine similarity on embeddings
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(self.emb_space)
        n_samples = similarity_matrix.shape[0]

        scores_list = []
        indices_list = []

        # Get upper triangular indices
        triu_indices = np.triu_indices(n_samples, k=1)

        for i, j in zip(triu_indices[0], triu_indices[1]):
            similarity = similarity_matrix[i, j]

            # Use a lower threshold for embeddings
            if similarity >= 0.9:  # High similarity threshold for embeddings
                distance = 1.0 - similarity
                scores_list.append(distance)
                indices_list.append([i, j])

        if not scores_list:
            return np.array([]), np.array([]).reshape(0, 2)

        scores = np.array(scores_list)
        indices = np.array(indices_list)

        # Sort by scores and limit to top 1000 to avoid memory issues
        sort_idx = np.argsort(scores)
        top_n = min(1000, len(sort_idx))

        return scores[sort_idx[:top_n]], indices[sort_idx[:top_n]]
