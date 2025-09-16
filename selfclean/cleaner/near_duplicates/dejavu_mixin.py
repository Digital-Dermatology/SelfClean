import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torchaudio

from ...cleaner.near_duplicates.base_near_duplicate_mixin import BaseNearDuplicateMixin
from ...core.src.utils.plotting import plot_dist


class DejavuNearDuplicateMixin(BaseNearDuplicateMixin):
    """
    Near duplicate detection using Dejavu audio fingerprinting.

    Based on the baseline implementation from external_code/mse-cleaning-audio.
    Uses acoustic fingerprinting to detect similar audio segments.
    """

    def __init__(
        self,
        database_config: dict | None = None,
        audio_length: int = 59049,
        spec_sr: int = 22050,
        segment_strategy: str = "single",
        temp_dir: str | None = None,
        **kwargs,
    ):
        """
        Initialize Dejavu near duplicate mixin.

        Args:
            database_config (dict): Database configuration for Dejavu.
                If None, uses default PostgreSQL config.
            audio_length (int): Length of audio segments for fingerprinting.
            spec_sr (int): Sample rate for audio processing.
            segment_strategy (str): Segmentation strategy ("single", "consecutive", "all").
            temp_dir (str): Temporary directory for audio segments.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.audio_length = audio_length
        self.spec_sr = spec_sr
        self.segment_strategy = segment_strategy

        if database_config is None:
            self.database_config = {
                "database": {
                    "host": "db",  # Use 'db' hostname for Docker compatibility
                    "user": "postgres",
                    "password": "password",
                    "database": "dejavu",
                },
                "database_type": "postgres",
            }
        else:
            self.database_config = database_config

        # Setup temporary directory for audio segments
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="dejavu_segments_"))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect near duplicates using Dejavu audio fingerprinting.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) where lower scores
            indicate more likely near duplicates. Indices are pairs [i, j].
        """
        try:
            # Import Dejavu dependencies (using PyDejavu for Python 3 compatibility)
            from dejavu import Dejavu
            from dejavu.logic.recognizer.file_recognizer import FileRecognizer
        except ImportError as e:
            raise ImportError(
                f"Dejavu dependencies not available: {e}. "
                "Please install PyDejavu and psycopg2-binary packages: "
                "pip install PyDejavu psycopg2-binary"
            ) from e

        # Check if we have audio paths available
        if not hasattr(self, "paths") or self.paths is None:
            raise ValueError(
                "Audio file paths not available. Dejavu requires audio files for fingerprinting."
            )

        # Create audio segments for fingerprinting
        segment_paths = self._create_audio_segments()

        if not segment_paths:
            raise RuntimeError(
                "Dejavu: No valid audio segments created for fingerprinting."
            )

        # Clean database from previous experiments
        # Clean database from previous experiments (required by Dejavu)
        self._clean_database()

        # Process segments with Dejavu
        results_list = []

        # Create Dejavu instance
        djv = Dejavu(self.database_config)

        # Fingerprint all audio segments
        djv.fingerprint_directory(str(self.temp_dir), [".wav"])

        # Recognize each segment against the database
        for segment_path in segment_paths:
            results = djv.recognize(FileRecognizer, str(segment_path))
            if results and "results" in results:
                seg_name = segment_path.stem

                for result in results["results"]:
                    song_name = result["song_name"]
                    confidence = result.get("fingerprinted_confidence", 0.0)

                    # Store all matches (including self-matches) with confidence > 0
                    # The original implementation stores all results and lets the evaluation handle filtering
                    if confidence > 0:
                        results_list.append([seg_name, song_name, confidence])

        # Clean up temporary files
        self._cleanup_segments()

        if not results_list:
            raise RuntimeError(
                "Dejavu: No matches found after fingerprinting and recognition."
            )

        # Convert results to DataFrame and process
        df_results = pd.DataFrame(
            results_list, columns=["nearDup1_id", "nearDup2_id", "scores"]
        )
        processed_results = self._prepare_dejavu_results(df_results)

        if processed_results.empty:
            raise RuntimeError(
                "Dejavu: No valid matches remain after processing results."
            )

        # Convert segment-based matches to sample-based ranking
        scores, indices = self._convert_to_sample_ranking(processed_results)

        if self.plot_distribution and len(scores) > 0:
            plot_dist(
                scores=scores,
                title="Distribution of near-duplicates (Dejavu)",
            )

        return scores, indices

    def _convert_to_sample_ranking(self, df_results):
        """Convert segment-based matches to sample-based ranking with full pairwise ranking."""
        from loguru import logger

        if not hasattr(self, "paths") or self.paths is None:
            raise RuntimeError(
                "Dejavu: Audio file paths not available for sample mapping"
            )

        n_samples = len(self.paths)
        logger.info(
            f"Dejavu: Converting {len(df_results)} segment matches to full sample ranking for {n_samples} samples"
        )

        # Create filename to index mapping
        filename_to_idx = {}
        for idx, path in enumerate(self.paths):
            filename = Path(path).stem
            filename_to_idx[filename] = idx

        # Create confidence matrix for all pairs
        confidence_matrix = np.zeros((n_samples, n_samples))
        match_count_matrix = np.zeros((n_samples, n_samples), dtype=int)

        # Process actual matches from Dejavu
        for _, row in df_results.iterrows():
            file1 = row["nearDup1_id"]
            file2 = row["nearDup2_id"]
            score = row["scores"]

            if file1 in filename_to_idx and file2 in filename_to_idx:
                idx1 = filename_to_idx[file1]
                idx2 = filename_to_idx[file2]

                # Update confidence matrix (symmetric)
                confidence_matrix[idx1, idx2] = max(
                    confidence_matrix[idx1, idx2], score
                )
                confidence_matrix[idx2, idx1] = max(
                    confidence_matrix[idx2, idx1], score
                )

                # Track number of segment matches
                match_count_matrix[idx1, idx2] += 1
                match_count_matrix[idx2, idx1] += 1

        # Create full pairwise ranking
        # Get upper triangle indices (avoid duplicates and self-matches)
        triu_indices = np.triu_indices(n_samples, k=1)

        # Extract confidence scores for all pairs
        confidence_scores = confidence_matrix[triu_indices]

        # Convert confidence to distance (1 - confidence)
        # Pairs with no matches get maximum distance (1.0)
        distance_scores = 1.0 - confidence_scores

        # Create indices array
        indices = np.column_stack([triu_indices[0], triu_indices[1]])

        # Sort by distance (ascending = most similar first)
        sort_order = np.argsort(distance_scores)
        sorted_scores = distance_scores[sort_order]
        sorted_indices = indices[sort_order]

        logger.info(f"Dejavu: Returning full ranking with {len(sorted_scores)} pairs")
        logger.info(
            f"Dejavu: Score range: {sorted_scores.min():.3f} - {sorted_scores.max():.3f}"
        )
        logger.info(
            f"Dejavu: Found confident matches for {np.sum(confidence_scores > 0)} pairs out of {len(sorted_scores)} total pairs"
        )

        return sorted_scores, sorted_indices

    def _create_audio_segments(self) -> list[Path]:
        """
        Create audio segments for fingerprinting based on the specified strategy.

        Returns:
            List of paths to created audio segments.
        """
        segment_paths = []

        for idx, audio_path in enumerate(self.paths):
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Dejavu: Audio file not found: {audio_path}")

            # Load audio
            audio, sample_rate = torchaudio.load(str(audio_path))

            # Resample if needed
            if self.spec_sr != sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sample_rate, new_freq=self.spec_sr
                )
                sample_rate = self.spec_sr

            # Create segments based on strategy
            segments = self._segment_audio(audio, audio_path.stem)

            # Save segments
            for seg_idx, segment in enumerate(segments):
                segment_path = self.temp_dir / f"{audio_path.stem}_{seg_idx}.wav"
                torchaudio.save(str(segment_path), segment, sample_rate)
                segment_paths.append(segment_path)

        return segment_paths

    def _segment_audio(self, audio, base_name):
        """Segment audio based on the specified strategy."""
        audio_samples = audio.shape[-1]

        if self.segment_strategy == "single":
            # Single random segment
            if audio_samples >= self.audio_length:
                start_idx = np.random.randint(0, audio_samples - self.audio_length + 1)
                return [audio[..., start_idx : start_idx + self.audio_length]]
            else:
                return [audio]

        elif self.segment_strategy == "multiple_random":
            # Multiple random segments for better coverage
            if audio_samples >= self.audio_length:
                num_segments = min(
                    3, audio_samples // (self.audio_length // 2)
                )  # Allow overlap
                segments = []
                for _ in range(num_segments):
                    start_idx = np.random.randint(
                        0, audio_samples - self.audio_length + 1
                    )
                    segments.append(
                        audio[..., start_idx : start_idx + self.audio_length]
                    )
                return segments
            else:
                return [audio]

        elif self.segment_strategy == "consecutive":
            # 1-5 consecutive segments (increased from 1-3 for better coverage)
            num_segments = min(5, audio_samples // self.audio_length)
            if num_segments >= 1:
                # Try multiple starting positions to increase match probability
                max_start = max(0, audio_samples - (num_segments * self.audio_length))
                if max_start > 0:
                    start_idx = np.random.randint(0, max_start + 1)
                else:
                    start_idx = 0
                segments = []
                for i in range(num_segments):
                    seg_start = start_idx + (i * self.audio_length)
                    seg_end = min(seg_start + self.audio_length, audio_samples)
                    if (
                        seg_end - seg_start >= self.audio_length // 2
                    ):  # Accept segments at least half length
                        segments.append(audio[..., seg_start:seg_end])
                return segments if segments else [audio]
            else:
                return [audio]

        elif self.segment_strategy == "all":
            # All non-overlapping segments
            num_segments = audio_samples // self.audio_length
            segments = []
            for i in range(num_segments):
                seg_start = i * self.audio_length
                seg_end = seg_start + self.audio_length
                segments.append(audio[..., seg_start:seg_end])
            return segments

        else:
            raise ValueError(f"Unknown segment strategy: {self.segment_strategy}")

    def _prepare_dejavu_results(self, df_results):
        """Process Dejavu results into standardized format."""
        if df_results.empty:
            return df_results

        # Convert bytes to strings if needed
        if df_results["nearDup2_id"].dtype == "object":
            df_results["nearDup2_id"] = df_results["nearDup2_id"].astype(str)

        # Remove all reverse "duplicates"
        # Sort each row to ensure consistent ordering (nearDup1 < nearDup2)
        for idx, row in df_results.iterrows():
            if row["nearDup1_id"] > row["nearDup2_id"]:
                # Swap the values
                (
                    df_results.loc[idx, "nearDup1_id"],
                    df_results.loc[idx, "nearDup2_id"],
                ) = (row["nearDup2_id"], row["nearDup1_id"])

        # Remove duplicates
        df_results = df_results.drop_duplicates(subset=["nearDup1_id", "nearDup2_id"])
        df_results = df_results[["nearDup1_id", "nearDup2_id", "scores"]].copy()

        # Remove all rows where nearDup1_id is the same as nearDup2_id
        df_results = df_results.query("nearDup1_id != nearDup2_id").copy()

        # Convert scores: 1 - scores for SelfClean format
        df_results["scores"] = 1 - df_results["scores"]

        # Sort by lowest score
        df_results = df_results.sort_values("scores").reset_index(drop=True)

        # Extract segment number from id into separate column
        # Handle the case where segment names follow pattern: "filename_segmentnum"
        def extract_segment_info(id_str):
            """Extract base filename and segment number from ID."""
            if "_" in id_str:
                parts = id_str.rsplit("_", 1)  # Split from right, only once
                try:
                    segment_num = int(parts[1])
                    return parts[0], segment_num
                except (ValueError, IndexError):
                    # If parsing fails, use the original ID and segment 0
                    return id_str, 0
            else:
                # No underscore, treat as base filename with segment 0
                return id_str, 0

        # Apply extraction to both columns
        df_results[["nearDup1_id", "nearDup1"]] = pd.DataFrame(
            df_results["nearDup1_id"].apply(extract_segment_info).tolist(),
            index=df_results.index,
        )
        df_results[["nearDup2_id", "nearDup2"]] = pd.DataFrame(
            df_results["nearDup2_id"].apply(extract_segment_info).tolist(),
            index=df_results.index,
        )

        return df_results[
            ["nearDup1", "nearDup1_id", "nearDup2", "nearDup2_id", "scores"]
        ]

    def _clean_database(self):
        """Clean the database from previous experiments."""
        import psycopg2

        conn = psycopg2.connect(**self.database_config["database"])
        cur = conn.cursor()
        try:
            # First, try to create tables if they don't exist
            self._create_database_tables(cur)
            # Then truncate them for a clean start
            cur.execute("TRUNCATE fingerprints, songs CASCADE;")
        except psycopg2.errors.ProgrammingError as e:
            # If tables still can't be created/accessed, raise error
            raise RuntimeError(
                "Dejavu: Failed to initialize or clean database tables (fingerprints, songs)."
            ) from e
        finally:
            conn.commit()
            cur.close()
            conn.close()

    def _create_database_tables(self, cursor):
        """Create Dejavu database tables if they don't exist."""
        # Create songs table
        create_songs_sql = """
        CREATE TABLE IF NOT EXISTS "songs" (
            "song_id" SERIAL,
            "song_name" VARCHAR(250) NOT NULL,
            "fingerprinted" SMALLINT DEFAULT 0,
            "file_sha1" BYTEA,
            "total_hashes" INT NOT NULL DEFAULT 0,
            "date_created" TIMESTAMP NOT NULL DEFAULT now(),
            "date_modified" TIMESTAMP NOT NULL DEFAULT now(),
            CONSTRAINT "pk_songs_song_id" PRIMARY KEY ("song_id"),
            CONSTRAINT "uq_songs_song_id" UNIQUE ("song_id")
        );
        """

        # Create fingerprints table
        create_fingerprints_sql = """
        CREATE TABLE IF NOT EXISTS "fingerprints" (
            "hash" BYTEA NOT NULL,
            "song_id" INT NOT NULL,
            "offset" INT NOT NULL,
            "date_created" TIMESTAMP NOT NULL DEFAULT now(),
            "date_modified" TIMESTAMP NOT NULL DEFAULT now(),
            CONSTRAINT "uq_fingerprints" UNIQUE ("song_id", "offset", "hash"),
            CONSTRAINT "fk_fingerprints_song_id" FOREIGN KEY ("song_id")
                REFERENCES "songs"("song_id") ON DELETE CASCADE
        );
        """

        # Create index for fingerprints table
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS "ix_fingerprints_hash" ON "fingerprints"
        USING hash ("hash");
        """

        cursor.execute(create_songs_sql)
        cursor.execute(create_fingerprints_sql)
        cursor.execute(create_index_sql)

    def _cleanup_segments(self):
        """Clean up temporary audio segments."""
        try:
            import shutil

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
