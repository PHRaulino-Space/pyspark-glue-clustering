"""
Synthetic embedding fixtures for local unit tests.

Generates embeddings clustered around n_topics centroids with Gaussian noise,
so clustering algorithms should recover meaningful structure without any real data.
"""

import random
import uuid


def generate_synthetic_embeddings(
    n_samples: int = 4000,
    n_dims: int = 384,
    n_topics: int = 20,
    seed: int = 42,
) -> list:
    """
    Generate synthetic embeddings clustered around n_topics centroids.

    Each centroid is a random unit vector.  Every sample is a centroid
    plus low-variance Gaussian noise, so the structure is easy to recover
    even with simple KMeans.

    Args:
        n_samples: Total number of embedding records to produce.
        n_dims:    Dimensionality of each embedding vector.
        n_topics:  Number of synthetic topics / true clusters.
        seed:      Random seed for reproducibility.

    Returns:
        List of dicts: {"id": str, "frase": str, "embedding": list[float]}.
    """
    rng = random.Random(seed)

    # Build centroids: n_topics vectors, each l2-normalised
    centroids = []
    for _ in range(n_topics):
        vec = [rng.gauss(0, 1) for _ in range(n_dims)]
        norm = sum(v * v for v in vec) ** 0.5
        centroids.append([v / norm for v in vec])

    records = []
    for i in range(n_samples):
        topic = i % n_topics
        centroid = centroids[topic]
        # Add small Gaussian noise (std=0.05 keeps points close to centroid)
        noisy = [centroid[d] + rng.gauss(0, 0.05) for d in range(n_dims)]
        records.append(
            {
                "id": str(uuid.UUID(int=rng.getrandbits(128))),
                "frase": f"Frase sintética do tópico {topic}, amostra {i}",
                "embedding": noisy,
            }
        )

    return records


def generate_small_synthetic_embeddings(
    n_samples: int = 200,
    n_dims: int = 32,
    n_topics: int = 5,
    seed: int = 7,
) -> list:
    """
    Small, low-dimensional variant for fast unit tests.

    Args:
        n_samples: Total samples (default 200).
        n_dims:    Vector dimensions (default 32).
        n_topics:  Number of topics (default 5).
        seed:      Random seed.

    Returns:
        List of dicts: {"id": str, "frase": str, "embedding": list[float]}.
    """
    return generate_synthetic_embeddings(
        n_samples=n_samples,
        n_dims=n_dims,
        n_topics=n_topics,
        seed=seed,
    )
