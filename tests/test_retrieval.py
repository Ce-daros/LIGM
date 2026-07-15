import numpy as np

from ligm.retrieval import _top_k


def test_top_k_returns_descending_cosine_neighbors():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    corpus = np.array([[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]], dtype=np.float32)

    assert _top_k(queries, corpus, 2).tolist() == [[1, 2]]
