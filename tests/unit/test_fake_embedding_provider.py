from app.providers.embeddings.fake_provider import FakeEmbeddingProvider


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are already unit-normalized


def test_same_text_produces_identical_vector():
    provider = FakeEmbeddingProvider(dimensions=64)
    v1 = provider.embed_batch(["hello world"])[0]
    v2 = provider.embed_batch(["hello world"])[0]
    assert v1 == v2


def test_vectors_are_unit_length():
    provider = FakeEmbeddingProvider(dimensions=64)
    v = provider.embed_batch(["some sample text here"])[0]
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_similar_texts_are_closer_than_dissimilar_texts():
    provider = FakeEmbeddingProvider(dimensions=256)
    base = "the cat sat on the mat"
    similar = "the cat sat on a mat"
    different = "quantum physics and relativity theory"

    v_base, v_similar, v_different = provider.embed_batch([base, similar, different])

    assert cosine_similarity(v_base, v_similar) > cosine_similarity(v_base, v_different)


def test_embed_batch_preserves_order_and_length():
    provider = FakeEmbeddingProvider(dimensions=32)
    texts = ["one", "two", "three"]
    vectors = provider.embed_batch(texts)
    assert len(vectors) == 3
    assert all(len(v) == 32 for v in vectors)
