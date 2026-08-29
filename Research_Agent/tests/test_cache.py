"""
Test runner for SemanticCache layer with Upfront Raw Query Lookup support.
Run with: python -m Research_Agent.tests.test_cache
"""

import time
from qdrant_client.models import Distance, VectorParams
from Research_Agent.src.cache.semantic_cache import SemanticCache
from Research_Agent.src.schemas.query_schema import SearchParameters


def test_semantic_cache():
    cache = SemanticCache(
        qdrant_path="./data/test_cache_db", collection_name="test_collection"
    )

    # Reset collection state to ensure test starts completely clean
    if cache.client.collection_exists(cache.collection_name):
        cache.client.delete_collection(cache.collection_name)
    cache.client.create_collection(
        collection_name=cache.collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    try:
        # 1. Mock query 1 metadata and response
        query_1 = "Find 2021 Census land area and population density for Toronto and Vancouver."
        params_1 = SearchParameters(
            indicators=["population density", "land area"],
            locations=["Toronto", "Vancouver"],
            timeframe="2021",
            target_domains=["statcan.gc.ca"],
        )
        mock_response_1 = {
            "status": "completed",
            "result": "Toronto 2021 population density was 4,427.8 per sq km. Vancouver was 5,749.9 per sq km.",
        }

        # --- Test 1: Cold Upfront Raw Lookup (Expect MISS) ---
        print("\n--- TEST 1: Cold Upfront Raw Lookup ---")
        start_time = time.time()
        raw_hit_cold = cache.lookup_raw_query(query_1)
        latency_ms = (time.time() - start_time) * 1000
        print(f"Upfront Lookup Latency: {latency_ms:.2f} ms")
        assert raw_hit_cold is None

        # --- Test 2: Cold Parameter Lookup (Expect MISS) ---
        print("\n--- TEST 2: Cold Parameter Lookup ---")
        hit_cold = cache.lookup(query_1, params_1)
        assert hit_cold is None

        # --- Store Result in Cache ---
        print("\n--- STORING RESULT ---")
        cache.store(query_1, params_1, mock_response_1)

        # --- Test 3: Upfront Raw Query Lookup on Exact Query (Expect FAST HIT, No Params Needed) ---
        print("\n--- TEST 3: Upfront Raw Query Lookup (Exact Query) ---")
        start_time = time.time()
        raw_hit_exact = cache.lookup_raw_query(query_1)
        latency_ms = (time.time() - start_time) * 1000
        print(f"Upfront Lookup Latency: {latency_ms:.2f} ms")
        assert raw_hit_exact is not None
        assert raw_hit_exact["result"] == mock_response_1["result"]

        # --- Test 4: Upfront Raw Query Lookup on Rephrased Query (Expect FAST HIT) ---
        query_2 = "What were the population density and land area figures for Toronto and Vancouver in the 2021 Census?"
        print("\n--- TEST 4: Upfront Raw Query Lookup (Rephrased Query) ---")
        start_time = time.time()
        raw_hit_rephrased = cache.lookup_raw_query(query_2)
        latency_ms = (time.time() - start_time) * 1000
        print(f"Upfront Rephrased Lookup Latency: {latency_ms:.2f} ms")
        assert raw_hit_rephrased is not None
        assert raw_hit_rephrased["result"] == mock_response_1["result"]

        # --- Test 5: Parameter-Verified Lookup on Rephrased Query (Expect HIT) ---
        print("\n--- TEST 5: Parameter-Verified Lookup ---")
        hit_rephrased = cache.lookup(query_2, params_1)
        assert hit_rephrased is not None

        # --- Test 6: Parameter Mismatch Guardrail (Different Location) (Expect MISS) ---
        print("\n--- TEST 6: Location Mismatch Parameter Guardrail ---")
        params_mismatch = SearchParameters(
            indicators=["population density", "land area"],
            locations=["Toronto", "Montreal"],  # Different location
            timeframe="2021",
            target_domains=["statcan.gc.ca"],
        )
        hit_mismatch = cache.lookup(query_1, params_mismatch)
        assert hit_mismatch is None

        print("\n✅ All semantic cache test assertions passed successfully!")

    finally:
        cache.close()


if __name__ == "__main__":
    test_semantic_cache()