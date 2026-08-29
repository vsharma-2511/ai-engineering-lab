"""
Semantic Cache Layer with Upfront Raw Query Lookup, Metadata Guardrails, and Dynamic TTL.
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from Research_Agent.src.config.settings import settings
from Research_Agent.src.schemas.query_schema import SearchParameters


class SemanticCache:

    def __init__(
        self,
        qdrant_path: str = settings.QDRANT_STORAGE_PATH,
        collection_name: str = "research_cache",
        similarity_threshold: float = 0.90,
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "gemini-embedding-001"

        # Explicitly matching 768 dimensions requested in output_dimensionality
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE
                ),
            )

    def close(self):
        """Explicitly close Qdrant connection to prevent teardown warnings."""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def _get_embedding(self, text: str) -> List[float]:
        """Generate 768-dim embedding using Gemini API."""
        response = self.gemini_client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        if hasattr(response, "embedding") and response.embedding:
            return response.embedding.values
        elif hasattr(response, "embeddings") and response.embeddings:
            return response.embeddings[0].values
        raise ValueError(f"Unexpected response format from Gemini embedding call: {response}")

    def lookup_raw_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Upfront vector lookup performed BEFORE query decomposition.
        Uses pure vector similarity + creation timestamp TTL validation.
        Bypasses LLM calls on cache hit.
        """
        query_vector = self._get_embedding(user_query)

        # Vector search in Qdrant without metadata parameter filters
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=3,
            score_threshold=self.similarity_threshold,
        ).points

        for hit in results:
            payload = hit.payload or {}
            extracted_params = payload.get("extracted_params", {})
            created_at = extracted_params.get("created_at", payload.get("created_at", 0))
            is_historical = extracted_params.get("is_historical", False)

            # Check dynamic TTL (24 hours for real-time/latest queries; perpetual for historical)
            if not is_historical and (time.time() - created_at > 86400):
                print(f"⚠️ [CACHE EXPIRED] Hit ID {hit.id} score {hit.score:.4f} exceeded TTL.")
                continue

            print(f"🎯 [UPFRONT CACHE HIT] Score: {hit.score:.4f} | Point ID: {hit.id}")
            return payload.get("response")

        print("❌ [UPFRONT CACHE MISS] Vector match below threshold or expired.")
        return None

    def _is_cache_valid(
        self, cached_params: Dict[str, Any], current_params: SearchParameters
    ) -> bool:
        """Validate exact scope alignment between cached metadata and query parameters."""

        cached_locations = set(cached_params.get("locations", []))
        current_locations = set(current_params.locations)
        if cached_locations != current_locations:
            return False

        cached_indicators = set(cached_params.get("indicators", []))
        current_indicators = set(current_params.indicators)
        if not current_indicators.issubset(cached_indicators):
            return False

        if cached_params.get("timeframe") != current_params.timeframe:
            return False

        is_historical = cached_params.get("is_historical", False)
        created_at = cached_params.get("created_at", 0)
        if not is_historical and (time.time() - created_at > 86400):
            return False

        return True

    def lookup(
        self, user_query: str, extracted_params: SearchParameters
    ) -> Optional[Dict[str, Any]]:
        """Perform vector similarity search + strict metadata parameter verification."""
        query_vector = self._get_embedding(user_query)

        must_filters = []
        if extracted_params.timeframe:
            must_filters.append(
                FieldCondition(
                    key="timeframe",
                    match=MatchValue(value=extracted_params.timeframe),
                )
            )

        search_filter = Filter(must=must_filters) if must_filters else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=3,
            score_threshold=self.similarity_threshold,
        ).points

        for hit in results:
            payload = hit.payload or {}
            if self._is_cache_valid(payload.get("extracted_params", {}), extracted_params):
                print(
                    f"🎯 [PARAM CACHE HIT] Similarity Score: {hit.score:.4f} | ID: {hit.id}"
                )
                return payload.get("response")

        print("❌ [PARAM CACHE MISS] Proceeding to execution pipeline...")
        return None

    def store(
        self,
        user_query: str,
        extracted_params: SearchParameters,
        response: Dict[str, Any],
    ):
        """Store query embedding, metadata parameters, and final response."""
        vector = self._get_embedding(user_query)
        is_historical = (
            extracted_params.timeframe is not None
            and extracted_params.timeframe.lower() != "latest"
        )

        point_id = str(uuid.uuid4())
        payload = {
            "query": user_query,
            "timeframe": extracted_params.timeframe,
            "created_at": time.time(),
            "extracted_params": {
                "locations": extracted_params.locations,
                "indicators": extracted_params.indicators,
                "timeframe": extracted_params.timeframe,
                "target_domains": extracted_params.target_domains,
                "is_historical": is_historical,
                "created_at": time.time(),
            },
            "response": response,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=point_id, vector=vector, payload=payload)
            ],
        )
        print(f"💾 [CACHE STORED] Point ID: {point_id} (Historical: {is_historical})")