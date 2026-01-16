from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class RerankWeights:
    milvus: float = 1.0
    graph_prior: float = 0.5
    helpful: float = 0.1
    same_category: float = 0.1

class GraphMilvusReranker:
    def __init__(self, weights: Optional[RerankWeights] = None):
        self.w = weights or RerankWeights()

    @staticmethod
    def _coerce_hits(milvus_results: Any) -> List[Any]:
        if not milvus_results:
            return []
        if isinstance(milvus_results, list) and len(milvus_results) > 0 and isinstance(milvus_results[0], list):
            return milvus_results[0]
        if isinstance(milvus_results, list):
            return milvus_results
        return []

    @staticmethod
    def _extract_hit_fields(hit: Any) -> Tuple[float, Dict[str, Any]]:
        if isinstance(hit, dict):
            score = hit.get("score", hit.get("distance", 0.0))
            fields = hit.get("entity", {}) or {}
            return float(score), dict(fields)

        score = getattr(hit, "score", None)
        if score is None:
            score = getattr(hit, "distance", 0.0)

        ent = getattr(hit, "entity", None)
        if isinstance(ent, dict):
            fields = dict(ent)
        else:
            fields = getattr(hit, "fields", {}) or {}

        return float(score), fields

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [0.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _distance_to_similarity(scores: List[float]) -> List[float]:
        return [1.0 / (1.0 + s) for s in scores]

    @staticmethod
    def _solution_node_from_fields(fields: Dict[str, Any]) -> Optional[str]:
        rc = fields.get("resolution_code") or ""
        return f"solution:{rc}" if rc else None

    def rerank(
        self,
        milvus_results: Any,
        graph_solution_priors: Dict[str, float],
        pred_category: Optional[str] = None,
        pred_subcategory: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:

        hits = self._coerce_hits(milvus_results)

        raw_scores: List[float] = []
        hit_rows: List[Tuple[float, Dict[str, Any]]] = []
        for h in hits:
            s, f = self._extract_hit_fields(h)
            raw_scores.append(s)
            hit_rows.append((s, f))

        # treat as distance -> similarity
        sim_scores = self._distance_to_similarity(raw_scores)
        milvus_norm = self._normalize_scores(sim_scores)

        reranked: List[Dict[str, Any]] = []
        for (raw_s, fields), s_norm in zip(hit_rows, milvus_norm):
            ticket_id = fields.get("ticket_id")

            sol_node = self._solution_node_from_fields(fields)
            g_prior = float(graph_solution_priors.get(sol_node, 0.0)) if sol_node else 0.0

            helpful = 1.0 if bool(fields.get("resolution_helpful", False)) else 0.0

            same_cat = 0.0
            if pred_category and fields.get("category") == pred_category:
                same_cat += 0.5
            if pred_subcategory and fields.get("subcategory") == pred_subcategory:
                same_cat += 0.5

            final = (
                self.w.milvus * s_norm
                + self.w.graph_prior * g_prior
                + self.w.helpful * helpful
                + self.w.same_category * same_cat
            )

            reranked.append({
                "ticket_id": ticket_id,
                "final_score": float(final),
                "milvus_score": float(s_norm),
                "graph_prior": float(g_prior),
                "fields": fields,
                "debug": {
                    "milvus_raw": float(raw_s),
                    "solution_node": sol_node,
                    "helpful": helpful,
                    "same_category": same_cat,
                },
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_n]
