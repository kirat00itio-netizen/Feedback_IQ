"""
FeedbackIQ - Product Feedback Retrieval Engine (RAG Component).
Provides product-isolated vector retrieval with FAISS and Sentence Transformers,
query intent detection (complaints vs. praises vs. general), and aspect enrichment.

Key Requirements:
1. Product ID isolation: The system MUST NOT retrieve reviews from another product.
2. Negative review prioritization: Prioritize negative reviews when the query is about complaints.
3. Aspect and sentiment metadata: Each retrieved review contains sentiment and domain aspect labels.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import os
import sys

# Ensure Windows OpenMP DLL compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, PROCESSED_REVIEWS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Taxonomies
COMMON_ASPECTS = [
    "packaging",
    "quality",
    "price",
    "delivery",
    "customer_service",
]

ELECTRONICS_SPECIFIC = [
    "battery",
    "performance",
    "compatibility",
    "durability",
]

FASHION_SPECIFIC = [
    "size_fit",
    "material",
    "design",
    "color",
    "durability",
]

DOMAIN_ASPECTS = {
    "Electronics": COMMON_ASPECTS + ELECTRONICS_SPECIFIC,
    "Fashion": COMMON_ASPECTS + FASHION_SPECIFIC,
}

ASPECT_DISPLAY_NAMES = {
    "battery": "Battery & Charging",
    "performance": "Performance & Functionality",
    "compatibility": "Compatibility & Connection",
    "durability": "Durability & Build Quality",
    "size_fit": "Size & Fit",
    "material": "Material & Fabric",
    "design": "Design & Style",
    "color": "Color & Appearance",
    "packaging": "Packaging & Box Condition",
    "quality": "General Product Quality",
    "price": "Price & Value",
    "delivery": "Delivery & Shipping",
    "customer_service": "Customer Service & Returns",
}

# High-precision keyword lexical patterns per aspect
ASPECT_KEYWORD_PATTERNS = {
    "battery": [
        r"\bbatter(y|ies)\b", r"\bcharg(e|er|ing|ed)\b", r"\bpower\b",
        r"\bdrain(s|ed|ing)?\b", r"\bdie(s|d)?\b", r"\brun time\b"
    ],
    "performance": [
        r"\b(slow|lag|laggy|freeze|freez(es|ing)|frozen|crash|crashes|reboot|restart|glitch)\b",
        r"\b(volume|sound|audio|speaker|microphone|hear|loud|quiet)\b",
        r"\b(speed|overheat|hot|unresponsive|buffering|stutter|stops working)\b",
        r"\b(sporadic|unreliable|failed to work)\b"
    ],
    "compatibility": [
        r"\b(compatib(le|ility)|incompatib(le|ility))\b",
        r"\b(connect|connection|disconnect|pairing|pair|bluetooth|wifi|wi-fi|sync|network)\b",
        r"\b(hdmi|port|cable|adapter|usb|aux|jack|plug)\b",
        r"\b(app|application|ios|android|mac|windows|tv|pc|iphone|ipad)\b",
        r"\b(recogniz(e|ed)|detect(ed)?)\b"
    ],
    "durability": [
        r"\b(durab(le|ility)|flimsy|sturdy|broke(n)?|snapped|crack(ed)?|shatter(ed)?)\b",
        r"\b(died after|lasted|stopped working after|wear and tear|fell apart)\b",
        r"\b(tear|tore|torn|rip|ripped|hole|fray(ed)?|seam split|unravel)\b"
    ],
    "size_fit": [
        r"\b(size|sizing|fit|fits|fitted|tight|loose|small|smaller|large|larger|big|baggy)\b",
        r"\b(length|waist|chest|inseam|shrink|shrunk|shrinking|true to size)\b",
        r"\b(too short|too long|narrow|wide)\b"
    ],
    "material": [
        r"\b(material|fabric|cotton|polyester|wool|linen|silk|leather|canvas)\b",
        r"\b(thin|see through|sheer|rough|scratchy|itchy|soft|softness|breathab(le|ility))\b",
        r"\b(synthetic|blend|elastic|stretch(y)?)\b"
    ],
    "design": [
        r"\b(design|style|look|looks|cute|ugly|cut|pattern|print|graphic)\b",
        r"\b(pocket(s)?|zipper|button(s)?|seam|stitch(ing)?|collar|sleeve(s)?)\b",
        r"\b(shape|appearance|aesthetic)\b"
    ],
    "color": [
        r"\b(color|shade|fade(d)?|fading|dye|bleeding)\b",
        r"\b(black|white|blue|red|green|yellow|pink|grey|gray|dark|bright)\b",
        r"\b(washed out|discolor(ed)?|not the right color|wrong color)\b"
    ],
    "packaging": [
        r"\b(packag(e|ing|ed)|box|boxed|bubble wrap|envelope|damaged box|seal(ed)?)\b",
        r"\b(opened box|crushed box|unsealed)\b"
    ],
    "quality": [
        r"\b(quality|cheap(ly)?|poor quality|low quality|junk|garbage|trash|terrible)\b",
        r"\b(defective|defect|poorly made|subpar|waste of money|fake|counterfeit)\b",
        r"\b(horrible|awful|disaster|shoddy)\b"
    ],
    "price": [
        r"\b(price|pricing|cost|expensive|overpriced|cheap|worth|money|dollar|cent)\b",
        r"\b(value for money|waste of money|overcharged|deal|rip off|ripoff)\b"
    ],
    "delivery": [
        r"\b(deliver(y|ed)?|ship(ped|ping)?|arriv(e|ed|ing)|late|delayed|delay)\b",
        r"\b(transit|tracking|lost in mail|carrier|on time)\b"
    ],
    "customer_service": [
        r"\b(customer service|support|help desk|agent|representative)\b",
        r"\b(return(ed|ing)?|refund(ed)?|exchange|warranty|guarantee|seller)\b",
        r"\b(contacted|no response|refused)\b"
    ],
}

# Compiled regex patterns for fast matching
COMPILED_PATTERNS = {
    aspect: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for aspect, patterns in ASPECT_KEYWORD_PATTERNS.items()
}

# Query intent keyword signals
COMPLAINT_SIGNALS = [
    "unhappy", "complaint", "complaints", "issue", "issues", "problem", "problems",
    "worst", "hate", "hated", "bad", "drawback", "drawbacks", "poor", "fault",
    "defect", "defects", "fail", "fails", "failed", "broken", "disappointed",
    "disappointing", "negative", "sucks", "terrible", "awful", "return", "returned",
    "useless", "flaw", "flaws", "trouble", "warning", "wrong", "annoying", "annoyed"
]

PRAISE_SIGNALS = [
    "like", "likes", "liked", "love", "loves", "loved", "best", "great", "good",
    "praise", "praises", "recommend", "recommended", "pros", "advantage",
    "advantages", "favorite", "positive", "enjoy", "enjoyed", "awesome",
    "excellent", "happy", "impressive", "satisfied", "pleased"
]


def extract_aspects_from_text(text: str, domain: str = "Electronics") -> List[str]:
    """
    Extract relevant taxonomy aspects mentioned in review text using domain constraints
    and pattern matching.
    """
    clean_domain = domain.capitalize() if domain else "Electronics"
    valid_aspects = DOMAIN_ASPECTS.get(clean_domain, COMMON_ASPECTS + ELECTRONICS_SPECIFIC)
    
    text_lower = text.lower()
    detected = []
    
    for aspect in valid_aspects:
        patterns = COMPILED_PATTERNS.get(aspect, [])
        for pat in patterns:
            if pat.search(text_lower):
                detected.append(aspect)
                break
                
    # If no specific aspect is triggered but negative words exist, map to general quality
    if not detected and any(w in text_lower for w in ["bad", "terrible", "awful", "horrible", "junk", "poor"]):
        detected.append("quality")
        
    return detected


class ReviewRetriever:
    """
    Product-Isolated Semantic Review Retriever powered by Sentence Transformers and FAISS.
    Guarantees strict product isolation (no cross-product review contamination).
    """

    def __init__(
        self,
        reviews_path: Optional[Path] = None,
        products_path: Optional[Path] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.reviews_path = reviews_path or (PROCESSED_DATA_DIR / "reviews_with_predictions.parquet")
        if not self.reviews_path.exists():
            self.reviews_path = PROCESSED_REVIEWS_FILE

        self.products_path = products_path or (PROCESSED_DATA_DIR / "product_insights.parquet")
        self.embedding_model_name = embedding_model_name
        
        logger.info("Initializing SentenceTransformer '%s'...", embedding_model_name)
        self.encoder = SentenceTransformer(embedding_model_name, device=device)
        if hasattr(self.encoder, "get_embedding_dimension"):
            self.embedding_dim = self.encoder.get_embedding_dimension()
        else:
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Load datasets
        self.df_reviews = self._load_reviews()
        self.df_products = self._load_products()

        # In-memory index cache: {parent_asin: {"index": faiss.Index, "df": pd.DataFrame, "embeddings": np.ndarray}}
        self._product_indices: Dict[str, Dict[str, Any]] = {}

    def _load_reviews(self) -> pd.DataFrame:
        """Load and prepare reviews dataframe with combined text and aspect annotations."""
        # Use cache file with aspects pre-computed — instant load on every run after the first
        cache_path = self.reviews_path.parent / (self.reviews_path.stem + "_with_aspects.parquet")
        if cache_path.exists():
            logger.info("Loading cached reviews from '%s' (instant)...", cache_path.name)
            return pd.read_parquet(cache_path)

        logger.info("Loading reviews from '%s'...", self.reviews_path.name)
        df = pd.read_parquet(self.reviews_path)

        # Prepare combined text for semantic embedding
        titles = df["title"].fillna("").astype(str).str.strip()
        texts = df["text"].fillna("").astype(str).str.strip()
        df["combined_text"] = titles.where(titles == "", titles + ". ") + texts

        # Ensure predicted_sentiment column exists
        if "predicted_sentiment" not in df.columns:
            logger.info("'predicted_sentiment' column not found — using 'sentiment' as fallback.")
            df["predicted_sentiment"] = df["sentiment"]

        # Fast numpy-based vectorized aspect extraction
        if "aspects" not in df.columns:
            logger.info("Running aspect extraction on %d reviews (one-time, will cache to disk)...", len(df))
            df["aspects"] = self._extract_aspects_vectorized(df)
            logger.info("Saving aspect cache to '%s'...", cache_path.name)
            df.to_parquet(cache_path, index=False)
            logger.info("Cache saved — next startup will be instant.")

        return df

    @staticmethod
    def _extract_aspects_vectorized(df: pd.DataFrame) -> pd.Series:
        """
        Fast numpy boolean matrix approach — no per-row Python loop over 92k rows.
        Each domain group computes a (n_rows x n_aspects) bool matrix via str.contains,
        then converts rows to lists in a single vectorized pass.
        """
        import numpy as np

        n = len(df)
        result = [[] for _ in range(n)]
        pos_array = np.arange(n)

        for domain, valid_aspects in DOMAIN_ASPECTS.items():
            domain_col = df.get("domain", pd.Series("Electronics", index=df.index))
            domain_mask = domain_col.str.capitalize() == domain
            if not domain_mask.any():
                continue

            local_positions = pos_array[domain_mask.values]
            sub_text = df.loc[domain_mask, "combined_text"].str.lower().fillna("")

            # Build one boolean array per aspect via vectorized str.contains
            hit_cols, aspect_names = [], []
            import warnings
            for aspect in valid_aspects:
                patterns = COMPILED_PATTERNS.get(aspect, [])
                if not patterns:
                    continue
                combined_pat = "|".join(f"(?:{p.pattern})" for p in patterns)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    hit_cols.append(sub_text.str.contains(combined_pat, case=False, regex=True, na=False).values)
                aspect_names.append(aspect)

            if not hit_cols:
                continue

            # Stack into 2D bool matrix: (n_domain_rows, n_aspects)
            hit_matrix = np.column_stack(hit_cols)

            # Vectorized negative-word fallback mask
            neg_pat = r"\b(bad|terrible|awful|horrible|junk|poor)\b"
            has_neg = sub_text.str.contains(neg_pat, case=False, regex=True, na=False).values

            # Convert matrix rows to aspect lists
            for local_i, global_i in enumerate(local_positions):
                row = hit_matrix[local_i]
                detected = [aspect_names[j] for j in range(len(aspect_names)) if row[j]]
                if not detected and has_neg[local_i]:
                    detected = ["quality"]
                result[global_i] = detected

        return pd.Series(result, index=df.index, dtype=object)


    def _load_products(self) -> pd.DataFrame:
        """Load product metadata or insights table."""
        if self.products_path.exists():
            return pd.read_parquet(self.products_path)
        prod_meta_path = PROCESSED_DATA_DIR / "products.parquet"
        if prod_meta_path.exists():
            return pd.read_parquet(prod_meta_path)
        return pd.DataFrame()

    def _build_product_catalog(self) -> None:
        """Vectorized computation of product metadata and summaries (sub-second for 70k+ products)."""
        if hasattr(self, "_cached_available_products") and self._cached_available_products is not None:
            return

        df = self.df_reviews
        agg = df.groupby(["parent_asin", "domain"]).agg(
            total_reviews=("rating", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index()

        pos_map = (df["sentiment"] == "positive").groupby(df["parent_asin"]).sum()
        neu_map = (df["sentiment"] == "neutral").groupby(df["parent_asin"]).sum()
        neg_map = (df["sentiment"] == "negative").groupby(df["parent_asin"]).sum()

        agg["pos_count"] = agg["parent_asin"].map(pos_map).fillna(0).astype(int)
        agg["neu_count"] = agg["parent_asin"].map(neu_map).fillna(0).astype(int)
        agg["neg_count"] = agg["parent_asin"].map(neg_map).fillna(0).astype(int)
        agg = agg.sort_values("total_reviews", ascending=False)

        title_lookup = {}
        if not self.df_products.empty and "parent_asin" in self.df_products.columns and "title" in self.df_products.columns:
            title_lookup = dict(zip(self.df_products["parent_asin"], self.df_products["title"]))

        result = []
        info_cache = {}
        for row in agg.itertuples(index=False):
            asin = str(row.parent_asin)
            tot = int(row.total_reviews)
            title = str(title_lookup.get(asin, f"Product {asin}"))
            pos_c = int(row.pos_count)
            neu_c = int(row.neu_count)
            neg_c = int(row.neg_count)
            info = {
                "parent_asin": asin,
                "title": title,
                "domain": str(row.domain),
                "total_reviews": tot,
                "average_rating": round(float(row.avg_rating), 2),
                "sentiment_counts": {
                    "positive": pos_c,
                    "neutral": neu_c,
                    "negative": neg_c,
                },
                "sentiment_percentages": {
                    "positive": round((pos_c / tot) * 100, 1) if tot > 0 else 0.0,
                    "neutral": round((neu_c / tot) * 100, 1) if tot > 0 else 0.0,
                    "negative": round((neg_c / tot) * 100, 1) if tot > 0 else 0.0,
                },
            }
            result.append(info)
            info_cache[asin] = info

        self._cached_available_products = result
        self._product_info_cache = info_cache

    def get_available_products(self) -> List[Dict[str, Any]]:
        """Return list of all unique products available for queries, sorted by review count."""
        if not hasattr(self, "_cached_available_products") or self._cached_available_products is None:
            self._build_product_catalog()
        return self._cached_available_products

    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Get product summary metadata in O(1) time."""
        if not hasattr(self, "_product_info_cache") or self._product_info_cache is None:
            self._build_product_catalog()

        if product_id in self._product_info_cache:
            return self._product_info_cache[product_id]

        raise ValueError(f"Product '{product_id}' not found in reviews database.")

    def detect_query_intent(self, query: str) -> str:
        """
        Classifies user query intent:
        - 'complaints': Questions about issues, unhappy customers, flaws, negatives
        - 'praises': Questions about likes, best features, positive reviews
        - 'general': Informational questions about specific aspects or overall sentiment
        """
        query_lower = query.lower()
        
        has_complaint_signal = any(
            re.search(rf"\b{re.escape(w)}\b", query_lower) for w in COMPLAINT_SIGNALS
        )
        has_praise_signal = any(
            re.search(rf"\b{re.escape(w)}\b", query_lower) for w in PRAISE_SIGNALS
        )

        if has_complaint_signal and not has_praise_signal:
            return "complaints"
        elif has_praise_signal and not has_complaint_signal:
            return "praises"
        return "general"

    def _get_or_create_product_index(self, product_id: str) -> Dict[str, Any]:
        """
        Builds or retrieves an isolated FAISS index containing ONLY reviews for product_id.
        Prevents any potential cross-product review leakage.
        """
        if product_id in self._product_indices:
            return self._product_indices[product_id]

        prod_df = self.df_reviews[self.df_reviews["parent_asin"] == product_id].copy().reset_index(drop=True)
        if prod_df.empty:
            raise ValueError(f"Product '{product_id}' has no reviews available.")

        logger.info(
            "Building isolated FAISS index for product '%s' (%d reviews)...",
            product_id,
            len(prod_df),
        )

        # Generate L2-normalized embeddings for inner product (cosine similarity) search
        texts = prod_df["combined_text"].tolist()
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        cached_data = {
            "index": index,
            "df": prod_df,
            "embeddings": embeddings,
        }
        self._product_indices[product_id] = cached_data
        return cached_data

    def retrieve_reviews(
        self,
        product_id: str,
        query: str,
        top_k: int = 5,
        prioritize_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve the top relevant reviews for a specific product.

        Parameters:
            product_id: Target product parent_asin.
            query: User's feedback question.
            top_k: Number of relevant reviews to retrieve.
            prioritize_sentiment: If True, prioritizes negative reviews for complaints
                                  and positive reviews for praise queries.

        Returns:
            Dict containing retrieved reviews, aspect distributions, and product metadata.
        """
        product_info = self.get_product_info(product_id)
        prod_index_data = self._get_or_create_product_index(product_id)

        index: faiss.Index = prod_index_data["index"]
        prod_df: pd.DataFrame = prod_index_data["df"]

        # Determine query intent
        intent = self.detect_query_intent(query)
        total_in_product = len(prod_df)
        effective_k = min(top_k, total_in_product)

        # Encode query
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")

        # Perform global search within product index
        # Search all items in product to allow flexible intent filtering
        all_scores, all_indices = index.search(query_vec, total_in_product)
        all_scores = all_scores[0]
        all_indices = all_indices[0]

        selected_indices = []

        if prioritize_sentiment and intent == "complaints":
            # Prioritize reviews marked as negative (either rating <= 2 or predicted_sentiment == 'negative')
            is_neg = (
                (prod_df.loc[all_indices, "sentiment"] == "negative")
                | (prod_df.loc[all_indices, "predicted_sentiment"] == "negative")
                | (prod_df.loc[all_indices, "rating"] <= 2.0)
            ).values

            neg_indices = all_indices[is_neg]
            selected_indices.extend(neg_indices[:effective_k])

            # If fewer than k negative reviews, fill remaining slots with neutral or other top-scoring reviews
            if len(selected_indices) < effective_k:
                remaining = [idx for idx in all_indices if idx not in selected_indices]
                needed = effective_k - len(selected_indices)
                selected_indices.extend(remaining[:needed])

        elif prioritize_sentiment and intent == "praises":
            # Prioritize reviews marked as positive
            is_pos = (
                (prod_df.loc[all_indices, "sentiment"] == "positive")
                | (prod_df.loc[all_indices, "predicted_sentiment"] == "positive")
                | (prod_df.loc[all_indices, "rating"] >= 4.0)
            ).values

            pos_indices = all_indices[is_pos]
            selected_indices.extend(pos_indices[:effective_k])

            if len(selected_indices) < effective_k:
                remaining = [idx for idx in all_indices if idx not in selected_indices]
                needed = effective_k - len(selected_indices)
                selected_indices.extend(remaining[:needed])
        else:
            # General query: select purely by highest semantic similarity
            selected_indices = list(all_indices[:effective_k])

        # Compile structured retrieved reviews
        retrieved_reviews = []
        aspect_frequency: Dict[str, int] = {}

        for idx in selected_indices:
            row = prod_df.iloc[idx]
            
            # Find similarity score for this index
            score_match = [score for score, i in zip(all_scores, all_indices) if i == idx]
            sim_score = round(float(score_match[0]), 4) if score_match else 0.0

            aspects_raw = row.get("aspects", [])
            if hasattr(aspects_raw, "tolist"):
                aspects = aspects_raw.tolist()
            elif isinstance(aspects_raw, (list, tuple)):
                aspects = list(aspects_raw)
            else:
                aspects = []
            display_aspects = [ASPECT_DISPLAY_NAMES.get(a, a.replace("_", " ").title()) for a in aspects]

            for a in aspects:
                aspect_frequency[a] = aspect_frequency.get(a, 0) + 1

            retrieved_reviews.append({
                "review_index": int(idx),
                "parent_asin": product_id,
                "domain": str(row.get("domain", product_info["domain"])),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "rating": float(row.get("rating", 3.0)),
                "sentiment": str(row.get("sentiment", "neutral")),
                "predicted_sentiment": str(row.get("predicted_sentiment", "neutral")),
                "predicted_confidence": float(row.get("predicted_confidence", 0.0)),
                "similarity_score": sim_score,
                "aspects": aspects,
                "aspect_labels": display_aspects,
            })

        # Top aspects among retrieved reviews
        num_retrieved = len(retrieved_reviews)
        top_aspects = []
        for a, count in sorted(aspect_frequency.items(), key=lambda x: x[1], reverse=True):
            pct = round((count / num_retrieved) * 100, 1) if num_retrieved > 0 else 0.0
            top_aspects.append({
                "aspect": a,
                "label": ASPECT_DISPLAY_NAMES.get(a, a.replace("_", " ").title()),
                "count": count,
                "percentage": pct,
            })

        # Sentiment distribution among retrieved reviews
        retrieved_sentiments = [r["sentiment"] for r in retrieved_reviews]
        sentiment_breakdown = {
            "positive": retrieved_sentiments.count("positive"),
            "neutral": retrieved_sentiments.count("neutral"),
            "negative": retrieved_sentiments.count("negative"),
        }

        return {
            "product_id": product_id,
            "product_title": product_info["title"],
            "domain": product_info["domain"],
            "query": query,
            "intent": intent,
            "total_product_reviews": total_in_product,
            "retrieved_count": len(retrieved_reviews),
            "retrieved_reviews": retrieved_reviews,
            "sentiment_breakdown": sentiment_breakdown,
            "top_aspects": top_aspects,
            "overall_product_stats": product_info,
        }
