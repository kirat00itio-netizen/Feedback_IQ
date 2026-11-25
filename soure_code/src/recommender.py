"""
FeedbackIQ - Intelligent Product Recommendation & Comparison Engine.
Provides:
1. Aspect-Aware Product Recommendations (by rating, review volume, aspect excellence)
2. Alternative Product Recommendations (better products when one has severe complaints)
3. Head-to-Head Product Comparisons (side-by-side sentiment, rating & aspect breakdown)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.retrieval import ASPECT_DISPLAY_NAMES, DOMAIN_ASPECTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRODUCT_INSIGHTS_FILE = PROCESSED_DATA_DIR / "product_insights.parquet"
PRODUCTS_FILE = PROCESSED_DATA_DIR / "products.parquet"


class ProductRecommender:
    """
    Intelligent recommendation and product comparison engine powered by
    aggregated customer sentiment, review volumes, and aspect complaint analytics.
    """

    def __init__(
        self,
        insights_path: Optional[Path] = None,
        products_path: Optional[Path] = None,
    ):
        self.insights_path = insights_path or PRODUCT_INSIGHTS_FILE
        self.products_path = products_path or PRODUCTS_FILE
        self._df_insights: Optional[pd.DataFrame] = None
        self._df_products: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        if self.insights_path.exists():
            self._df_insights = pd.read_parquet(self.insights_path)
        else:
            logger.warning("Product insights file not found at %s", self.insights_path)
            self._df_insights = pd.DataFrame()

        if self.products_path.exists():
            self._df_products = pd.read_parquet(self.products_path)
        else:
            self._df_products = pd.DataFrame()

    @property
    def insights(self) -> pd.DataFrame:
        if self._df_insights is None:
            self._load()
        return self._df_insights

    def find_product(self, identifier: str) -> Optional[pd.Series]:
        """Find product row by exact parent_asin or title substring."""
        if self.insights.empty:
            return None

        clean_id = str(identifier).strip()
        # 1. Exact ASIN match
        match = self.insights[self.insights["parent_asin"].str.upper() == clean_id.upper()]
        if not match.empty:
            return match.iloc[0]

        # 2. Case-insensitive Title Substring Match
        title_matches = self.insights[
            self.insights["title"].str.lower().str.contains(clean_id.lower(), na=False)
        ]
        if not title_matches.empty:
            # Prefer product with highest review count among matches
            return title_matches.sort_values("total_reviews", ascending=False).iloc[0]

        return None

    def recommend_products(
        self,
        domain: Optional[str] = None,
        min_rating: float = 4.0,
        min_reviews: int = 5,
        aspect_focus: Optional[str] = None,
        exclude_asin: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend top products matching domain, quality thresholds, and aspect preferences.

        Parameters:
            domain: 'Electronics', 'Fashion', or None for all.
            min_rating: Minimum average rating threshold (e.g. 4.0).
            min_reviews: Minimum verified review count threshold.
            aspect_focus: Key aspect to prioritize (e.g. 'battery', 'durability', 'material').
            exclude_asin: Optional ASIN to exclude (e.g. current product).
            top_k: Number of recommendations to return.

        Returns:
            List of structured product recommendations sorted by recommendation score.
        """
        df = self.insights.copy()
        if df.empty:
            return []

        # 1. Filter by domain
        if domain and domain.lower() != "all":
            df = df[df["domain"].str.lower() == domain.lower()]

        # 2. Filter by minimum rating and volume
        df = df[df["average_rating"] >= min_rating]
        df = df[df["total_reviews"] >= min_reviews]

        if exclude_asin:
            df = df[df["parent_asin"].str.upper() != exclude_asin.upper()]

        if df.empty:
            # Relax rating constraint if dataset is small
            df = self.insights.copy()
            if domain and domain.lower() != "all":
                df = df[df["domain"].str.lower() == domain.lower()]
            if exclude_asin:
                df = df[df["parent_asin"].str.upper() != exclude_asin.upper()]
            df = df.sort_values("total_reviews", ascending=False).head(top_k)

        # 3. Calculate recommendation score
        # Base quality: 40% rating (0-5 -> 0-1) + 40% positive pct + 20% negative penalty
        rating_norm = df["average_rating"] / 5.0
        pos_norm = df["positive_percentage"] / 100.0
        neg_penalty = df["negative_percentage"] / 100.0
        # Volume log bonus (caps at +0.10 for high volume products)
        vol_bonus = np.clip(np.log10(np.maximum(df["total_reviews"], 1)) * 0.04, 0.0, 0.12)

        base_score = (rating_norm * 0.40) + (pos_norm * 0.40) - (neg_penalty * 0.20) + vol_bonus

        # Aspect focus adjustment: boost products with low/zero complaints in the requested aspect
        clean_aspect = aspect_focus.lower().replace(" ", "_") if aspect_focus else None
        aspect_bonuses = []
        for _, row in df.iterrows():
            bonus = 0.0
            if clean_aspect:
                complaint_aspects = self._parse_aspects(row.get("top_complaint_aspects_json"))
                # Check if product has complaints in focused aspect
                asp_match = next((a for a in complaint_aspects if a["aspect"] == clean_aspect), None)
                if asp_match is None:
                    # Clean record on this aspect!
                    bonus = 0.15
                else:
                    # Penalize according to complaint percentage
                    complaint_rate = asp_match.get("percentage", 0.0)
                    bonus = -0.15 if complaint_rate > 20 else -0.05
            aspect_bonuses.append(bonus)

        df["recommendation_score"] = base_score + np.array(aspect_bonuses)
        top_products = df.sort_values("recommendation_score", ascending=False).head(top_k)

        results = []
        for _, r in top_products.iterrows():
            top_complaints = self._parse_aspects(r.get("top_complaint_aspects_json"))
            highlight_complaints = [
                f"{a['display_name']} ({a['percentage']}%)" for a in top_complaints[:2]
            ]
            results.append({
                "parent_asin": r["parent_asin"],
                "title": r["title"],
                "domain": r["domain"],
                "average_rating": float(round(r["average_rating"], 2)),
                "total_reviews": int(r["total_reviews"]),
                "positive_percentage": float(r["positive_percentage"]),
                "neutral_percentage": float(r["neutral_percentage"]),
                "negative_percentage": float(r["negative_percentage"]),
                "recommendation_score": float(round(r["recommendation_score"], 3)),
                "complaint_highlights": highlight_complaints,
                "summary": r.get("top_complaints_summary", ""),
            })

        return results

    def recommend_alternatives(
        self,
        product_id: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Find superior alternatives for a product that has customer complaints or low rating.
        Searches the same domain for products with higher rating and fewer complaints.
        """
        curr = self.find_product(product_id)
        if curr is None:
            raise ValueError(f"Product '{product_id}' not found in catalog.")

        curr_domain = curr["domain"]
        curr_rating = curr["average_rating"]
        curr_neg = curr["negative_percentage"]
        curr_aspects = self._parse_aspects(curr.get("top_complaint_aspects_json"))

        # Find main complaint aspect to avoid
        main_issue = curr_aspects[0]["aspect"] if curr_aspects else None

        # Recommend alternatives in same domain with higher rating
        alts = self.recommend_products(
            domain=curr_domain,
            min_rating=max(curr_rating, 3.8),
            min_reviews=min(5, int(curr["total_reviews"])),
            aspect_focus=main_issue,
            exclude_asin=curr["parent_asin"],
            top_k=top_k,
        )

        return {
            "current_product": {
                "parent_asin": curr["parent_asin"],
                "title": curr["title"],
                "domain": curr_domain,
                "average_rating": float(round(curr_rating, 2)),
                "total_reviews": int(curr["total_reviews"]),
                "positive_percentage": float(curr["positive_percentage"]),
                "negative_percentage": float(curr_neg),
                "top_complaint_aspects": curr_aspects[:3],
            },
            "primary_complaint_aspect": main_issue,
            "alternatives": alts,
        }

    def compare_products(
        self,
        product_id_1: str,
        product_id_2: str,
    ) -> Dict[str, Any]:
        """
        Perform a head-to-head comparison between two products on customer feedback metrics.
        """
        p1 = self.find_product(product_id_1)
        if p1 is None:
            raise ValueError(f"First product '{product_id_1}' not found in catalog.")

        p2 = self.find_product(product_id_2)
        if p2 is None:
            raise ValueError(f"Second product '{product_id_2}' not found in catalog.")

        asp1 = self._parse_aspects(p1.get("top_complaint_aspects_json"))
        asp2 = self._parse_aspects(p2.get("top_complaint_aspects_json"))

        r1, r2 = float(p1["average_rating"]), float(p2["average_rating"])
        pos1, pos2 = float(p1["positive_percentage"]), float(p2["positive_percentage"])
        neg1, neg2 = float(p1["negative_percentage"]), float(p2["negative_percentage"])

        # Determine winner
        score1 = r1 * 0.5 + (pos1 - neg1) * 0.05
        score2 = r2 * 0.5 + (pos2 - neg2) * 0.05

        if score1 > score2 + 0.15:
            winner = p1["parent_asin"]
            winner_title = p1["title"]
            verdict_text = (
                f"**{winner_title}** leads with higher customer satisfaction "
                f"({r1}/5.0 vs {r2}/5.0, {pos1}% vs {pos2}% positive reviews)."
            )
        elif score2 > score1 + 0.15:
            winner = p2["parent_asin"]
            winner_title = p2["title"]
            verdict_text = (
                f"**{winner_title}** leads with higher customer satisfaction "
                f"({r2}/5.0 vs {r1}/5.0, {pos2}% vs {pos1}% positive reviews)."
            )
        else:
            winner = "TIE"
            verdict_text = "Both products perform very closely in customer satisfaction and sentiment."

        return {
            "product_1": {
                "parent_asin": p1["parent_asin"],
                "title": p1["title"],
                "domain": p1["domain"],
                "average_rating": r1,
                "total_reviews": int(p1["total_reviews"]),
                "positive_percentage": pos1,
                "neutral_percentage": float(p1["neutral_percentage"]),
                "negative_percentage": neg1,
                "top_complaints": asp1[:3],
            },
            "product_2": {
                "parent_asin": p2["parent_asin"],
                "title": p2["title"],
                "domain": p2["domain"],
                "average_rating": r2,
                "total_reviews": int(p2["total_reviews"]),
                "positive_percentage": pos2,
                "neutral_percentage": float(p2["neutral_percentage"]),
                "negative_percentage": neg2,
                "top_complaints": asp2[:3],
            },
            "winner_asin": winner,
            "verdict": verdict_text,
        }

    @staticmethod
    def _parse_aspects(val: Any) -> List[Dict[str, Any]]:
        """Safely parse top_complaint_aspects_json into a list of dicts."""
        if val is None or pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return []
        return []


# Global singleton instance for easy import
_recommender = ProductRecommender()


def recommend_top_products(domain: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    return _recommender.recommend_products(domain=domain, top_k=top_k)


def compare_products_side_by_side(id1: str, id2: str) -> Dict[str, Any]:
    return _recommender.compare_products(id1, id2)
