"""
Unit tests for ProductRecommender (src/recommender.py).
Tests aspect-aware recommendations, alternatives, and side-by-side product comparisons.
"""

import unittest
from src.recommender import ProductRecommender


class TestProductRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recommender = ProductRecommender()

    def test_recommender_loads_insights(self):
        self.assertFalse(self.recommender.insights.empty)
        self.assertIn("parent_asin", self.recommender.insights.columns)
        self.assertIn("average_rating", self.recommender.insights.columns)
        self.assertIn("positive_percentage", self.recommender.insights.columns)

    def test_recommend_products_electronics(self):
        recs = self.recommender.recommend_products(domain="Electronics", min_rating=4.0, top_k=3)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertEqual(r["domain"].lower(), "electronics")
            self.assertGreaterEqual(r["average_rating"], 4.0)
            self.assertIn("recommendation_score", r)
            self.assertIn("title", r)

    def test_recommend_products_fashion(self):
        recs = self.recommender.recommend_products(domain="Fashion", min_rating=3.5, top_k=3)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertEqual(r["domain"].lower(), "fashion")

    def test_aspect_focused_recommendations(self):
        recs = self.recommender.recommend_products(
            domain="Electronics",
            aspect_focus="battery",
            top_k=3,
        )
        self.assertGreater(len(recs), 0)
        self.assertIn("recommendation_score", recs[0])

    def test_recommend_alternatives(self):
        sample_asin = self.recommender.insights.iloc[0]["parent_asin"]
        alt_res = self.recommender.recommend_alternatives(sample_asin, top_k=2)
        self.assertIn("current_product", alt_res)
        self.assertIn("alternatives", alt_res)
        self.assertEqual(alt_res["current_product"]["parent_asin"], sample_asin)

    def test_compare_products(self):
        p1 = self.recommender.insights.iloc[0]["parent_asin"]
        p2 = self.recommender.insights.iloc[1]["parent_asin"]

        comp = self.recommender.compare_products(p1, p2)
        self.assertIn("product_1", comp)
        self.assertIn("product_2", comp)
        self.assertIn("verdict", comp)
        self.assertIn("winner_asin", comp)

    def test_invalid_product_compare(self):
        with self.assertRaises(ValueError):
            self.recommender.compare_products("NON_EXISTENT_1", "NON_EXISTENT_2")


if __name__ == "__main__":
    unittest.main()
