"""
Unit tests for Conversational Chatbot Engine (src/chatbot.py).
Tests intent detection, entity resolution, multi-turn memory, and integrated handlers.
"""

import unittest
from src.chatbot import FeedbackIQChatbot


class TestFeedbackIQChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bot = FeedbackIQChatbot()

    def test_intent_detection(self):
        self.assertEqual(self.bot.detect_intent("hello there"), "GREETING")
        self.assertEqual(self.bot.detect_intent("what can you do?"), "HELP")
        self.assertEqual(self.bot.detect_intent("compare B075X8471B vs B01K8B8YA8"), "COMPARE")
        self.assertEqual(self.bot.detect_intent("recommend the best electronics"), "RECOMMEND")
        self.assertEqual(self.bot.detect_intent("what should I buy instead of this?"), "ALTERNATIVE")
        self.assertEqual(self.bot.detect_intent("how is the battery life?"), "ASPECT_QUERY")
        self.assertEqual(self.bot.detect_intent("why are customers unhappy with this item?"), "PRODUCT_FEEDBACK")

    def test_extract_product_entities(self):
        query = "Can you compare B075X8471B with B01K8B8YA8?"
        entities = self.bot.extract_product_entities(query)
        self.assertIn("B075X8471B", entities)
        self.assertIn("B01K8B8YA8", entities)

    def test_greeting_response(self):
        resp = self.bot.respond("Hi there!")
        self.assertEqual(resp["intent"], "GREETING")
        self.assertIn("FeedbackIQ", resp["answer"])
        self.assertGreater(len(resp["suggested_questions"]), 0)

    def test_recommend_response(self):
        resp = self.bot.respond("Recommend top electronics products")
        self.assertEqual(resp["intent"], "RECOMMEND")
        self.assertIn("Top Recommended", resp["answer"])
        self.assertIn("ASIN", resp["answer"])

    def test_compare_response(self):
        resp = self.bot.respond("Compare B075X8471B and B01K8B8YA8")
        self.assertEqual(resp["intent"], "COMPARE")
        self.assertIn("Head-to-Head", resp["answer"])
        self.assertIn("Average Rating", resp["answer"])

    def test_multi_turn_history(self):
        self.bot.reset_conversation()
        self.assertEqual(len(self.bot.conversation_history), 0)

        self.bot.respond("Hello!")
        self.assertEqual(len(self.bot.conversation_history), 2)  # 1 user + 1 assistant

        self.bot.respond("Recommend top electronics")
        self.assertEqual(len(self.bot.conversation_history), 4)


if __name__ == "__main__":
    unittest.main()
