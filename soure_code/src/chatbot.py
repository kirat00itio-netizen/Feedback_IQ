"""
FeedbackIQ - Conversational AI Assistant & Chatbot Engine.
Integrates:
1. Multi-turn Conversational Memory & Context Tracking
2. Conversational Intent Classification (Q&A, Recommendations, Comparisons, Aspects)
3. Product Entity Resolution (ASIN / title detection & session state tracking)
4. Grounded RAG Feedback Retrieval (FAISS + Sentence Transformers)
5. Intelligent Aspect-Based Product Recommendations & Alternatives
6. Head-to-Head Product Comparisons
7. Modular LLM Generation (Grounded Synthesizer, Ollama, Gemini)
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import PROCESSED_DATA_DIR
from src.retrieval import ReviewRetriever, ASPECT_DISPLAY_NAMES, DOMAIN_ASPECTS
from src.rag import FeedbackRAGPipeline, GroundedSynthesizerLLM, BaseLLM, get_llm
from src.recommender import ProductRecommender
from src.product_analytics import get_product_summary, get_product_sentiment, get_product_aspects

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FeedbackIQChatbot:
    """
    Conversational AI Assistant that orchestrates review RAG, aspect analytics,
    intelligent product recommendations, and product comparisons.
    """

    def __init__(
        self,
        retriever: Optional[ReviewRetriever] = None,
        recommender: Optional[ProductRecommender] = None,
        llm: Optional[BaseLLM] = None,
        llm_type: str = "grounded",
        **llm_kwargs,
    ):
        self.retriever = retriever or ReviewRetriever()
        self.recommender = recommender or ProductRecommender()
        self.llm = llm or get_llm(provider=llm_type, **llm_kwargs)
        self.rag_pipeline = FeedbackRAGPipeline(retriever=self.retriever, llm=self.llm)

        # Multi-turn state
        self.conversation_history: List[Dict[str, Any]] = []
        self.active_product_id: Optional[str] = None
        self.active_domain: str = "Electronics"

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()

    def set_active_product(self, product_id: str, domain: Optional[str] = None):
        """Update current product context."""
        self.active_product_id = product_id
        if domain:
            self.active_domain = domain

    def extract_product_entities(self, query: str) -> List[str]:
        """
        Extract ASINs or product names mentioned in the user message.
        Looks for standard Amazon ASIN patterns (e.g. B0... or 10-char alphanumeric),
        or performs fuzzy title matching against available products.
        """
        extracted = []

        # 1. Direct ASIN regex match
        asin_matches = re.findall(r"\b([A-Z0-9]{10})\b", query.upper())
        for asin in asin_matches:
            # Verify if this ASIN exists in catalog
            prod = self.recommender.find_product(asin)
            if prod is not None:
                extracted.append(prod["parent_asin"])

        # 2. Known product title lookup if no direct ASIN found
        if not extracted:
            available = self.retriever.get_available_products()
            query_lower = query.lower()
            for p in available:
                p_title = p.get("title", "").lower()
                # Check if significant portion of title matches
                clean_title = re.sub(r"[^\w\s]", "", p_title)
                words = [w for w in clean_title.split() if len(w) > 3]
                if len(words) >= 2:
                    key_phrase = " ".join(words[:3])
                    if key_phrase in query_lower:
                        extracted.append(p["parent_asin"])
                        break

        # Deduplicate
        seen = set()
        deduped = []
        for x in extracted:
            if x not in seen:
                seen.add(x)
                deduped.append(x)

        return deduped

    def detect_intent(self, query: str) -> str:
        """
        Classifies user message intent into:
        - GREETING: Greetings or bot capabilities
        - COMPARE: Product comparison queries
        - RECOMMEND: Product recommendation requests
        - ALTERNATIVE: Asking for alternatives to a flawed product
        - ASPECT_QUERY: Questions focused on specific attributes (battery, size, etc.)
        - PRODUCT_FEEDBACK: Review inquiries, complaints, praises
        - GENERAL: Fallback
        """
        q = query.lower().strip()

        # Greeting check
        if re.search(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b", q) or q in ["hi", "hello", "hey"]:
            return "GREETING"

        if any(kw in q for kw in ["what can you do", "who are you", "how does this work", "help me"]):
            return "HELP"

        # Comparison check
        if any(kw in q for kw in ["compare", " vs ", " versus ", "which is better", "difference between"]):
            return "COMPARE"

        # Alternative check
        if any(kw in q for kw in ["alternative", "instead of", "better option", "replace", "switch to"]):
            return "ALTERNATIVE"

        # Recommendation check
        if any(kw in q for kw in [
            "recommend", "suggestion", "suggest", "best product", "top rated",
            "what should i buy", "which product should i", "find me", "show me good"
        ]):
            return "RECOMMEND"

        # Aspect query check
        all_aspects = list(ASPECT_DISPLAY_NAMES.keys())
        for asp in all_aspects:
            disp = ASPECT_DISPLAY_NAMES[asp].lower()
            if asp in q or any(term in q for term in disp.split("&")):
                return "ASPECT_QUERY"

        # Default to product feedback RAG
        return "PRODUCT_FEEDBACK"

    def respond(
        self,
        user_message: str,
        current_product_id: Optional[str] = None,
        current_domain: Optional[str] = None,
        top_k_reviews: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate conversational response handling multi-turn context and intents.
        """
        if current_domain:
            self.active_domain = current_domain
        if current_product_id:
            self.active_product_id = current_product_id

        # 1. Detect Intent & Entities
        intent = self.detect_intent(user_message)
        mentioned_asins = self.extract_product_entities(user_message)

        # Target product resolution
        target_asin = mentioned_asins[0] if mentioned_asins else self.active_product_id

        # Dispatch
        if intent == "GREETING":
            resp = self._handle_greeting()
        elif intent == "HELP":
            resp = self._handle_help()
        elif intent == "COMPARE":
            resp = self._handle_compare(mentioned_asins, user_message)
        elif intent == "ALTERNATIVE":
            resp = self._handle_alternative(target_asin)
        elif intent == "RECOMMEND":
            resp = self._handle_recommend(user_message)
        elif intent == "ASPECT_QUERY":
            resp = self._handle_aspect_query(target_asin, user_message, top_k_reviews)
        else:
            resp = self._handle_feedback_rag(target_asin, user_message, top_k_reviews)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": resp["answer"]})

        return resp

    # --- Internal Handlers ---

    def _handle_greeting(self) -> Dict[str, Any]:
        answer = (
            "👋 **Hello! I'm FeedbackIQ**, your AI Customer Feedback & Product Intelligence Assistant.\n\n"
            "I analyze thousands of verified Amazon reviews across **Electronics** and **Fashion** to give you:\n"
            "- 🔍 **Grounded Review Analysis**: Detailed complaint and praise breakdowns backed by real customer reviews.\n"
            "- 🎯 **Aspect Analytics**: Pinpoint problems in Battery, Fit, Fabric, Durability, Price, and more.\n"
            "- ⚖️ **Head-to-Head Comparisons**: Direct comparison between competing products.\n"
            "- 💡 **Intelligent Recommendations**: High-satisfaction product suggestions matching your needs.\n\n"
            "Select a product from the sidebar or ask any question to get started!"
        )
        return {
            "answer": answer,
            "intent": "GREETING",
            "product_id": self.active_product_id,
            "citations": [],
            "suggested_questions": [
                "What are the biggest complaints for this product?",
                "What do customers love most?",
                "Recommend top-rated products in this category",
                "Are there better alternatives to this product?",
            ],
        }

    def _handle_help(self) -> Dict[str, Any]:
        answer = (
            "### 🛠️ How FeedbackIQ Works\n\n"
            "Here are things you can ask me:\n"
            "1. **Product Feedback**: *'Why are customers unhappy with this item?'* or *'What is the audio quality like?'*\n"
            "2. **Specific Aspects**: *'Is the sizing accurate?'* or *'How is the battery life?'*\n"
            "3. **Recommendations**: *'Recommend top 3 electronics with high ratings'* or *'Show me products with good battery'*.\n"
            "4. **Product Comparisons**: *'Compare B075X8471B and B01K8B8YA8'*.\n"
            "5. **Alternatives**: *'What should I buy instead of this product?'*\n"
        )
        return {
            "answer": answer,
            "intent": "HELP",
            "product_id": self.active_product_id,
            "citations": [],
            "suggested_questions": [
                "What are the top complaint aspects?",
                "Recommend the best products in Electronics",
                "Recommend the best products in Fashion",
            ],
        }

    def _handle_compare(self, asins: List[str], query: str) -> Dict[str, Any]:
        # If user provided 2 ASINs directly
        if len(asins) >= 2:
            p1_id, p2_id = asins[0], asins[1]
        elif len(asins) == 1 and self.active_product_id and asins[0] != self.active_product_id:
            p1_id, p2_id = self.active_product_id, asins[0]
        else:
            # Suggest available product to compare against
            top_prods = self.recommender.recommend_products(
                domain=self.active_domain,
                exclude_asin=self.active_product_id,
                top_k=2,
            )
            if top_prods and self.active_product_id:
                p1_id = self.active_product_id
                p2_id = top_prods[0]["parent_asin"]
            else:
                return {
                    "answer": "To compare products, please provide two product IDs or names (e.g. *'Compare B075X8471B and B01K8B8YA8'*).",
                    "intent": "COMPARE",
                    "product_id": self.active_product_id,
                    "citations": [],
                    "suggested_questions": [],
                }

        try:
            comp = self.recommender.compare_products(p1_id, p2_id)
            p1, p2 = comp["product_1"], comp["product_2"]

            p1_complaints = ", ".join([f"{c['display_name']} ({c['percentage']}%)" for c in p1["top_complaints"]]) or "None detected"
            p2_complaints = ", ".join([f"{c['display_name']} ({c['percentage']}%)" for c in p2["top_complaints"]]) or "None detected"

            answer = (
                f"### ⚖️ Head-to-Head Product Comparison\n\n"
                f"| Metric | **{p1['title'][:35]}...** | **{p2['title'][:35]}...** |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **ASIN** | `{p1['parent_asin']}` | `{p2['parent_asin']}` |\n"
                f"| **Domain** | {p1['domain']} | {p2['domain']} |\n"
                f"| **Average Rating** | **{p1['average_rating']} / 5.0** | **{p2['average_rating']} / 5.0** |\n"
                f"| **Total Reviews** | {p1['total_reviews']} | {p2['total_reviews']} |\n"
                f"| **Positive Sentiment** | `{p1['positive_percentage']}%` | `{p2['positive_percentage']}%` |\n"
                f"| **Neutral Sentiment** | `{p1['neutral_percentage']}%` | `{p2['neutral_percentage']}%` |\n"
                f"| **Negative Sentiment** | `{p1['negative_percentage']}%` | `{p2['negative_percentage']}%` |\n"
                f"| **Top Complaints** | {p1_complaints} | {p2_complaints} |\n\n"
                f"#### 🏆 Analysis & Verdict\n{comp['verdict']}"
            )

            return {
                "answer": answer,
                "intent": "COMPARE",
                "product_id": p1_id,
                "comparison": comp,
                "citations": [],
                "suggested_questions": [
                    f"Why do customers like {p1['parent_asin']}?",
                    f"Why do customers like {p2['parent_asin']}?",
                    "What are the main complaints for each?",
                ],
            }
        except Exception as e:
            return {
                "answer": f"Unable to compare products: {e}",
                "intent": "COMPARE",
                "product_id": self.active_product_id,
                "citations": [],
                "suggested_questions": [],
            }

    def _handle_alternative(self, target_asin: Optional[str]) -> Dict[str, Any]:
        if not target_asin:
            target_asin = self.recommender.insights.iloc[0]["parent_asin"]

        try:
            alt_data = self.recommender.recommend_alternatives(target_asin, top_k=3)
            curr = alt_data["current_product"]
            alts = alt_data["alternatives"]

            lines = [
                f"### 🔄 Recommended Alternatives for \"{curr['title'][:50]}...\"\n",
                f"- **Current Rating**: {curr['average_rating']} / 5.0 ({curr['total_reviews']} reviews)",
                f"- **Negative Feedback Rate**: `{curr['negative_percentage']}%`",
            ]
            if alt_data["primary_complaint_aspect"]:
                lines.append(f"- **Primary Area of Concern**: `{ASPECT_DISPLAY_NAMES.get(alt_data['primary_complaint_aspect'], alt_data['primary_complaint_aspect'])}`\n")

            lines.append("#### Top Rated Alternatives in this Domain:")
            for i, a in enumerate(alts, 1):
                lines.append(
                    f"{i}. **{a['title']}** (ASIN: `{a['parent_asin']}`)  \n"
                    f"   ⭐ **Rating**: {a['average_rating']} / 5.0 | 👍 **Positive**: `{a['positive_percentage']}%` | 👎 **Negative**: `{a['negative_percentage']}%`  \n"
                    f"   *Complaints*: {', '.join(a['complaint_highlights']) if a['complaint_highlights'] else 'No major issues'}"
                )

            return {
                "answer": "\n".join(lines),
                "intent": "ALTERNATIVE",
                "product_id": target_asin,
                "citations": [],
                "suggested_questions": [
                    f"Compare {target_asin} with {alts[0]['parent_asin']}" if alts else "Show more recommendations",
                    "What are the customer praises for these alternatives?",
                ],
            }
        except Exception as e:
            return {
                "answer": f"Error finding alternatives: {e}",
                "intent": "ALTERNATIVE",
                "product_id": target_asin,
                "citations": [],
                "suggested_questions": [],
            }

    def _handle_recommend(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        domain = "Fashion" if "fashion" in q_lower or "clothing" in q_lower else (
            "Electronics" if "electronic" in q_lower or "device" in q_lower else self.active_domain
        )

        # Detect aspect focus if requested
        aspect_focus = None
        for a in ASPECT_DISPLAY_NAMES:
            if a in q_lower or ASPECT_DISPLAY_NAMES[a].lower() in q_lower:
                aspect_focus = a
                break

        recs = self.recommender.recommend_products(
            domain=domain,
            min_rating=4.0,
            aspect_focus=aspect_focus,
            top_k=4,
        )

        aspect_note = f" focused on **{ASPECT_DISPLAY_NAMES.get(aspect_focus, aspect_focus)}**" if aspect_focus else ""
        lines = [
            f"### 🎯 Top Recommended Products in **{domain}**{aspect_note}\n",
            "Based on verified customer reviews, high satisfaction rates, and low complaint incidence:\n",
        ]

        for i, r in enumerate(recs, 1):
            lines.append(
                f"{i}. **{r['title']}**  \n"
                f"   - **ASIN**: `{r['parent_asin']}` | **Rating**: **{r['average_rating']} / 5.0** ({r['total_reviews']} reviews)  \n"
                f"   - **Sentiment**: `{r['positive_percentage']}%` positive, only `{r['negative_percentage']}%` negative  \n"
                f"   - **Quality Score**: `{r['recommendation_score']}`  \n"
            )

        return {
            "answer": "\n".join(lines),
            "intent": "RECOMMEND",
            "product_id": recs[0]["parent_asin"] if recs else self.active_product_id,
            "citations": [],
            "suggested_questions": [
                f"What do customers say about {recs[0]['parent_asin']}?" if recs else "Show more products",
                f"Compare {recs[0]['parent_asin']} and {recs[1]['parent_asin']}" if len(recs) > 1 else "How are the reviews?",
            ],
        }

    def _handle_aspect_query(self, target_asin: Optional[str], query: str, top_k: int) -> Dict[str, Any]:
        if not target_asin:
            target_asin = self.recommender.insights.iloc[0]["parent_asin"]

        # Forward to RAG with aspect prioritization
        rag_res = self.rag_pipeline.answer_question(
            product_id=target_asin,
            question=query,
            top_k=top_k,
            prioritize_sentiment=True,
        )

        return {
            "answer": rag_res["answer"],
            "intent": "ASPECT_QUERY",
            "product_id": target_asin,
            "citations": rag_res["retrieval_context"]["retrieved_reviews"],
            "suggested_questions": [
                "What other complaints do customers have?",
                "What are the best features of this product?",
                "Are there better alternatives?",
            ],
        }

    def _handle_feedback_rag(self, target_asin: Optional[str], query: str, top_k: int) -> Dict[str, Any]:
        if not target_asin:
            target_asin = self.recommender.insights.iloc[0]["parent_asin"]

        rag_res = self.rag_pipeline.answer_question(
            product_id=target_asin,
            question=query,
            top_k=top_k,
            prioritize_sentiment=True,
        )

        return {
            "answer": rag_res["answer"],
            "intent": "PRODUCT_FEEDBACK",
            "product_id": target_asin,
            "citations": rag_res["retrieval_context"]["retrieved_reviews"],
            "suggested_questions": [
                "What are the biggest complaints?",
                "What do customers like about this product?",
                "Recommend alternatives with higher satisfaction",
            ],
        }
