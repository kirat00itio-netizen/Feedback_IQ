"""
FeedbackIQ — AI-Powered Customer Feedback Intelligence & Conversational RAG System.
Features:
- Multi-turn Conversational AI Chatbot with Grounded RAG
- Domain & Aspect-Based Sentiment Analysis
- Intelligent Aspect-Aware Product Recommendations & Alternatives
- Head-to-Head Product Comparisons
- Modular LLM generation: Grounded Synthesizer, Google Gemini API, and local Ollama

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Ensure Windows OpenMP DLL compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repository root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except Exception:
    pass

import pandas as pd
import streamlit as st

from src.retrieval import ReviewRetriever, ASPECT_DISPLAY_NAMES
from src.rag import FeedbackRAGPipeline, BaseLLM, OllamaLLM, GroundedSynthesizerLLM, GeminiLLM, get_llm
from src.recommender import ProductRecommender
from src.chatbot import FeedbackIQChatbot
from src.product_analytics import get_product_sentiment, get_product_aspects

# Page Configuration
st.set_page_config(
    page_title="FeedbackIQ — Customer Feedback Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .card-pos { color: #16A34A; }
    .card-neu { color: #D97706; }
    .card-neg { color: #DC2626; }
    .card-neutral { color: #2563EB; }
    .evidence-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .aspect-pill {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .intent-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading FeedbackIQ Core Engines (FAISS + SentenceTransformers + Recommender)...")
def load_core_engines():
    retriever = ReviewRetriever()
    recommender = ProductRecommender()
    return retriever, recommender


retriever, recommender = load_core_engines()

# App Header
st.markdown('<div class="main-title">🛍️ FeedbackIQ: Customer Feedback Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Aspect-Based Sentiment Analysis, Smart Recommendations & Conversational Grounded RAG Chatbot</div>',
    unsafe_allow_html=True,
)

# Sidebar: Controls & Settings
with st.sidebar:
    st.header("⚙️ Controls & Catalog")

    # 1. Domain Selection
    selected_domain = st.selectbox(
        "📁 Product Domain:",
        options=["Electronics", "Fashion"],
        index=0,
        help="Filter products by Amazon product category.",
    )

    # Filter catalog by selected domain
    all_products = retriever.get_available_products()
    domain_products = [p for p in all_products if p.get("domain", "").lower() == selected_domain.lower()]

    if not domain_products:
        st.warning(f"No products found for domain '{selected_domain}'.")
        st.stop()

    # 2. Product Volume Filter
    vol_choice = st.selectbox(
        "📊 Filter Products by Volume:",
        options=[
            "Recommended (>= 5 reviews)",
            "Moderate (>= 2 reviews)",
            "All Products",
        ],
        index=0,
    )
    threshold = 5 if ">= 5" in vol_choice else (2 if ">= 2" in vol_choice else 1)
    filtered_products = [p for p in domain_products if p.get("total_reviews", 0) >= threshold]
    if not filtered_products:
        filtered_products = domain_products

    # 3. Product Selection
    product_options = {}
    for p in filtered_products:
        title = p.get("title", f"Product {p['parent_asin']}")
        if len(title) > 60:
            display_title = title[:57].rsplit(" ", 1)[0] + "..."
        else:
            display_title = title
        product_options[p["parent_asin"]] = f"{display_title} ({p['total_reviews']} revs) [{p['parent_asin']}]"

    selected_asin = st.selectbox(
        "📦 Active Product:",
        options=list(product_options.keys()),
        format_func=lambda asin: product_options[asin],
        index=0,
        help="The product currently in conversational focus.",
    )

    st.markdown("---")
    st.markdown("**🧠 LLM Generation Backend**")

    # Detect Ollama
    def _is_ollama_available() -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.3) as r:
                return r.status == 200
        except Exception:
            return False

    ollama_ready = _is_ollama_available()
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY"))

    llm_backend_choice = st.selectbox(
        "Select LLM Engine:",
        options=[
            "📊 Grounded Synthesizer (Fast, Zero-hallucination)",
            "✨ Google Gemini (Cloud API)",
            f"🦙 Ollama Local ({'Online' if ollama_ready else 'Offline'})",
        ],
        index=0,
        help="Grounded Synthesizer runs deterministically with 0 hallucination. Gemini and Ollama provide natural language synthesis.",
    )

    llm_instance: BaseLLM = GroundedSynthesizerLLM()

    if "Gemini" in llm_backend_choice:
        gemini_key_input = st.text_input(
            "Gemini API Key:",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Enter your Gemini API key or set GEMINI_API_KEY environment variable.",
        )
        if gemini_key_input:
            try:
                llm_instance = GeminiLLM(model_name="gemini-1.5-flash", api_key=gemini_key_input)
                st.success("✅ Gemini API connected")
            except Exception as e:
                st.error(f"Gemini init error: {e}")
                llm_instance = GroundedSynthesizerLLM()
        else:
            st.info("Using Grounded Synthesizer fallback until Gemini API key is provided.")
            llm_instance = GroundedSynthesizerLLM()

    elif "Ollama" in llm_backend_choice:
        ollama_model = st.text_input("Ollama model:", value="llama3.2:3b")
        if ollama_ready:
            try:
                llm_instance = OllamaLLM(model_name=ollama_model)
            except Exception:
                llm_instance = GroundedSynthesizerLLM()
        else:
            st.info("Ollama is not running locally. Using Grounded Synthesizer.")
            llm_instance = GroundedSynthesizerLLM()
    else:
        llm_instance = GroundedSynthesizerLLM()

    st.markdown("---")
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

# Initialize Chatbot Engine
chatbot = FeedbackIQChatbot(
    retriever=retriever,
    recommender=recommender,
    llm=llm_instance,
)
chatbot.set_active_product(selected_asin, selected_domain)

# Product metadata
product_info = retriever.get_product_info(selected_asin)
prod_title = product_info["title"]
total_revs = product_info["total_reviews"]
avg_rating = product_info["average_rating"]
pos_pct = product_info["sentiment_percentages"]["positive"]
neu_pct = product_info["sentiment_percentages"]["neutral"]
neg_pct = product_info["sentiment_percentages"]["negative"]

# Active product banner
st.caption(f"**Focused Product:** **{prod_title}** | ASIN: `{selected_asin}` | Domain: `{selected_domain}` | Rating: `{avg_rating} ★` ({total_revs} reviews)")

# Main Tabs Navigation
tab_chat, tab_analytics, tab_recs, tab_compare = st.tabs([
    "💬 Conversational AI Chatbot",
    "📊 Sentiment & Aspect Analytics",
    "🎯 Smart Recommendations",
    "⚖️ Head-to-Head Comparison",
])

# =========================================================================
# TAB 1: CONVERSATIONAL AI CHATBOT
# =========================================================================
with tab_chat:
    st.subheader("💬 Interactive FeedbackIQ Assistant")
    st.write(
        "Chat with FeedbackIQ about this product, ask about specific aspects (battery, sizing, quality), "
        "request recommendations, or ask to compare products side-by-side."
    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hello! I'm **FeedbackIQ**. I've analyzed **{total_revs} customer reviews** for "
                    f"**{prod_title}** ({selected_domain}).\n\n"
                    f"- ⭐ **Average Rating**: {avg_rating} / 5.0\n"
                    f"- 👍 **Positive Feedback**: {pos_pct}%\n"
                    f"- 👎 **Negative Feedback**: {neg_pct}%\n\n"
                    "What would you like to know? You can ask about complaints, specific aspects, better alternatives, or compare this with another product!"
                ),
                "citations": [],
                "intent": "GREETING",
            }
        ]

    # Quick prompt suggestion pills
    st.markdown("**Suggested Quick Prompts:**")
    pill_cols = st.columns(4)
    quick_prompts = [
        "What are the biggest complaints?",
        "What do customers like most?",
        "What are better alternatives to this?",
        "How is the quality and durability?",
    ]

    clicked_prompt = None
    for idx, prompt in enumerate(quick_prompts):
        if pill_cols[idx].button(prompt, key=f"chip_btn_{idx}", use_container_width=True):
            clicked_prompt = prompt

    # Display conversation messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent") and msg["role"] == "assistant" and msg["intent"] != "GREETING":
                st.markdown(f'<span class="intent-badge">Intent: {msg["intent"]}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])

            if msg.get("citations"):
                with st.expander(f"🔎 View Grounded Review Citations ({len(msg['citations'])})"):
                    for c_idx, r in enumerate(msg["citations"], 1):
                        asp_str = ", ".join(r.get("aspect_labels", [])) if r.get("aspect_labels") else "General"
                        st.markdown(
                            f"""
                            <div class="evidence-card">
                                <b>#{c_idx} — {r.get('title', 'Review')}</b> (Rating: <b>{r.get('rating', '')} ★</b> | Sentiment: <code>{r.get('sentiment', '')}</code>)<br>
                                <span style="color: #475569; font-size: 0.92rem;">"{r.get('text', '')}"</span><br>
                                <span style="font-size: 0.78rem; color: #64748B;">Aspects: {asp_str} | Similarity: {r.get('similarity_score', 'N/A')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Handle input from chat_input or quick prompt click
    user_input = st.chat_input("Ask FeedbackIQ anything about product reviews, aspects, or recommendations...")
    active_prompt = clicked_prompt or user_input

    if active_prompt:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing verified customer reviews and generating grounded answer..."):
                response = chatbot.respond(
                    user_message=active_prompt,
                    current_product_id=selected_asin,
                    current_domain=selected_domain,
                    top_k_reviews=5,
                )

            st.markdown(f'<span class="intent-badge">Intent: {response["intent"]}</span>', unsafe_allow_html=True)
            st.markdown(response["answer"])

            citations = response.get("citations", [])
            if citations:
                with st.expander(f"🔎 View Grounded Review Citations ({len(citations)})"):
                    for c_idx, r in enumerate(citations, 1):
                        asp_str = ", ".join(r.get("aspect_labels", [])) if r.get("aspect_labels") else "General"
                        st.markdown(
                            f"""
                            <div class="evidence-card">
                                <b>#{c_idx} — {r.get('title', 'Review')}</b> (Rating: <b>{r.get('rating', '')} ★</b> | Sentiment: <code>{r.get('sentiment', '')}</code>)<br>
                                <span style="color: #475569; font-size: 0.92rem;">"{r.get('text', '')}"</span><br>
                                <span style="font-size: 0.78rem; color: #64748B;">Aspects: {asp_str} | Similarity: {r.get('similarity_score', 'N/A')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Save to chat history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response["answer"],
            "citations": citations,
            "intent": response["intent"],
        })
        st.rerun()

# =========================================================================
# TAB 2: SENTIMENT & ASPECT ANALYTICS
# =========================================================================
with tab_analytics:
    st.subheader(f"📊 Sentiment & Aspect Breakdown for {prod_title}")

    # Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value card-neutral">{avg_rating} ★</div>'
            f'<div class="metric-label">Avg Rating</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value card-neutral">{total_revs}</div>'
            f'<div class="metric-label">Total Reviews</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value card-pos">{pos_pct}%</div>'
            f'<div class="metric-label">Positive</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value card-neu">{neu_pct}%</div>'
            f'<div class="metric-label">Neutral</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value card-neg">{neg_pct}%</div>'
            f'<div class="metric-label">Negative</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### 🔍 Top Complaint Aspects Breakdown (from Negative Reviews)")

    aspects_list = get_product_aspects(selected_asin)
    if aspects_list:
        aspect_cols = st.columns(2)
        for i, asp in enumerate(aspects_list[:6]):
            col = aspect_cols[i % 2]
            pct = asp.get("percentage", 0.0)
            cnt = asp.get("count", 0)
            name = asp.get("display_name", asp["aspect"])
            with col:
                st.write(f"**{name}** — `{pct}%` of negative reviews ({cnt} mentions)")
                st.progress(min(pct / 100.0, 1.0))
    else:
        st.info("No concentrated complaint clusters detected for this product.")

    st.write("")
    st.markdown("#### 📝 Sample Customer Reviews")
    prod_reviews_res = retriever.retrieve_reviews(selected_asin, query="customer reviews", top_k=4)
    sample_reviews = prod_reviews_res.get("retrieved_reviews", [])
    for r in sample_reviews:
        st.markdown(
            f"""
            <div class="evidence-card">
                <b>{r.get('title', '')}</b> (Rating: <b>{r.get('rating', '')} ★</b> | Sentiment: <code>{r.get('sentiment', '')}</code>)<br>
                <span>"{r.get('text', '')}"</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================================
# TAB 3: SMART RECOMMENDATIONS & ALTERNATIVES
# =========================================================================
with tab_recs:
    st.subheader("🎯 Aspect-Aware Product Recommendations")
    st.write("Discover the highest-satisfaction products filtered by domain and specific customer aspect excellence.")

    rc_col1, rc_col2, rc_col3 = st.columns(3)
    with rc_col1:
        rec_domain = st.selectbox("Category Domain:", ["All", "Electronics", "Fashion"], index=1 if selected_domain == "Electronics" else 2)
    with rc_col2:
        aspect_focus_options = ["None (Overall Best)"] + list(ASPECT_DISPLAY_NAMES.values())
        chosen_aspect_disp = st.selectbox("Focus Aspect:", aspect_focus_options, index=0)
    with rc_col3:
        min_rev_threshold = st.slider("Min Review Volume:", min_value=1, max_value=50, value=5)

    aspect_key = None
    if chosen_aspect_disp != "None (Overall Best)":
        for k, v in ASPECT_DISPLAY_NAMES.items():
            if v == chosen_aspect_disp:
                aspect_key = k
                break

    target_domain = None if rec_domain == "All" else rec_domain
    recommendations = recommender.recommend_products(
        domain=target_domain,
        min_rating=4.0,
        min_reviews=min_rev_threshold,
        aspect_focus=aspect_key,
        top_k=6,
    )

    if recommendations:
        for idx, r in enumerate(recommendations, 1):
            with st.container():
                st.markdown(
                    f"### {idx}. {r['title']} `[{r['parent_asin']}]`"
                )
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                r_c1.metric("Rating", f"{r['average_rating']} ★")
                r_c2.metric("Total Reviews", r["total_reviews"])
                r_c3.metric("Positive Sentiment", f"{r['positive_percentage']}%")
                r_c4.metric("Recommendation Score", r["recommendation_score"])

                if r["complaint_highlights"]:
                    st.caption(f"Top issue areas: {', '.join(r['complaint_highlights'])}")
                else:
                    st.caption("Clean record: No prominent complaint patterns.")
                st.markdown("---")
    else:
        st.info("No products match the selected criteria. Try lowering the review threshold.")

    # Alternative finder for active product
    st.write("")
    st.markdown("#### 🔄 Recommended Alternatives for Current Product")
    if st.button(f"Find Better Alternatives for '{prod_title[:45]}...'"):
        alts_res = recommender.recommend_alternatives(selected_asin, top_k=3)
        curr_p = alts_res["current_product"]
        st.write(f"Current product rating: **{curr_p['average_rating']} ★** with **{curr_p['negative_percentage']}%** negative reviews.")
        for a in alts_res["alternatives"]:
            st.markdown(
                f"- **{a['title']}** (`{a['parent_asin']}`): **{a['average_rating']} ★** | **{a['positive_percentage']}%** positive | Score: `{a['recommendation_score']}`"
            )

# =========================================================================
# TAB 4: HEAD-TO-HEAD COMPARISON
# =========================================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side Product Comparison")
    st.write("Compare customer sentiment, rating distributions, and complaint aspects between any two products in the catalog.")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        comp_p1 = st.selectbox(
            "Product A:",
            options=list(product_options.keys()),
            format_func=lambda asin: product_options[asin],
            index=0,
            key="comp_p1_select",
        )
    with comp_col2:
        default_p2_idx = 1 if len(product_options) > 1 else 0
        comp_p2 = st.selectbox(
            "Product B:",
            options=list(product_options.keys()),
            format_func=lambda asin: product_options[asin],
            index=default_p2_idx,
            key="comp_p2_select",
        )

    if comp_p1 and comp_p2:
        if comp_p1 == comp_p2:
            st.warning("Please select two different products to perform a comparison.")
        else:
            try:
                comp_data = recommender.compare_products(comp_p1, comp_p2)
                p1_d, p2_d = comp_data["product_1"], comp_data["product_2"]

                # Side by side table
                cmp_df = pd.DataFrame({
                    "Metric": ["Title", "Domain", "Average Rating", "Total Reviews", "Positive Sentiment", "Neutral Sentiment", "Negative Sentiment"],
                    f"Product A ({p1_d['parent_asin']})": [
                        p1_d["title"],
                        p1_d["domain"],
                        f"{p1_d['average_rating']} ★",
                        p1_d["total_reviews"],
                        f"{p1_d['positive_percentage']}%",
                        f"{p1_d['neutral_percentage']}%",
                        f"{p1_d['negative_percentage']}%",
                    ],
                    f"Product B ({p2_d['parent_asin']})": [
                        p2_d["title"],
                        p2_d["domain"],
                        f"{p2_d['average_rating']} ★",
                        p2_d["total_reviews"],
                        f"{p2_d['positive_percentage']}%",
                        f"{p2_d['neutral_percentage']}%",
                        f"{p2_d['negative_percentage']}%",
                    ],
                })
                st.table(cmp_df)

                # Verdict Callout
                st.success(f"**Verdict**: {comp_data['verdict']}")

            except Exception as e:
                st.error(f"Error comparing products: {e}")

# Footer
st.markdown("---")
st.caption("FeedbackIQ — AI Customer Feedback Intelligence System • Powered by FAISS, Sentence Transformers & Aspect Extraction")
