import streamlit as st
import pandas as pd
import os
import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel
from utils.sentiment import load_sentiment_model, predict_sentiment
from utils.clustering import load_clustering_model, apply_clustering, reduce_clusters
from utils.gpt_summary import generate_category_title, summarize_category
from utils.pdf_utils import generate_product_pdf, zip_summaries
import openai

openai.api_key = "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"

st.title("\ud83d\udcca Product Review Analyzer")

uploaded_csv = st.file_uploader("Upload CSV file with product reviews", type=["csv"])

@st.cache_resource
def load_embedding_model():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

def get_bert_embeddings(texts):
    all_embeddings = []
    tokenizer, model = load_embedding_model()
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    return np.concatenate(all_embeddings, axis=0)

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)

    model, tokenizer = load_sentiment_model()
    df["clean_review"] = df["reviews.text"].astype(str)
    df["sentiment_label"] = predict_sentiment(df["clean_review"], model, tokenizer)

    df["text_for_clustering"] = df["name"].fillna("") + " " + df["categories"].fillna("")
    embeddings = get_bert_embeddings(df["text_for_clustering"].tolist())

    cluster_model = load_clustering_model("models/sentiment_model/hdbscan_Final_Model.pkl")
    original_labels = apply_clustering(cluster_model, embeddings)

    threshold = st.slider("Cosine Similarity Threshold", 0.80, 0.99, 0.92, 0.01)
    reduced_labels = reduce_clusters(embeddings, original_labels, similarity_threshold=threshold)

    unique_clusters = sorted(set(reduced_labels[reduced_labels >= 0]))
    cluster_to_meta = {cl: f"Category {i}" for i, cl in enumerate(unique_clusters)}
    meta_categories = [cluster_to_meta.get(cl, "Noise") if cl >= 0 else "Noise" for cl in reduced_labels]

    df_result = pd.DataFrame({
        "original_cluster": original_labels,
        "reduced_cluster": reduced_labels,
        "meta_category": meta_categories
    })

    df_combined = df.copy()
    df_combined["meta_category"] = df_result["meta_category"].values

    st.markdown("### \ud83d\udcc8 Category Distribution")
    category_counts = df_combined["meta_category"].value_counts().reset_index()
    st.bar_chart(category_counts.set_index("index"))

    selected_category = st.selectbox("Choose a Meta Category:", sorted(df_combined["meta_category"].unique()))
    if selected_category:
        cat_df = df_combined[df_combined["meta_category"] == selected_category]
        product_names = cat_df["name"].dropna().unique().tolist()[:10]
        products_info = cat_df[["name", "reviews.text", "reviews.rating"]].dropna().astype(str).agg(" | ".join, axis=1).tolist()[:10]

        title = generate_category_title(selected_category, product_names)
        summary = summarize_category(selected_category, products_info)
        st.markdown(f"## \ud83c\udff7\ufe0f {title}")
        st.markdown(summary)

        pdf_bytes = generate_product_pdf(title, summary, filename=f"{selected_category}_summary.pdf", save_path="summaries")
        st.download_button("\ud83d\udcc5 Download Summary as PDF", data=pdf_bytes, file_name=f"{selected_category}_summary.pdf", mime="application/pdf")

        selected_product = st.selectbox("Choose a product to summarize its reviews:", cat_df["name"].dropna().unique())
        if selected_product:
            product_reviews = cat_df[cat_df["name"] == selected_product]["reviews.text"].dropna().astype(str).tolist()[:15]
            review_prompt = f"""You are a review analyst. Summarize the following reviews for product: '{selected_product}':\n\n{chr(10).join(product_reviews)}\n\nYour summary should include:\n- General sentiment\n- Top compliments and complaints\n- Who is this product best for?"""
            response = openai.Completion.create(engine="text-davinci-003", prompt=review_prompt, max_tokens=200)
            product_summary = response.choices[0].text.strip()
            st.markdown(f"### \ud83e\uddd2 Summary for **{selected_product}**")
            st.markdown(product_summary)

            product_pdf = generate_product_pdf(selected_product, product_summary, save_path="summaries")
            st.download_button("\ud83d\udcc5 Download Product Summary as PDF", data=product_pdf, file_name=f"{selected_product}_summary.pdf", mime="application/pdf")

    if os.path.exists("summaries"):
        zip_file = zip_summaries("summaries")
        st.download_button("\ud83d\udce6 Download All Summaries (ZIP)", data=zip_file, file_name="all_summaries.zip", mime="application/zip")
")
