import os
import requests
import streamlit as st

API_BASE = os.getenv("KGRA_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="KGRA", layout="wide")
st.title("KGRA -- Knowledge Graph & Retrieval Augmented")

tab_ingest, tab_query, tab_graph, tab_admin = st.tabs(["Ingest", "Query", "Graph Explorer", "Admin"])

with tab_ingest:
    st.subheader("Ingest documents from data/pdfs")
    if st.button("Start ingestion"):
        r = requests.post(f"{API_BASE}/ingest")
        st.json(r.json())
    if st.button("Check status"):
        r = requests.get(f"{API_BASE}/ingest/status")
        st.json(r.json())

with tab_query:
    st.subheader("Ask a question")
    question = st.text_input("Question")
    top_k = st.slider("top_k", 1, 10, 5)
    graph_depth = st.slider("graph depth", 0, 4, 2)
    if st.button("Ask") and question:
        r = requests.post(f"{API_BASE}/query", json={"question": question, "top_k": top_k, "graph_depth": graph_depth})
        data = r.json()
        st.markdown("### Answer")
        st.write(data.get("answer"))
        st.markdown("### Retrieved chunks")
        for c in data.get("chunks", []):
            st.markdown(f"**{c.get('source')}** (score {c.get('score', 0):.3f})")
            st.caption(c.get("text", "")[:500])
        st.markdown("### Graph facts used")
        st.json(data.get("subgraph"))

with tab_graph:
    st.subheader("Explore the knowledge graph")
    q = st.text_input("Search entities")
    if st.button("Search entities") and q:
        st.json(requests.get(f"{API_BASE}/graph/search", params={"q": q}).json())
    name = st.text_input("Entity name for neighborhood")
    depth = st.slider("neighborhood depth", 0, 4, 2, key="entity_depth")
    if st.button("Get neighborhood") and name:
        st.json(requests.get(f"{API_BASE}/graph/entity/{name}", params={"depth": depth}).json())

with tab_admin:
    st.subheader("Admin")
    st.json(requests.get(f"{API_BASE}/health").json())
    if st.button("Reset everything (Qdrant + Neo4j + BM25)"):
        st.json(requests.delete(f"{API_BASE}/reset").json())
