import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

# --- PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="ProSearch | Patent Intelligence",
    page_icon="logo.png", # Your logo in the browser tab!
    layout="wide",        # Uses the full width of the screen
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR SLEEK LOOK ---
st.markdown("""
    <style>
    /* Makes buttons look modern */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #004aad;
        color: white;
    }
    /* Cleans up the top padding */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- SETUP AI & DATABASE ---
@st.cache_resource
def load_ai():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai()
chroma_client = chromadb.PersistentClient(path="./patent_database")
collection = chroma_client.get_or_create_collection(name="uae_patents")

def scrape_moe_claims(patent_id):
    try:
        url = f"https://example-moe-site.gov.ae/patents/{patent_id}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        claims_box = soup.find('div', class_='patent-claims')
        return claims_box.text.strip() if claims_box else "Claims not found online."
    except Exception as e:
        return f"Error connecting to MoE: {e}"

# --- SIDEBAR (LOGIN & NAVIGATION) ---
with st.sidebar:
    # Display your actual logo image
    st.image("logo.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### User Authentication")
    
    # Sleek login toggle
    account_choice = st.radio("Access Level", ["Free Account", "Premium Account"])
    st.session_state.account_type = account_choice
    
    st.markdown("---")
    st.caption("© 2026 Your Company Name. All rights reserved.")

# --- MAIN PAGE DASHBOARD ---
st.title("Patent Intelligence Platform")
st.markdown("Identify similar patent applications and analyze freedom to operate.")

# Use a sleek container for the search bar
with st.container():
    user_query = st.text_area("Enter abstract, keywords, or paste your application details here:", height=150)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_clicked = st.button("🚀 Analyze Database")

st.markdown("---")

# --- SEARCH LOGIC ---
if search_clicked and user_query:
    with st.spinner("Analyzing proprietary vector database..."):
        query_vector = model.encode([user_query]).tolist()
        
        results = collection.query(
            query_embeddings=query_vector,
            n_results=5 
        )
        
        st.subheader("Analysis Results")
        
        # FREE TIER
        if st.session_state.account_type == "Free Account":
            st.warning("🔒 Limited Free View")
            st.metric(label="Highly Similar Patents Found", value=len(results['ids'][0]))
            st.info("Upgrade to a Premium Account to unlock full patent titles, abstracts, classifications, and live claims scraping.")
        
        # PREMIUM TIER
        elif st.session_state.account_type == "Premium Account":
            st.success("🔓 Premium Access Authorized")
            
            for i in range(len(results['ids'][0])):
                p_id = results['ids'][0][i]
                meta = results['metadatas'][0][i]
                
                with st.expander(f"📄 Match: {meta['title']} (ID: {p_id})"):
                    st.markdown(f"**Classification:** `{meta['classification']}`")
                    st.markdown(f"**Abstract:** {meta['abstract']}")
                    
                    st.markdown("#### 🌐 Live Claims (MoE Portal)")
                    live_claims = scrape_moe_claims(p_id)
                    st.code(live_claims, language="text")
