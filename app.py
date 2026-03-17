import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import os

# 1. Setup App Look & Feel
st.set_page_config(page_title="Patent Intelligence", page_icon="logo.png", layout="wide")

# 2. ALL-IN-ONE CLOUD DATABASE BUILDER
# This tells Streamlit to only build the database once and remember it while the app is awake
@st.cache_resource(show_spinner="Cloud Engine is reading patents.zip and building the AI database. Please wait...")
def initialize_system():
    # Load AI Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Setup Temporary Cloud Database
    chroma_client = chromadb.Client() # In-memory database for the cloud
    collection = chroma_client.create_collection(name="uae_patents")
    
    # Read the zipped file directly
    df = pd.read_csv("patents.zip", compression='zip')
    df = df.fillna("N/A")
    
    documents = []
    metadatas = []
    ids = []
    
    for index, row in df.iterrows():
        combined_text = f"Title: {row['Title in English']}. Abstract: {row['Abstract in English']}"
        documents.append(combined_text)
        
        metadatas.append({
            "Application Number": str(row['Application Number']),
            "Application Date": str(row['Application Date']),
            "Title": str(row['Title in English']),
            "Abstract": str(row['Abstract in English']),
            "Priority Date": str(row['Priority Date']),
            "Earliest Priority Date": str(row['Earliest Priority Date']),
            "Priority Country": str(row['Country Name (Priority)']),
            "Priority Number": str(row['Priority Number'])
        })
        ids.append(str(row['Application Number']))
        
    # Convert text to vectors
    embeddings = model.encode(documents).tolist()
    
    # Save to cloud database
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    return model, collection

# Boot up the system!
model, collection = initialize_system()

def scrape_moe_claims(patent_id):
    return f"Live claims for Application {patent_id} will be fetched here from the MoE portal."

# 3. Sidebar & Login System
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### User Sign In")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if 'account_type' not in st.session_state:
        st.session_state.account_type = None
        
    if st.button("Log In / Sign Up"):
        if password == "premium":
            st.session_state.account_type = "Premium"
            st.success("Welcome, Premium User!")
        elif password != "":
            st.session_state.account_type = "Free"
            st.success("Welcome, Free User!")

# 4. Main Search Interface
st.title("AI-Powered Patent Search")

if st.session_state.account_type:
    st.markdown("Search our database to find similar patent applications.")
    
    user_query = st.text_area("1. Type keywords or abstract here:")
    uploaded_file = st.file_uploader("2. OR Upload Patent File (PDF/TXT)", type=["txt", "pdf"])
    
    extracted_text = ""
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            extracted_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()
                
    final_search_text = extracted_text if extracted_text else user_query

    if st.button("Search Database") and final_search_text:
        with st.spinner("Scanning database..."):
            query_vector = model.encode([final_search_text]).tolist()
            results = collection.query(query_embeddings=query_vector, n_results=10)
            
            st.markdown("---")
            st.subheader("Search Results")
            
            if st.session_state.account_type == "Free":
                st.warning("🔒 Free Tier Restricted View")
                st.info(f"We found **{len(results['ids'][0])} similar patent applications**.")
                st.write("🌟 **Upgrade to a Premium Account** to view full details.")
                
            elif st.session_state.account_type == "Premium":
                st.success("🔓 Premium Access: Showing Full Detailed Records")
                for i in range(len(results['ids'][0])):
                    app_id = results['ids'][0][i]
                    meta = results['metadatas'][0][i]
                    
                    with st.expander(f"Hit #{i+1}: {meta['Title']} (App Number: {meta['Application Number']})"):
                        st.markdown(f"**Application Number:** {meta['Application Number']}")
                        st.markdown(f"**Application Date:** {meta['Application Date']}")
                        st.markdown(f"**Title:** {meta['Title']}")
                        st.markdown(f"**Abstract:** {meta['Abstract']}")
                        st.markdown(f"**Priority Date:** {meta['Priority Date']}")
                        st.markdown(f"**Earliest Priority Date:** {meta['Earliest Priority Date']}")
                        st.markdown(f"**Priority Country:** {meta['Priority Country']}")
                        st.markdown(f"**Priority Number:** {meta['Priority Number']}")
                        st.markdown("---")
                        st.markdown("### 🌐 Fetched Claims")
                        st.code(scrape_moe_claims(meta['Application Number']))
    elif st.button("Search Database") and not final_search_text:
         st.error("Please enter text or upload a file first.")
else:
    st.info("Please log in on the left menu to use the search tool.")
