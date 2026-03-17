import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import os

# 1. Setup App Look & Feel
st.set_page_config(page_title="Patent Intelligence", page_icon="logo.png", layout="wide")

# 2. ALL-IN-ONE CLOUD DATABASE BUILDER
@st.cache_resource(show_spinner="Cloud Engine is initializing the AI database. Please wait...")
def initialize_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chroma_client = chromadb.Client() 
    
    # --- THE FIX: Clean up any "ghost" databases from previous crashes ---
    try:
        chroma_client.delete_collection(name="uae_patents")
    except:
        pass # If it doesn't exist, just ignore and move on
        
    collection = chroma_client.create_collection(name="uae_patents")
    # ---------------------------------------------------------------------
    
    if os.path.exists("patents.zip"):
        df = pd.read_csv("patents.zip", compression='zip')
    elif os.path.exists("patents.csv"):
        df = pd.read_csv("patents.csv")
    else:
        st.error("🚨 Error: The database file could not be found. Please ensure 'patents.zip' or 'patents.csv' is uploaded to your GitHub repository.")
        st.stop()
    
    # Fill empty cells to prevent crashes
    df = df.fillna("N/A")
    
    # Optional: Remove exact duplicate rows from the spreadsheet to save memory
    df = df.drop_duplicates(subset=['Application Number', 'Title in English'])
    
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
        
        # We add the row index to the ID so it is 100% mathematically unique!
        unique_id = f"{row['Application Number']}_row_{index}"
        ids.append(unique_id)
        
    embeddings = model.encode(documents).tolist()
    
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    return model, collection

# Boot up the system!
model, collection = initialize_system()

def scrape_moe_claims(patent_id):
    return f"Live claims for Application {patent_id} will be fetched here from the MoE portal."

# 3. Sidebar & Login System
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("### Your Company Logo")
        
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
        else:
            st.error("Please enter a password.")

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

    if st.button("Search Database"):
        if final_search_text:
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
                        # We use metadata to show the real Application Number, not the hidden ID
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
        else:
             st.error("Please enter text or upload a file first.")
else:
    st.info("Please log in on the left menu to use the search tool.")
