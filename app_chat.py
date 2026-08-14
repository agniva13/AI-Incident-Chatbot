import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import random
from datetime import datetime
from datasets import load_dataset
# ====================== PAGE CONFIGURATION ======================
st.set_page_config(page_title="AI Incident Chatbot", page_icon="🔧", layout="centered")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "problem_dictionary" not in st.session_state:
    st.session_state.problem_dictionary = {}
if "resolution_dictionary" not in st.session_state:
    st.session_state.resolution_dictionary = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# ====================== SIDEBAR ADDITIONS ======================
with st.sidebar:
    st.header("Chat History")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_chat_id = None
        st.rerun()
    if st.session_state.problem_dictionary:
        for chat_id, problem in reversed(list(st.session_state.problem_dictionary.items())):
            history_label = problem[:60] + ("…" if len(problem) > 60 else "")
            if st.button(f"☰  {history_label}", key=f"history_{chat_id}", use_container_width=True):
                st.session_state.active_chat_id = chat_id
                st.session_state.messages = [
                    {"role": "user", "content": st.session_state.problem_dictionary[chat_id]},
                    {"role": "assistant", "content": st.session_state.resolution_dictionary[chat_id]},
                ]
                st.rerun()
    else:
        st.caption("No incidents resolved yet.")
    st.divider()
    dark_mode = st.toggle("Dark mode", value=st.session_state.theme == "dark")
    selected_theme = "dark" if dark_mode else "light"
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

# ====================== THEME-AWARE ORIGINAL STYLING ======================
dark = st.session_state.theme == "dark"
bg = "#000000" if dark else "#FFFFFF"
panel = "#1A1A1A" if dark else "#F5F7FA"
text = "#FFFFFF" if dark else "#202124"
muted = "#888888" if dark else "#5F6368"
border = "#333333" if dark else "#DADCE0"
message_bg = "#0A0A0A" if dark else "#F8F9FA"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .stApp header {{ background-color: {bg}; }}
    [data-testid="stHeader"] {{ background-color: rgba(0, 0, 0, 0); }}
    [data-testid="stSidebar"] {{ background-color: {panel}; }}
    [data-testid="stSidebar"] * {{ color: {text} !important; }}
    h1, h2, h3, p, label, .stMarkdown {{ color: {text} !important; }}
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {{ background-color: {panel}; color: {text}; border-color: {border}; }}
    .stButton > button {{ background-color: #0066FF; color: {text}; font-weight: bold; }}
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"], [data-testid="stBottom"] > div, .stChatFloatingInputContainer {{ background-color: {bg} !important; border: none !important; box-shadow: none !important; }}
    [data-testid="stChatInput"] {{ background-color: transparent !important; border: none !important; box-shadow: none !important; }}
    [data-testid="stChatInput"] > div {{ background-color: {panel} !important; border: 1px solid {border} !important; border-radius: 12px; position: relative !important; }}
    [data-testid="stChatInput"] textarea {{ background-color: {panel} !important; color: {text} !important; }}
    [data-testid="stChatInput"] textarea::placeholder {{ color: {muted} !important; }}
    [data-testid="stChatMessage"] {{ background-color: {message_bg}; border: 1px solid {border}; }}
    [data-testid="stMetricValue"] {{ color: {text} !important; }}
    .stCaption {{ color: {muted} !important; }}
    .voice-mic-btn {{ display:flex; align-items:center; justify-content:center; width:2.25rem; height:2.25rem; padding:0; border:none; border-radius:8px; background:transparent; color:{muted}; cursor:pointer; flex-shrink:0; pointer-events:auto !important; position:relative; z-index:1000; }}
    .voice-mic-btn:hover {{ color:{text}; background-color:rgba(128,128,128,.12); }}
    .voice-mic-btn.listening {{ color:#FF5555; background-color:rgba(255,85,85,.15); }}
    </style>
""", unsafe_allow_html=True)

# ====================== API KEYS ======================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# ====================== DARK THEME ======================
# The original UI and voice recorder continue below unchanged.
#prompt = st.chat_input("Describe the #incident...")
# inject_voice_recorder()
st.title("🔧 AI Incident Chatbot")
# ====================== KNOWLEDGE BASE ======================
@st.cache_resource
def load_knowledge():
    servicenow_ds = load_dataset(
    "6StringNinja/synthetic-servicenow-incidents"
)

    servicenow_df = servicenow_ds["train"].to_pandas()

    servicenow_data = pd.DataFrame({
    "problem_description": servicenow_df["description"],
    "resolution": servicenow_df["resolution"]
})


# ============================================================
# 2. LOAD DEVOPS INCIDENT RESPONSE
# ============================================================

    devops_ds = load_dataset(
    "Snaseem2026/devops-incident-response"
)

    devops_df = devops_ds["train"].to_pandas()

    devops_data = pd.DataFrame({
    "problem_description": devops_df["description"],
    "resolution": devops_df["resolution_steps"]
})


# ============================================================
# 3. COMBINE BOTH DATASETS
# ========================================================
    combined_df = pd.concat(
    [servicenow_data, devops_data],
    ignore_index=True
)


# ============================================================
# 4. CLEAN THE DATA
# ============================================================

    combined_df["problem_description"] = (
    combined_df["problem_description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

    combined_df["resolution"] = (
    combined_df["resolution"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove rows where either field is empty

    combined_df = combined_df[
    (combined_df["problem_description"] != "") &
     (combined_df["resolution"] != "")
]


# Remove duplicate problem-resolution pairs

    combined_df = combined_df.drop_duplicates(
    subset=["problem_description", "resolution"]
).reset_index(drop=True)


# ============================================================
# 5. CREATE THE DICTIONARY
# ============================================================

    data = {
    "problem_description":
        combined_df["problem_description"].tolist(),

    "resolution":
        combined_df["resolution"].tolist()
}
    df = pd.DataFrame(data)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['embedding'] = df['problem_description'].apply(lambda x: model.encode(x))
    embeddings = np.vstack(df['embedding'].values)
    return df, embeddings, model, data

df, embeddings, embed_model, knowledge_data = load_knowledge()

# ====================== CHAT HISTORY ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
prompt = st.chat_input("Describe the incident...")
#inject_voice_recorder()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating Resolution..."):
            query_emb = embed_model.encode(prompt)
            similarities = cosine_similarity([query_emb], embeddings)[0]
            top_idx = np.argmax(similarities)
            confidence = similarities[top_idx]
            retrieved = df.iloc[top_idx]['resolution']
            # Threshold logic
            if confidence < 0.60:
                sarvam_res = "❌ Can't help with this incident. Confidence too low."
                confidence = 0.0
            else:
                retrieved = df.iloc[top_idx]["resolution"]

            full_prompt = f"""You are an expert Site Reliability Engineer.
New incident: "{prompt}"
Similar past resolution: {retrieved}
Provide clear, actionable, step-by-step resolution."""

            # Use Sarvam when configured; otherwise keep the original knowledge-base resolution.
            if SARVAM_API_KEY and confidence >= 0.60:
                try:
                    sarvam_client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=SARVAM_API_KEY)
                    response = sarvam_client.chat.completions.create(
                        model="sarvam-105b",
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.3,
                        max_tokens=600,
                    )
                    msg = response.choices[0].message
                    sarvam_res = (msg.content or getattr(msg, "reasoning_content", "")).strip()
                    if not sarvam_res:
                        sarvam_res = retrieved
                except Exception as e:
                    sarvam_res = f"{retrieved}\n\nSarvam unavailable: {e}"
            elif confidence >= 0.60:
                sarvam_res = retrieved

            st.markdown("## 🛠️ Resolution")
            st.markdown(sarvam_res)
            st.markdown("---")

            # Append the new incident to the original knowledge-base dictionaries.
            knowledge_data["problem_description"].append(prompt)
            knowledge_data["resolution"].append(sarvam_res)

            # Add the new embedding so future searches can use this incident.
            new_embedding = embed_model.encode(prompt)
            df.loc[len(df)] = {
                "problem_description": prompt,
                "resolution": sarvam_res,
                "embedding": new_embedding,
            }
            embeddings = np.vstack([embeddings, new_embedding])

            # Store both values under the same ID so they remain paired.
            chat_id = f"{datetime.now().isoformat(timespec='seconds')}-{len(st.session_state.problem_dictionary) + 1}"
            st.session_state.problem_dictionary[chat_id] = prompt
            st.session_state.resolution_dictionary[chat_id] = sarvam_res
            st.session_state.active_chat_id = chat_id
            st.session_state.messages.append({"role": "assistant", "content": sarvam_res})

