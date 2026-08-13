import streamlit as st
import pandas as pd
import numpy as np
from sentence-transformers==3.3.1 import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import random
from datetime import datetime

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


def inject_voice_recorder():
    """Inject a mic button into the chat input (browser speech-to-text)."""
    st.html(f"""
        <div id="voice-recorder-anchor" style="display:none"></div>
        <script>
        (function() {{
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            function setTextareaValue(textarea, value) {{
                const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
                setter.call(textarea, value);
                textarea.dispatchEvent(new Event("input", {{ bubbles: true }}));
                textarea.dispatchEvent(new Event("change", {{ bubbles: true }}));
            }}
            function attachVoiceRecorder() {{
                const chatInput = document.querySelector('[data-testid="stChatInput"]');
                if (!chatInput || chatInput.querySelector('.voice-mic-btn')) return;
                const textarea = chatInput.querySelector('textarea');
                if (!textarea) return;
                const micBtn = document.createElement('button');
                micBtn.type = 'button'; micBtn.className = 'voice-mic-btn'; micBtn.textContent = '🎙';
                micBtn.title = 'Voice input';
                const submitBtn = chatInput.querySelector('[data-testid="stChatInputSubmitButton"]');
                if (submitBtn && submitBtn.parentElement) submitBtn.parentElement.insertBefore(micBtn, submitBtn);
                if (!SpeechRecognition) {{ micBtn.disabled = true; return; }}
                const recognition = new SpeechRecognition(); recognition.continuous = false; recognition.interimResults = true; recognition.lang = 'en-US';
                micBtn.onclick = function() {{ micBtn.classList.add('listening'); recognition.start(); }};
                recognition.onresult = function(event) {{ let text = ''; for (let i=0;i<event.results.length;i++) text += event.results[i][0].transcript; setTextareaValue(textarea, text.trim()); }};
                recognition.onend = function() {{ micBtn.classList.remove('listening'); }};
            }}
            attachVoiceRecorder();
            new MutationObserver(attachVoiceRecorder).observe(document.body, {{ childList:true, subtree:true }});
        }})();
        </script>
        <!-- voice-recorder-{random.random()} -->
    """, unsafe_allow_javascript=True)
st.title("🔧 AI Incident Chatbot")
# ====================== KNOWLEDGE BASE ======================
@st.cache_resource
def load_knowledge():
    data = {
    "problem_description": [
        "High CPU usage on production server",
        "Database connection timeout errors",
        "Service returning 502 Bad Gateway",
        "Application throwing 500 Internal Server Error",
        "Memory leak causing pod restarts",
        "Slow API response times",
        "Redis connection refused",
        "Kubernetes pod in CrashLoopBackOff",
        "Disk space usage at 95%",
        "Network latency between services",
        "CPU spikes during peak traffic",
        "MySQL deadlocks occurring frequently",
        "API gateway returning 504 Gateway Timeout",
        "Frontend not loading static assets",
        "Microservice failing health checks",
        "Container image pull failure",
        "Authentication service downtime",
        "High GC (garbage collection) pauses",
        "Kafka consumer lag increasing",
        "Elasticsearch cluster red status",
        "S3 file upload failures",
        "Load balancer not distributing traffic evenly",
        "TLS handshake failures",
        "DNS resolution failures",
        "High memory utilization on nodes",
        "Docker container exits immediately",
        "CI/CD pipeline failing builds",
        "Webhook delivery failures",
        "Session expiry issues in production",
        "Rate limiting blocking valid users",
        "Database replication lag",
        "High disk I/O wait times",
        "Application freezing intermittently",
        "Background jobs not executing",
        "Cron jobs failing execution",
        "Message queue backlog increasing",
        "Service discovery failures",
        "Ingress controller misrouting traffic",
        "API returning inconsistent data",
        "Cache invalidation issues",
        "High error rate in logs",
        "Unexpected service restarts",
        "SSL certificate expired",
        "Memory fragmentation issues",
        "CPU throttling in containers",
        "Pod scheduling failures",
        "Node not joining cluster",
        "Persistent volume mount failures",
        "File permission errors",
        "API latency spikes",
        "Database query performance degradation",
        "Sudden traffic surge causing overload",
        "Application dependency downtime",
        "Third-party API failures",
        "Timeout errors in batch processing",
        "High network packet loss",
        "Service mesh misconfiguration",
        "Configuration drift across environments",
        "Broken deployment after release",
        "Rollback failures",
        "High authentication failure rate",
        "OAuth token validation errors",
        "JWT signature verification failure",
        "Session store unavailability",
        "WebSocket connection drops",
        "Frontend API mismatch errors",
        "Backend schema migration failure",
        "Data corruption in storage",
        "Logging system not capturing logs",
        "Monitoring alerts not triggering",
        "High latency in database writes",
        "Read replicas not syncing",
        "API returning stale data",
        "Service stuck in pending state",
        "Autoscaling not triggering",
        "Container runtime errors",
        "Image registry access denied",
        "Firewall blocking internal traffic",
        "Memory limit exceeded errors",
        "CPU limit exceeded errors",
        "Server overheating issues",
        "Virtual machine reboot loops",
        "Cloud provider service outage impact",
        "Load balancer health check failures",
        "API version mismatch errors",
        "Deprecated library causing crashes",
        "High thread contention in application",
        "Race condition causing inconsistent output",
        "Dead service endpoint still receiving traffic",
        "Improper retry causing request storms",
        "Application unable to establish secure HTTPS connection",
        "High CPU utilization on Kubernetes worker node",
        "Database backup job completed with errors",
        "Redis cache synchronization delay",
        "API authentication requests failing intermittently",
        "Docker container entering Restarting state repeatedly",
        "Disk I/O latency affecting application performance",
        "Load balancer routing traffic to unhealthy instances",
        "Microservice unable to connect to message broker",
        "SSL certificate chain validation failed"
    ],

    "resolution": [
        "Identify CPU-heavy processes using top/htop, optimize or scale service horizontally.",
        "Increase DB pool size, check network latency, and verify DB server health.",
        "Check upstream services, restart reverse proxy, and validate routing rules.",
        "Inspect application logs, fix code exceptions, and restart affected pods.",
        "Profile memory usage, fix leaks, and increase memory limits temporarily.",
        "Add caching, optimize queries, and enable load balancing.",
        "Verify Redis service status, check configs, and restart Redis instance.",
        "Inspect pod logs, fix startup error, and redeploy container.",
        "Clean logs and unused files, or expand disk storage.",
        "Check network routes and optimize inter-service communication.",
        "Enable autoscaling and optimize CPU-intensive workloads.",
        "Analyze lock contention, optimize queries, and reduce transaction scope.",
        "Scale API gateway and optimize upstream response times.",
        "Check CDN configuration and fix missing asset paths.",
        "Fix liveness/readiness probes and restart failing service.",
        "Fix container registry credentials and retry image pull.",
        "Restart auth service and verify identity provider health.",
        "Tune JVM GC settings and optimize memory allocation.",
        "Increase consumer partitions and optimize Kafka consumers.",
        "Rebalance Elasticsearch shards and add cluster nodes.",
        "Check IAM permissions and fix S3 bucket policies.",
        "Adjust load balancing algorithm and health checks.",
        "Fix TLS certificates and update cipher configurations.",
        "Fix DNS records and clear caching resolver.",
        "Scale node group or optimize memory usage.",
        "Fix Docker entrypoint or missing dependencies.",
        "Fix pipeline scripts and dependency versions.",
        "Verify webhook endpoint availability and retry logic.",
        "Fix session store configuration and TTL settings.",
        "Adjust rate limiting thresholds.",
        "Optimize replication configuration and network throughput.",
        "Reduce disk writes and clean temporary files.",
        "Fix thread deadlocks and optimize concurrency.",
        "Restart job scheduler and verify worker status.",
        "Fix cron expressions and system timezone.",
        "Scale message brokers and increase partitions.",
        "Fix service registry health checks.",
        "Correct ingress routing rules.",
        "Fix API versioning mismatch.",
        "Invalidate and rebuild cache.",
        "Analyze logs and fix root cause of errors.",
        "Enable auto-restart policies and fix instability.",
        "Renew SSL certificates and update configuration.",
        "Restart memory allocator and optimize usage.",
        "Increase CPU limits and reduce workload spikes.",
        "Fix scheduler constraints and node labels.",
        "Join node to cluster using correct credentials.",
        "Fix volume mount permissions and storage class.",
        "Correct file permissions using chmod/chown.",
        "Optimize backend queries and indexing.",
        "Add indexes and optimize database queries.",
        "Enable autoscaling and queue requests.",
        "Replace or restart failing dependencies.",
        "Retry API calls and add fallback logic.",
        "Parallelize batch jobs and optimize execution.",
        "Check network infrastructure and reduce packet loss.",
        "Fix service mesh routing policies.",
        "Standardize config management across environments.",
        "Rollback faulty deployment.",
        "Fix CI/CD pipeline and redeploy stable version.",
        "Fix authentication backend and rate limits.",
        "Fix OAuth provider configuration.",
        "Correct JWT secret mismatch.",
        "Restart session store service.",
        "Fix WebSocket keepalive settings.",
        "Align frontend-backend API contracts.",
        "Fix migration script and rerun safely.",
        "Restore data from backup and fix integrity.",
        "Enable log shipping and collectors.",
        "Fix alert rules and monitoring config.",
        "Optimize write path and indexing.",
        "Fix replication lag and network bottlenecks.",
        "Invalidate stale cache entries.",
        "Fix scheduler and resource limits.",
        "Enable horizontal autoscaling.",
        "Fix runtime dependencies.",
        "Fix registry authentication.",
        "Open firewall rules correctly.",
        "Increase memory limits.",
        "Increase CPU allocation.",
        "Improve cooling and monitor hardware health.",
        "Fix VM boot configuration.",
        "Switch to fallback region or wait for recovery.",
        "Fix health check endpoints.",
        "Align API versions.",
        "Upgrade deprecated libraries.",
        "Optimize threading model.",
        "Fix race condition with locks.",
        "Remove stale routing entries.",
        "Throttle retries and add backoff strategy.",
        "Verify SSL certificate validity, inspect TLS configuration, update trusted CA certificates, and restart the affected application.",
        "Identify CPU-intensive pods using kubectl top, optimize resource usage, increase node capacity if required, and rebalance workloads.",
"Review backup logs, verify storage availability and database permissions, resolve the reported errors, and rerun the backup job.",
"Check Redis replication status, verify network connectivity between master and replicas, restart replication if necessary, and monitor synchronization.",
"Inspect authentication service logs, validate API keys or tokens, verify identity provider availability, and retry failed requests.",
"Review container logs using docker logs, identify startup failures or missing dependencies, correct the configuration, and redeploy the container.",
"Analyze disk performance using iostat, identify heavy read/write operations, clean unnecessary files, and upgrade storage if required.",
"Verify health check configuration, remove unhealthy instances from the backend pool, restart failed services, and monitor load balancer status.",
"Check message broker availability, validate connection strings and credentials, restart the broker if required, and verify queue health.",
"Install the complete certificate chain, verify intermediate certificates, update the web server configuration, and restart the affected service."
]

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
inject_voice_recorder()

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

