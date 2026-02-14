"""
VERIFAI Streamlit UI

Interactive frontend for the diagnostic system.
"""

import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="VERIFAI - Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
    }
    .deferred-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🫁 VERIFAI</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>Evidence-First, Uncertainty-Gated Diagnostic AI for Chest X-Rays</p>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    patient_id = st.text_input(
        "Patient ID (optional)",
        placeholder="e.g., PAT-12345",
        help="Enable FHIR context retrieval"
    )
    
    st.divider()
    
    st.header("📖 Pipeline")
    st.markdown("""
    **Agents:**
    - 🔍 Radiologist (MedSigLIP + MedGemma-4B)
    - 🎯 Critic (Overconfidence detector)
    - 📋 Historian (FHIR context)
    - 📚 Literature (PubMed RAG)
    - 👨‍⚕️ Chief (MedGemma-27B)
    
    **Routing Thresholds:**
    - U < 30%: Direct diagnosis
    - U ≥ 30%: + Patient context
    - U ≥ 40%: + Literature evidence  
    - U ≥ 50%: Escalate to Chief
    """)

# Main content
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📤 Upload X-Ray")
    
    uploaded = st.file_uploader(
        "Choose chest X-ray image",
        type=["png", "jpg", "jpeg", "dcm"],
        help="PNG, JPEG, or DICOM format"
    )
    
    if uploaded:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception:
            st.info("Preview not available")
        uploaded.seek(0)

with col2:
    st.header("🔬 Analysis")
    
    if uploaded:
        if st.button("▶️ Run VERIFAI Pipeline", type="primary", use_container_width=True):
            with st.spinner("Agents deliberating..."):
                try:
                    files = {"image": (uploaded.name, uploaded.getvalue(), "image/png")}
                    params = {"patient_id": patient_id} if patient_id else {}
                    
                    response = requests.post(
                        f"{API_URL}/diagnose",
                        files=files,
                        params=params,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Result card
                        if data["deferred"]:
                            st.markdown(f"""
                            <div class="deferred-card">
                                <h2>⚠️ Deferred to Human Review</h2>
                                <p>{data["deferral_reason"]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="diagnosis-card">
                                <h2>📋 {data["diagnosis"]}</h2>
                                <p style="font-size: 1.5rem;">Confidence: {data["confidence"]:.0%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Uncertainty
                        unc = data["uncertainty"]
                        unc_color = "green" if unc < 0.3 else "orange" if unc < 0.5 else "red"
                        st.markdown(f"**Final Uncertainty:** :{unc_color}[{unc:.1%}]")
                        
                        # Tabs for evidence
                        tabs = st.tabs(["📊 Visual", "📋 Clinical", "📚 Literature", "🧠 Debate", "🎯 Critic", "📝 Trace"])
                        
                        evidence = data["evidence_packet"]
                        
                        with tabs[0]:
                            if evidence.get("visual"):
                                v = evidence["visual"]
                                st.subheader("Findings")
                                st.write(v.get("findings", "No findings available."))
                                st.subheader("Impression")
                                st.write(v.get("impression", "No impression available."))
                            else:
                                st.caption("No visual data")
                        
                        with tabs[1]:
                            if evidence.get("clinical"):
                                c = evidence["clinical"]
                                if c.get("supporting_facts"):
                                    st.subheader("Supporting Facts")
                                    for f in c["supporting_facts"]:
                                        st.success(f"{f['description']} ({f['fhir_resource_type']})")
                                st.info(c.get("summary", ""))
                            else:
                                st.caption("No clinical context (uncertainty was low)")
                        
                        with tabs[2]:
                            if evidence.get("literature"):
                                l = evidence["literature"]
                                st.caption(f"Evidence Strength: **{l['overall_strength'].upper()}**")
                                for cit in l.get("citations", []):
                                    with st.expander(f"[{cit['pmid']}] {cit['title'][:60]}..."):
                                        st.markdown(f"**Authors:** {', '.join(cit.get('authors', []))}")
                                        st.markdown(f"**Journal:** {cit.get('journal', 'N/A')} ({cit.get('year', 'N/A')})")
                                        st.markdown(f"**Relevance:** {cit.get('relevance_summary', '')}")
                            else:
                                st.caption("No literature retrieved")

                        with tabs[3]:
                            if evidence.get("debate"):
                                d = evidence["debate"]
                                st.info(d.get("debate_summary", "Debate completed."))
                                
                                for r in d.get("rounds", []):
                                    with st.expander(f"Round {r['round_number']} ({'Consensus Reached' if r.get('round_consensus') else 'No Consensus'})"):
                                        st.caption(f"Confidence Delta: {r.get('confidence_delta', 0):+.2%}")
                                        
                                        # Critic
                                        if r.get("critic_challenge"):
                                            c = r["critic_challenge"]
                                            st.warning(f"**Critic ({c.get('position', 'challenge')}):** {c.get('argument', '')}")
                                        
                                        # Historian
                                        if r.get("historian_response"):
                                            h = r["historian_response"]
                                            st.info(f"**Historian ({h.get('position', 'response')}):** {h.get('argument', '')}")
                                            
                                        # Literature
                                        if r.get("literature_response"):
                                            l = r["literature_response"]
                                            st.success(f"**Literature ({l.get('position', 'response')}):** {l.get('argument', '')}")
                            else:
                                st.caption("No debate history")

                        with tabs[4]:
                            if evidence.get("critic"):
                                cr = evidence["critic"]
                                st.metric("Overconfidence", f"{cr['overconfidence_probability']:.0%}")
                                if cr.get("concern_signals"):
                                    st.warning("Concerns: " + "; ".join(cr["concern_signals"]))
                            else:
                                st.caption("No critic assessment")
                        
                        with tabs[5]:
                            for line in data["trace"]:
                                st.code(line, language="text")
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Start backend with: `uvicorn app.main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("👈 Upload a chest X-ray to begin")

# Footer
st.markdown("---")
st.caption("⚠️ VERIFAI is a research prototype. All diagnoses require verification by licensed healthcare professionals.")
