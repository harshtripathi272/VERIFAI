"""
VERIFAI Streamlit Frontend

Provides interactive UI for uploading X-rays and viewing results.
"""

import json
import requests
import streamlit as st
from PIL import Image
import io


# --- Configuration ---
API_URL = "http://localhost:8000"


# --- Page Setup ---
st.set_page_config(
    page_title="VERIFAI - Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .deferred-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .uncertainty-low {
        color: #2ecc71;
    }
    .uncertainty-medium {
        color: #f39c12;
    }
    .uncertainty-high {
        color: #e74c3c;
    }
    .trace-item {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
        background: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown('<p class="main-header">🫁 VERIFAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Clinically Grounded, Uncertainty-Aware Diagnostic AI for Chest X-Ray Interpretation</p>',
    unsafe_allow_html=True,
)


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    patient_id = st.text_input(
        "Patient ID (optional)",
        placeholder="e.g., PAT-12345",
        help="Enter patient ID to retrieve FHIR context (simulated)",
    )

    st.divider()

    st.header("📖 About")
    st.markdown("""
    **VERIFAI** uses a multi-agent architecture:

    - 🔍 **Radiologist Agent**: Visual interpretation
    - 🎯 **Critic Agent**: Overconfidence detection
    - 📋 **Historian Agent**: Patient context (FHIR)
    - 📚 **Literature Agent**: Evidence retrieval
    - 👨‍⚕️ **Chief Orchestrator**: Final arbitration

    Routing is **uncertainty-gated**:
    - U < 0.30: Direct diagnosis
    - U ≥ 0.30: Add patient context
    - U ≥ 0.40: Add literature evidence
    - U ≥ 0.50: Escalate to Chief
    """)

    st.divider()

    st.caption("Built with HAI-DEF Medical Foundation Models")


# --- Main Content ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📤 Upload X-Ray")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["png", "jpg", "jpeg", "dcm"],
        help="Upload a chest X-ray in PNG, JPEG, or DICOM format",
    )

    if uploaded_file:
        # Display uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        except Exception:
            st.info("Preview not available for this file type")

        # Reset file position for upload
        uploaded_file.seek(0)


with col2:
    st.header("🔬 Analysis Results")

    if uploaded_file:
        with st.spinner("Running VERIFAI diagnostic pipeline..."):
            try:
                # Make API request
                files = {"image": (uploaded_file.name, uploaded_file, "image/png")}
                params = {"patient_id": patient_id} if patient_id else {}

                response = requests.post(
                    f"{API_URL}/diagnose",
                    files=files,
                    params=params,
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display main diagnosis
                    if result["deferred"]:
                        st.markdown(
                            f"""
                            <div class="deferred-box">
                                <h2>⚠️ Deferred to Human Review</h2>
                                <p style="font-size: 1.1rem;">{result["deferral_reason"]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="diagnosis-box">
                                <h2>📋 {result["diagnosis"]}</h2>
                                <p style="font-size: 1.5rem;">Confidence: {result["confidence"]:.0%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

                    # Uncertainty display
                    uncertainty = result["uncertainty"]
                    if uncertainty < 0.30:
                        unc_class = "uncertainty-low"
                        unc_label = "LOW"
                    elif uncertainty < 0.50:
                        unc_class = "uncertainty-medium"
                        unc_label = "MODERATE"
                    else:
                        unc_class = "uncertainty-high"
                        unc_label = "HIGH"

                    st.markdown(
                        f"**Final Uncertainty:** <span class='{unc_class}'>{uncertainty:.1%} ({unc_label})</span>",
                        unsafe_allow_html=True,
                    )

                    # Evidence Packet Tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "🔍 Visual Findings",
                        "📋 Clinical Context",
                        "📚 Literature",
                        "📝 Audit Trail",
                    ])

                    evidence = result["evidence_packet"]

                    with tab1:
                        if evidence.get("visual_evidence"):
                            ve = evidence["visual_evidence"]

                            st.subheader("Findings")
                            for finding in ve.get("findings", []):
                                st.markdown(
                                    f"- **{finding['location']}**: {finding['observation']} "
                                    f"(severity: {finding['severity']:.0%})"
                                )

                            st.subheader("Differential Diagnosis")
                            for dx in ve.get("differential", []):
                                st.progress(dx["probability"], text=f"{dx['diagnosis']}: {dx['probability']:.0%}")

                            if ve.get("reasoning"):
                                st.subheader("Reasoning")
                                st.info(ve["reasoning"])
                        else:
                            st.info("No visual findings available")

                    with tab2:
                        if evidence.get("clinical_context") and evidence["clinical_context"].get("conditions"):
                            cc = evidence["clinical_context"]

                            st.subheader("Active Conditions")
                            for cond in cc.get("conditions", []):
                                st.markdown(f"- {cond}")

                            st.subheader("Risk Factors")
                            for rf in cc.get("risk_factors", []):
                                st.warning(rf)

                            if cc.get("labs"):
                                st.subheader("Relevant Labs")
                                st.json(cc["labs"])

                            if cc.get("summary"):
                                st.subheader("Clinical Summary")
                                st.info(cc["summary"])
                        else:
                            st.info("Patient context not retrieved (uncertainty was low enough)")

                    with tab3:
                        if evidence.get("literature_support") and evidence["literature_support"].get("supporting"):
                            ls = evidence["literature_support"]

                            st.markdown(f"**Evidence Strength:** {ls.get('evidence_strength', 'N/A').upper()}")

                            st.subheader("Supporting Evidence")
                            for cit in ls.get("supporting", []):
                                with st.expander(f"[PMID: {cit['pmid']}] {cit['title'][:60]}..."):
                                    st.markdown(f"**Relevance:** {cit['relevance']:.0%}")
                                    st.markdown(f"**Excerpt:** {cit['excerpt']}")

                            if ls.get("contradicting"):
                                st.subheader("Contradicting Evidence")
                                for cit in ls.get("contradicting", []):
                                    with st.expander(f"[PMID: {cit['pmid']}] {cit['title'][:60]}..."):
                                        st.markdown(f"**Relevance:** {cit['relevance']:.0%}")
                                        st.markdown(f"**Excerpt:** {cit['excerpt']}")
                        else:
                            st.info("Literature not retrieved (uncertainty was low enough)")

                    with tab4:
                        st.subheader("Agent Reasoning Trace")
                        for entry in result["trace"]:
                            st.markdown(f'<div class="trace-item">{entry}</div>', unsafe_allow_html=True)

                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to VERIFAI API. Make sure the backend is running on port 8000.")
                st.code("uvicorn app.main:app --reload", language="bash")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.info("👈 Upload a chest X-ray image to begin analysis")


# --- Footer ---
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: VERIFAI is a research prototype. "
    "All diagnoses should be verified by a licensed healthcare professional."
)
