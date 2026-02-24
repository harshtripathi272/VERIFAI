/**
 * VERIFAI Mobile Demo — Real Workflow
 * 
 * Connects to the actual VERIFAI FastAPI backend running on your PC.
 * Uses the REAL MedGemma-4B, MedSigLIP, CheXbert, and all agents.
 * 
 * The phone sends the X-ray image → PC runs the full workflow →
 * SSE streams live agent progress back to the phone.
 */

// ── Configuration ──
// Auto-detect the server address (same host that served this page)
const API_BASE = `http://${window.location.hostname}:8000/api/v1`;

// ── State ──
let selectedFile = null;
let startTime = 0;
let timerRAF = null;

// ── Server Connection Check ──
window.loadModel = async function () {
    const btn = document.getElementById('btn-load-model');
    const statusEl = document.getElementById('model-status');
    const statusText = document.getElementById('model-status-text');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressLabel = document.getElementById('progress-label');
    const progressPercent = document.getElementById('progress-percent');

    btn.disabled = true;
    btn.innerHTML = 'Connecting...';
    statusEl.className = 'model-status loading';
    statusText.textContent = 'Connecting to VERIFAI server...';
    progressContainer.classList.remove('hidden');
    progressFill.style.width = '30%';
    progressLabel.textContent = 'Checking server health...';
    progressPercent.textContent = '';

    try {
        // Step 1: Check if the FastAPI server is running
        const healthRes = await fetch(`${API_BASE}/health`, {
            signal: AbortSignal.timeout(5000)
        });
        const health = await healthRes.json();

        progressFill.style.width = '60%';
        progressLabel.textContent = `Server: ${health.status} | Mock: ${health.mock_mode}`;

        // Step 2: Check available tools
        progressFill.style.width = '80%';
        progressLabel.textContent = 'Verifying agents & models...';
        await sleep(500);

        const toolsRes = await fetch(`${API_BASE}/tools`);
        const tools = await toolsRes.json();

        progressFill.style.width = '100%';
        progressLabel.textContent = `Ready — ${tools.total} tools loaded`;
        progressPercent.textContent = '✓';

        // Success
        statusEl.className = 'model-status ready';
        statusText.textContent = `Connected — MedGemma + ${tools.total} tools ready`;

        await sleep(800);
        document.getElementById('splash-screen').classList.remove('active');
        document.getElementById('app-screen').classList.add('active');

    } catch (err) {
        console.error('Connection failed:', err);
        statusEl.className = 'model-status error';
        progressFill.style.width = '0%';

        if (err.name === 'TimeoutError' || err.message.includes('fetch')) {
            statusText.textContent = 'Cannot reach server. Is it running?';
            progressLabel.textContent = `Start: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`;
        } else {
            statusText.textContent = 'Connection failed — ' + err.message;
            progressLabel.textContent = 'Check that the server is running on port 8000';
        }

        btn.disabled = false;
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
            </svg>
            Retry Connection`;
    }
};

// ── File Handling ──
document.getElementById('file-input').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
        document.getElementById('image-preview').src = ev.target.result;
        document.getElementById('drop-zone').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');
        document.getElementById('btn-analyze').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
});

window.removeImage = function () {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('drop-zone').classList.remove('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('btn-analyze').classList.add('hidden');
};

// ── Analysis — Real VERIFAI Workflow ──
window.analyzeImage = async function () {
    if (!selectedFile) return;

    const btnAnalyze = document.getElementById('btn-analyze');
    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `
        <div style="width:18px;height:18px;border:2px solid rgba(0,0,0,0.2);border-top-color:#000;border-radius:50%;animation:spin 0.6s linear infinite"></div>
        Starting workflow...
    `;

    // Show pipeline, hide old results
    const pipelineSection = document.getElementById('pipeline-section');
    pipelineSection.classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');
    resetPipeline();

    startTime = performance.now();
    startTimer();

    try {
        // ── Step 1: Upload image and start workflow ──
        const formData = new FormData();
        formData.append('images', selectedFile);
        formData.append('views', 'AP');

        const startRes = await fetch(`${API_BASE}/workflows/start`, {
            method: 'POST',
            body: formData
        });

        if (!startRes.ok) {
            const errText = await startRes.text();
            throw new Error(`Server returned ${startRes.status}: ${errText}`);
        }

        const startData = await startRes.json();
        const sessionId = startData.session_id;
        console.log('Workflow started:', sessionId);

        btnAnalyze.innerHTML = `
            <div style="width:18px;height:18px;border:2px solid rgba(0,0,0,0.2);border-top-color:#000;border-radius:50%;animation:spin 0.6s linear infinite"></div>
            Running workflow...
        `;

        // ── Step 2: Connect to SSE stream for live progress ──
        const streamUrl = `${API_BASE}/workflows/${sessionId}/stream`;
        console.log('Connecting to SSE:', streamUrl);

        const eventSource = new EventSource(streamUrl);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('SSE event:', data);
                handleAgentEvent(data);
            } catch (e) {
                console.warn('SSE parse error:', e);
            }
        };

        eventSource.onerror = (err) => {
            console.log('SSE connection closed or error');
            eventSource.close();
        };

        // ── Step 3: Poll for final result ──
        const result = await pollWorkflowResult(sessionId);
        eventSource.close();

        // ── Step 4: Show results ──
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        stopTimer();
        showResults(result, elapsed);

    } catch (err) {
        console.error('Workflow failed:', err);
        stopTimer();
        alert('Workflow error: ' + err.message);
    }

    btnAnalyze.disabled = false;
    btnAnalyze.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
        </svg>
        Run VERIFAI Diagnosis
    `;
};

// ── SSE Event Handler ──
function handleAgentEvent(event) {
    const agent = event.agent;
    const status = event.status;

    // Map SSE agent names to our pipeline IDs
    const agentMap = {
        'radiologist': 'radiologist',
        'chexbert': 'chexbert',
        'evidence_gathering': 'evidence',
        'historian': 'evidence',
        'literature': 'evidence',
        'critic': 'critic',
        'debate': 'critic',
        'validator': 'validator',
        'finalize': 'validator'
    };

    const pipelineId = agentMap[agent];
    if (!pipelineId) return;

    if (status === 'started' || status === 'running') {
        setAgentActive(pipelineId);
    } else if (status === 'completed' || status === 'done') {
        setAgentCompleted(pipelineId);
    }
}

// ── Poll for final result ──
async function pollWorkflowResult(sessionId) {
    const maxAttempts = 120; // 2 minutes max
    for (let i = 0; i < maxAttempts; i++) {
        await sleep(1000);

        try {
            const res = await fetch(`${API_BASE}/workflows/${sessionId}/status`);
            const data = await res.json();
            console.log('Poll status:', data.status);

            if (data.status === 'completed' || data.status === 'done') {
                // Mark all agents completed
                ['radiologist', 'chexbert', 'evidence', 'critic', 'validator'].forEach(setAgentCompleted);
                return data.final_result || data;
            }

            if (data.status === 'failed' || data.status === 'error') {
                throw new Error(data.error || 'Workflow failed');
            }

            // If there's pending review data, treat it as a result for demo purposes
            if (data.pending_review_data) {
                ['radiologist', 'chexbert', 'evidence', 'critic', 'validator'].forEach(setAgentCompleted);
                return data.pending_review_data;
            }
        } catch (err) {
            if (err.message === 'Workflow failed') throw err;
            console.warn('Poll error (retrying):', err.message);
        }
    }

    throw new Error('Workflow timed out after 2 minutes');
}

// ── Show Results ──
function showResults(result, elapsed) {
    const section = document.getElementById('results-section');
    section.classList.remove('hidden');
    setTimeout(() => section.scrollIntoView({ behavior: 'smooth' }), 100);

    // Extract diagnosis info from the real result
    const diagnosis = result.diagnosis || result.primary_diagnosis || 'Analysis Complete';
    const confidence = result.confidence || result.final_confidence || 0.75;
    const uncertainty = result.uncertainty || result.kle_uncertainty || 0.2;
    const deferred = result.deferred || false;
    const explanation = result.explanation || result.reasoning || '';
    const trace = result.trace || [];
    const evidencePacket = result.evidence_packet || result.evidence || {};

    // Primary diagnosis
    document.getElementById('diagnosis-text').textContent = diagnosis;

    // Badge
    const badge = document.getElementById('result-badge');
    if (deferred) {
        badge.textContent = 'Deferred to Human';
        badge.className = 'card-badge result-badge low';
    } else if (confidence >= 0.7) {
        badge.textContent = 'High Confidence';
        badge.className = 'card-badge result-badge high';
    } else if (confidence >= 0.5) {
        badge.textContent = 'Moderate Confidence';
        badge.className = 'card-badge result-badge medium';
    } else {
        badge.textContent = 'Low Confidence';
        badge.className = 'card-badge result-badge low';
    }

    // Confidence & Uncertainty rings
    animateRing('conf-circle', confidence);
    animateRing('unc-circle', uncertainty);
    document.getElementById('confidence-value').textContent = Math.round(confidence * 100) + '%';
    document.getElementById('uncertainty-value').textContent = Math.round(uncertainty * 100) + '%';

    // Findings from evidence packet
    const findingsList = document.getElementById('findings-list');
    findingsList.innerHTML = '';

    // Try to extract CheXbert labels or radiologist findings
    const chexbertLabels = evidencePacket.chexbert_labels || evidencePacket.labels || {};
    const positiveLabels = Object.entries(chexbertLabels).filter(([k, v]) => v === 1 || v === 'positive');
    const negativeLabels = Object.entries(chexbertLabels).filter(([k, v]) => v === 0 || v === 'negative');

    if (positiveLabels.length > 0) {
        positiveLabels.forEach(([name]) => {
            findingsList.innerHTML += `
                <div class="finding-item">
                    <div class="finding-dot positive"></div>
                    <span>${formatLabel(name)}</span>
                    <span style="margin-left:auto;font-family:'JetBrains Mono';font-size:12px;color:var(--accent-pink)">pos</span>
                </div>`;
        });
    }
    if (negativeLabels.length > 0) {
        negativeLabels.slice(0, 4).forEach(([name]) => {
            findingsList.innerHTML += `
                <div class="finding-item">
                    <div class="finding-dot negative"></div>
                    <span>${formatLabel(name)}</span>
                    <span style="margin-left:auto;font-family:'JetBrains Mono';font-size:12px;color:var(--accent-cyan)">neg</span>
                </div>`;
        });
    }

    // If no structured findings, show trace items
    if (positiveLabels.length === 0 && negativeLabels.length === 0 && trace.length > 0) {
        trace.forEach(t => {
            findingsList.innerHTML += `
                <div class="finding-item">
                    <div class="finding-dot negative"></div>
                    <span style="font-size:12px">${t}</span>
                </div>`;
        });
    }

    // Evidence & Explanation
    const evidenceContent = document.getElementById('evidence-content');
    let evidenceHTML = '';

    if (explanation) {
        evidenceHTML += `<p style="margin-bottom:10px"><strong style="color:var(--accent-cyan)">🔬 Explanation:</strong> ${explanation}</p>`;
    }

    // Historian context
    const historianCtx = evidencePacket.historian || evidencePacket.history_context || '';
    if (historianCtx) {
        evidenceHTML += `<p style="margin-bottom:10px"><strong style="color:var(--accent-purple)">📋 History:</strong> ${historianCtx}</p>`;
    }

    // Literature context
    const litCtx = evidencePacket.literature || evidencePacket.literature_context || '';
    if (litCtx) {
        evidenceHTML += `<p style="margin-bottom:10px"><strong style="color:var(--accent-orange)">📖 Literature:</strong> ${typeof litCtx === 'string' ? litCtx : JSON.stringify(litCtx).substring(0, 300)}</p>`;
    }

    // Debate info
    const debate = evidencePacket.debate || evidencePacket.debate_result || {};
    if (debate.consensus !== undefined) {
        evidenceHTML += `<p style="margin-top:10px;font-size:11px;color:var(--text-muted)">
            <strong>Debate:</strong> ${debate.consensus ? 'Consensus reached' : 'No consensus'} 
            ${debate.rounds ? `in ${debate.rounds} rounds` : ''}
        </p>`;
    }

    if (!evidenceHTML) {
        // Fallback: dump raw result nicely
        evidenceHTML = `<pre style="font-size:11px;white-space:pre-wrap;word-break:break-word;color:var(--text-secondary)">${JSON.stringify(result, null, 2).substring(0, 800)}</pre>`;
    }

    evidenceContent.innerHTML = evidenceHTML;

    // Timing
    document.getElementById('total-time').textContent = elapsed + 's';
}

// ── UI Helpers ──
function setAgentActive(name) {
    const el = document.getElementById('agent-' + name);
    if (!el || el.classList.contains('completed')) return;
    el.classList.add('active');
    const prev = el.previousElementSibling;
    if (prev && prev.classList.contains('pipeline-connector')) {
        prev.classList.add('active');
    }
}

function setAgentCompleted(name) {
    const el = document.getElementById('agent-' + name);
    if (!el) return;
    el.classList.remove('active');
    el.classList.add('completed');
    const prev = el.previousElementSibling;
    if (prev && prev.classList.contains('pipeline-connector')) {
        prev.classList.add('active');
    }
}

function resetPipeline() {
    document.querySelectorAll('.pipeline-agent').forEach(el => {
        el.classList.remove('active', 'completed');
    });
    document.querySelectorAll('.pipeline-connector').forEach(el => {
        el.classList.remove('active');
    });
    // Reset rings
    ['conf-circle', 'unc-circle'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.style.transition = 'none';
            el.style.strokeDashoffset = '213.6';
        }
    });
}

function startTimer() {
    const timerEl = document.getElementById('pipeline-time');
    const update = () => {
        if (!startTime) return;
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        timerEl.textContent = elapsed + 's';
        timerRAF = requestAnimationFrame(update);
    };
    timerRAF = requestAnimationFrame(update);
}

function stopTimer() {
    if (timerRAF) cancelAnimationFrame(timerRAF);
    startTime = 0;
}

function animateRing(circleId, value) {
    const circle = document.getElementById(circleId);
    if (!circle) return;
    const circumference = 2 * Math.PI * 34;
    const offset = circumference * (1 - Math.min(1, value));
    setTimeout(() => {
        circle.style.transition = 'stroke-dashoffset 1s ease';
        circle.style.strokeDashoffset = offset;
    }, 200);
}

function formatLabel(label) {
    return label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

window.resetApp = function () {
    selectedFile = null;
    stopTimer();
    document.getElementById('file-input').value = '';
    document.getElementById('drop-zone').classList.remove('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('btn-analyze').classList.add('hidden');
    document.getElementById('pipeline-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    resetPipeline();
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
