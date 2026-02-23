const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface WorkflowStartResponse {
    session_id: string;
    status: string;
    message: string;
}

export interface WorkflowStatusResponse {
    session_id: string;
    status: "running" | "suspended" | "completed" | "failed" | "not_found";
    current_state?: any;
    pending_review_data?: any;
    final_result?: any;
}

export async function uploadAndStartWorkflow(
    files: File[],
    views: string[],
    patientId: string = "",
    fhirFile?: File | null
): Promise<WorkflowStartResponse> {
    const formData = new FormData();
    files.forEach(file => {
        formData.append("images", file);
    });
    views.forEach(view => {
        formData.append("views", view);
    });

    if (patientId) {
        formData.append("patient_id", patientId);
    }
    if (fhirFile) {
        formData.append("fhir_report", fhirFile);
    }

    const response = await fetch(`${API_BASE_URL}/workflows/start`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorDetails = await response.text();
        throw new Error(`Failed to start workflow: ${errorDetails}`);
    }

    return response.json();
}

export async function checkWorkflowStatus(sessionId: string): Promise<WorkflowStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/workflows/${sessionId}/status`, {
        method: "GET",
    });

    if (!response.ok) {
        const errorDetails = await response.text();
        throw new Error(`Failed to get status: ${errorDetails}`);
    }

    return response.json();
}

export async function submitHumanFeedback(sessionId: string, action: "approve" | "reject", feedback: string = "", correctDiagnosis: string = "") {
    const response = await fetch(`${API_BASE_URL}/workflows/${sessionId}/resume`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            action,
            feedback,
            correct_diagnosis: correctDiagnosis,
        }),
    });

    if (!response.ok) {
        const errorDetails = await response.text();
        throw new Error(`Failed to submit feedback: ${errorDetails}`);
    }

    return response.json();
}


// === NEW: Safety Guardrails API ===

export interface SafetyFlag {
    flag_type: string;
    severity: "high" | "medium" | "low";
    description: string;
    recommendation: string;
}

export interface CriticalFinding {
    condition: string;
    urgency: "STAT" | "URGENT" | "FOLLOW-UP";
    action: string;
    acr_category: string;
    icd10: string;
}

export interface SafetyReportResponse {
    passed: boolean;
    safety_score: number;
    critical_findings: CriticalFinding[];
    significant_findings: CriticalFinding[];
    requires_immediate_action: boolean;
    red_flags: SafetyFlag[];
    recommendations: string[];
    hallucination_risk: "low" | "medium" | "high";
    confidence_uncertainty_aligned: boolean;
    summary: string;
}

export async function getSafetyReport(sessionId: string): Promise<SafetyReportResponse> {
    const response = await fetch(`${API_BASE_URL}/workflows/${sessionId}/safety`, {
        method: "GET",
    });

    if (!response.ok) {
        const errorDetails = await response.text();
        throw new Error(`Failed to get safety report: ${errorDetails}`);
    }

    return response.json();
}


// === NEW: Evidence Report ===

export function getEvidenceReportUrl(sessionId: string): string {
    return `${API_BASE_URL}/workflows/${sessionId}/report`;
}


// === NEW: Observability Metrics ===

export interface MetricsSummary {
    system: {
        active_workflows: number;
        total_workflows: number;
        deferrals: number;
        critical_findings: number;
        workflow_duration: { count: number; mean: number; p50: number; p95: number; p99: number };
    };
    agents: {
        duration: Record<string, { count: number; mean: number; p50: number; p95: number }>;
        invocations: Record<string, number>;
    };
    diagnostics: {
        confidence: { count: number; mean: number; p50: number; p95: number };
        uncertainty: { count: number; mean: number; p50: number; p95: number };
        debate_rounds: { count: number; mean: number };
        safety_score: { count: number; mean: number; p50: number };
    };
    safety: {
        flags: Record<string, number>;
        errors: Record<string, number>;
    };
}

export async function getMetricsSummary(): Promise<MetricsSummary> {
    const response = await fetch(`${API_BASE_URL}/metrics/summary`, {
        method: "GET",
    });

    if (!response.ok) {
        const errorDetails = await response.text();
        throw new Error(`Failed to get metrics: ${errorDetails}`);
    }

    return response.json();
}
