const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface WorkflowStartResponse {
    session_id: string;
    status: string;
    message: string;
}

export interface WorkflowStatusResponse {
    session_id: string;
    status: "running" | "suspended" | "completed" | "failed" | "not_found";
    pending_review_data?: any;
    final_result?: any;
}

export async function uploadAndStartWorkflow(file: File, patientId: string = ""): Promise<WorkflowStartResponse> {
    const formData = new FormData();
    formData.append("image", file);
    if (patientId) {
        formData.append("patient_id", patientId);
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
