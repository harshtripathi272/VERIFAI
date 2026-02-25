"use client";
import { useState, useEffect, useRef } from "react";
import { CheckCircle2, AlertTriangle, FileText, ImageIcon, BookOpen, Activity, User, ShieldAlert, ChevronLeft, Download, Loader2, Shield, BarChart3, ExternalLink, AlertCircle, XCircle, Radio } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { GradientText } from "@/components/GradientText";
import { checkWorkflowStatus, submitHumanFeedback, getSafetyReport, getEvidenceReportUrl, WorkflowStatusResponse, SafetyReportResponse } from "@/lib/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface AgentEvent {
  agent: string;
  status: string;
  message: string;
  timestamp: string;
  data: Record<string, any>;
}

const AGENT_ICONS: Record<string, string> = {
  system: "🔗",
  radiologist: "🩻",
  chexbert: "🏷️",
  evidence: "📚",
  critic: "🔍",
  debate: "⚖️",
  validator: "✅",
  finalize: "📋",
};

// === Uncertainty Cascade Graph Component ===
// Pure SVG — no external chart library.
// Data points: { agent: string, system_uncertainty: number }[]
function UncertaintyGraph({ data }: { data: { agent: string; system_uncertainty: number }[] }) {
  if (!data || data.length === 0) return null;
  const W = 520;
  const H = 120;
  const PAD_L = 38;
  const PAD_R = 16;
  const PAD_T = 12;
  const PAD_B = 36;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;
  const n = data.length;
  // Each data point maps to an x-position
  const xOf = (i: number) => PAD_L + (n === 1 ? chartW / 2 : (i / (n - 1)) * chartW);
  // Y-axis: 0 at bottom, 1.0 at top
  const yOf = (u: number) => PAD_T + chartH * (1 - Math.max(0, Math.min(1, u)));
  // Build polyline points string
  const points = data.map((d, i) => `${xOf(i)},${yOf(d.system_uncertainty)}`).join(" ");
  // Y-axis gridlines at 0.25, 0.5, 0.75, 1.0
  const gridYs = [0.25, 0.5, 0.75, 1.0];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 130 }}>
      {/* Grid lines */}
      {gridYs.map((g) => (
        <g key={g}>
          <line
            x1={PAD_L} y1={yOf(g)} x2={W - PAD_R} y2={yOf(g)}
            stroke="rgba(255,255,255,0.06)" strokeWidth={1}
          />
          <text x={PAD_L - 4} y={yOf(g) + 4} textAnchor="end" fontSize={9} fill="rgba(255,255,255,0.25)">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {/* Uncertainty line */}
      <polyline
        points={points}
        fill="none"
        stroke="#00E5FF"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.85}
      />
      {/* Dots + agent labels */}
      {data.map((d, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(d.system_uncertainty)} r={3} fill="#00E5FF" opacity={0.9} />
          <text
            x={xOf(i)} y={H - 4}
            textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}
            fontSize={9} fill="rgba(255,255,255,0.35)"
          >
            {d.agent}
          </text>
          <text
            x={xOf(i)} y={yOf(d.system_uncertainty) - 7}
            textAnchor="middle" fontSize={9} fill="rgba(0,229,255,0.7)"
          >
            {d.system_uncertainty.toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function ResultsPage({ params }: { params: { id: string } }) {
  const [activeTab, setActiveTab] = useState("visual");
  const [workflowInfo, setWorkflowInfo] = useState<WorkflowStatusResponse | null>(null);
  const [safetyReport, setSafetyReport] = useState<SafetyReportResponse | null>(null);
  const [safetyLoading, setSafetyLoading] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [feedbackText, setFeedbackText] = useState("");
  const [correctDx, setCorrectDx] = useState("");
  const [liveEvents, setLiveEvents] = useState<AgentEvent[]>([]);
  const [sseConnected, setSseConnected] = useState(false);
  const [sseTrigger, setSseTrigger] = useState(0);
  const [rerunInProgress, setRerunInProgress] = useState(false);
  const rerunInProgressRef = useRef(false);
  const feedRef = useRef<HTMLDivElement>(null);

  // SSE Live Feed + Polling Fallback
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    let es: EventSource | null = null;

    // Start SSE connection
    try {
      es = new EventSource(`${API_BASE_URL}/workflows/${params.id}/stream`);

      es.onopen = () => setSseConnected(true);

      es.onmessage = (e) => {
        try {
          const event: AgentEvent = JSON.parse(e.data);
          setLiveEvents((prev) => [...prev, event]);

          // Auto-scroll feed
          setTimeout(() => {
            feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
          }, 100);

          // When workflow completes, fetch final status
          if (event.status === "workflow_complete" || event.status === "workflow_error") {
            es?.close();
            setSseConnected(false);
            // Fetch final result
            setTimeout(async () => {
              const data = await checkWorkflowStatus(params.id);
              setWorkflowInfo(data);
            }, 500);
          }
        } catch { }
      };

      es.onerror = () => {
        setSseConnected(false);
        es?.close();
      };
    } catch {
      // SSE not available, fall back to polling only
    }

    // Polling fallback (runs alongside SSE but less frequently)
    const poll = async () => {
      try {
        const data = await checkWorkflowStatus(params.id);
        setWorkflowInfo(data);
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(intervalId);
          rerunInProgressRef.current = false;
          setRerunInProgress(false);
        }
        // If rerun was in progress and workflow is now suspended again, the rerun finished
        if (data.status === "suspended" && rerunInProgressRef.current) {
          rerunInProgressRef.current = false;
          setRerunInProgress(false);
        }
      } catch { }
    };

    poll();
    intervalId = setInterval(poll, 5000);

    return () => {
      clearInterval(intervalId);
      es?.close();
    };
  }, [params.id, sseTrigger]);

  // Hydrate Live Events from the backend trace if available (for handling tab switch/navigation)
  useEffect(() => {
    if (workflowInfo?.current_state?.trace && liveEvents.length === 0) {
      const traceEvents = workflowInfo.current_state.trace.map((msg: string) => ({
        agent: msg.includes(":") ? msg.split(":")[0].toLowerCase().trim() : "system",
        status: "info",
        message: msg,
        timestamp: new Date().toISOString()
      }));
      setLiveEvents(traceEvents);
    }
  }, [workflowInfo?.current_state?.trace]);

  // Fetch safety report when workflow completes or is suspended
  useEffect(() => {
    if (workflowInfo && (workflowInfo.status === "completed" || workflowInfo.status === "suspended")) {
      fetchSafetyReport();
    }
  }, [workflowInfo?.status]);

  const fetchSafetyReport = async () => {
    setSafetyLoading(true);
    try {
      const report = await getSafetyReport(params.id);
      setSafetyReport(report);
    } catch (err) {
      console.error("Safety report error:", err);
    } finally {
      setSafetyLoading(false);
    }
  };

  const handleFeedback = async (action: "approve" | "reject") => {
    if (action === "reject" && !feedbackText.trim()) {
      alert("Please provide your feedback or notes before rerunning the workflow.");
      return;
    }

    setIsSubmittingFeedback(true);
    try {
      await submitHumanFeedback(params.id, action, feedbackText, correctDx);

      if (action === "reject") {
        // Hide the HITL box and show "rerunning" state
        rerunInProgressRef.current = true;
        setRerunInProgress(true);
        // Clear previous live events so the rerun shows fresh progress
        setLiveEvents([]);
        setFeedbackText("");
        setCorrectDx("");
      }

      if (action === "approve") {
        // Immediately reflect approved state in the UI
        setWorkflowInfo(prev => prev ? { ...prev, status: "completed" } : prev);
      }

      setTimeout(async () => {
        const data = await checkWorkflowStatus(params.id);
        setWorkflowInfo(data);
        setIsSubmittingFeedback(false);
        if (action === "reject") {
          // Re-trigger SSE to pick up the rerun's live events
          setSseTrigger(prev => prev + 1);
        }
      }, 1500);
    } catch (err) {
      console.error(err);
      alert("Failed to submit feedback. Please try again.");
      setIsSubmittingFeedback(false);
      rerunInProgressRef.current = false;
      setRerunInProgress(false);
    }
  };

  const tabs = [
    { id: "visual", label: "Visual Proof", icon: ImageIcon },
    { id: "clinical", label: "Clinical", icon: User },
    { id: "literary", label: "Literature", icon: BookOpen },
    { id: "safety", label: "Safety", icon: Shield },
    { id: "audit", label: "Audit Trail", icon: ShieldAlert },
  ];

  const renderLoadingState = () => (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6">
      <Loader2 className="w-12 h-12 text-[#00E5FF] animate-spin" />
      <h2 className="text-2xl font-semibold text-white/80">Analyzing Study...</h2>
      <p className="text-white/40">The Multi-Agent Pipeline is evaluating the case.</p>

      {/* Live Agent Feed */}
      {liveEvents.length > 0 && (
        <div className="w-full max-w-lg mt-6">
          <div className="flex items-center gap-2 mb-3">
            <Radio className={cn("w-4 h-4", sseConnected ? "text-green-400 animate-pulse" : "text-white/30")} />
            <span className="text-xs font-mono text-white/50 uppercase tracking-wider">
              Live Agent Feed {sseConnected ? "• Connected" : ""}
            </span>
          </div>
          <div
            ref={feedRef}
            className="bg-black/40 border border-white/10 rounded-xl p-4 max-h-72 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-white/10"
          >
            {liveEvents.map((event, i) => (
              <div
                key={i}
                className={cn(
                  "flex items-start gap-3 py-2 px-3 rounded-lg text-sm transition-all duration-300",
                  event.status === "started" && "bg-[#00E5FF]/5 border-l-2 border-[#00E5FF]/50",
                  event.status === "completed" && "bg-green-500/5 border-l-2 border-green-500/50",
                  event.status === "workflow_complete" && "bg-purple-500/10 border-l-2 border-purple-400/50",
                  event.status === "connected" && "bg-white/5 border-l-2 border-white/20",
                  event.status === "error" && "bg-red-500/10 border-l-2 border-red-500/50",
                )}
              >
                <span className="text-lg mt-0.5 shrink-0">{AGENT_ICONS[event.agent] || "🔹"}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white/90 capitalize">{event.agent}</span>
                    <span className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded-full font-mono uppercase",
                      event.status === "started" && "bg-[#00E5FF]/20 text-[#00E5FF]",
                      event.status === "completed" && "bg-green-500/20 text-green-400",
                      event.status === "workflow_complete" && "bg-purple-500/20 text-purple-300",
                      event.status === "connected" && "bg-white/10 text-white/50",
                      event.status === "error" && "bg-red-500/20 text-red-400",
                    )}>
                      {event.status === "workflow_complete" ? "done" : event.status}
                    </span>
                  </div>
                  <p className="text-white/50 text-xs mt-0.5 truncate">{event.message}</p>
                </div>
                <span className="text-[10px] text-white/20 font-mono shrink-0 mt-1">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2 text-[10px] text-[#00E5FF]/70 bg-[#00E5FF]/10 px-3 py-1 rounded-full font-mono uppercase tracking-widest mt-8">
        <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-ping inline-block mt-[2px]"></span>
        Session {params.id}
      </div>
    </div>
  );

  if (!workflowInfo) {
    return <div className="max-w-7xl mx-auto px-6 py-8">{renderLoadingState()}</div>;
  }

  const isRunning = workflowInfo.status === "running";

  // Uncertainty history from state (rolling last-2, shown as full list from current_state)
  const uncertaintyHistory: { agent: string; system_uncertainty: number }[] =
    workflowInfo?.current_state?.uncertainty_history ||
    workflowInfo?.final_result?.uncertainty_history ||
    [];

  // --- Real Logic variables for completed or suspended --- //
  let finalDx = "Pending Diagnosis";
  let confidence = 0;
  let uncertainty = 0;
  let evidence: any = null;

  if (workflowInfo.status === "completed" && workflowInfo.final_result) {
    finalDx = workflowInfo.final_result.diagnosis || "Undetermined";
    confidence = Math.round(workflowInfo.final_result.confidence * 100);
    evidence = workflowInfo.final_result.evidence_packet;
  } else if (workflowInfo.status === "suspended" && workflowInfo.pending_review_data) {
    finalDx = workflowInfo.pending_review_data.diagnosis;
    confidence = Math.round(workflowInfo.pending_review_data.confidence * 100);
    evidence = workflowInfo.pending_review_data.evidence;
  } else if (isRunning && workflowInfo.current_state) {
    const cs = workflowInfo.current_state;
    if (cs.radiologist?.impression) {
      finalDx = cs.radiologist.impression.split(".")[0];
    } else {
      finalDx = "Analyzing Study...";
    }
    evidence = {
      visual: cs.radiologist ? { findings: cs.radiologist.findings, impression: cs.radiologist.impression } : null,
      clinical: cs.historian ? {
        supporting_facts: cs.historian.supporting_facts,
        contradicting_facts: cs.historian.contradicting_facts,
        summary: cs.historian.clinical_summary
      } : null,
      literature: cs.literature ? {
        citations: cs.literature.citations || []
      } : null
    };
  }

  // Safety helpers
  const safetyColor = (score: number) => {
    if (score >= 0.8) return "#22c55e";
    if (score >= 0.6) return "#eab308";
    if (score >= 0.4) return "#f97316";
    return "#ef4444";
  };

  const severityIcon = (severity: string) => {
    if (severity === "high") return <XCircle className="h-4 w-4 text-red-400 shrink-0" />;
    if (severity === "medium") return <AlertTriangle className="h-4 w-4 text-yellow-400 shrink-0" />;
    return <AlertCircle className="h-4 w-4 text-green-400 shrink-0" />;
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 mb-20 relative z-10">
      {/* Top Bar */}
      <div className="flex justify-between items-center mb-8 animate-fadeInUp">
        <Link href="/diagnose" className="text-white/30 hover:text-white/60 text-sm flex items-center gap-1 transition-colors group">
          <ChevronLeft className="h-4 w-4 group-hover:-translate-x-0.5 transition-transform" /> Back
        </Link>
        <div className="flex items-center gap-3">
          <a
            href={getEvidenceReportUrl(params.id)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 text-[13px] text-[#E040FB] bg-[#E040FB]/[0.06] border border-[#E040FB]/15 rounded-lg hover:bg-[#E040FB]/10 transition-all"
          >
            <ExternalLink className="h-3.5 w-3.5" /> Evidence Report
          </a>
          <button className="flex items-center gap-2 px-4 py-2 text-[13px] text-[#00E5FF] bg-[#00E5FF]/[0.06] border border-[#00E5FF]/15 rounded-lg hover:bg-[#00E5FF]/10 transition-all glow-cyan">
            <Download className="h-3.5 w-3.5" /> Export PDF
          </button>
        </div>
      </div>

      {/* Workflow Rerunning Banner + Live Agent Feed - shown while feedback rerun is in progress */}
      {rerunInProgress && (
        <div className="mb-8 p-5 rounded-2xl border-2 border-amber-400/40 bg-amber-400/[0.02] shadow-[0_0_30px_rgba(251,191,36,0.05)] animate-fadeInUp">
          <div className="flex items-center gap-3 mb-2">
            <Loader2 className="h-5 w-5 animate-spin text-amber-400" />
            <h2 className="text-xl font-bold text-white">Workflow Rerunning with Your Feedback</h2>
          </div>
          <p className="text-white/60 text-sm mb-4">The AI pipeline is re-analyzing with your clinical guidance. View the live agent progress below.</p>

          {/* Live Agent Feed (same as first-run) */}
          {liveEvents.length > 0 && (
            <div className="mt-2">
              <div className="flex items-center gap-2 mb-3">
                <Radio className={cn("w-4 h-4", sseConnected ? "text-green-400 animate-pulse" : "text-white/30")} />
                <span className="text-xs font-mono text-white/50 uppercase tracking-wider">
                  Live Agent Feed {sseConnected ? "• Connected" : ""}
                </span>
              </div>
              <div
                ref={feedRef}
                className="bg-black/40 border border-white/10 rounded-xl p-4 max-h-72 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-white/10"
              >
                {liveEvents.map((event, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex items-start gap-3 py-2 px-3 rounded-lg text-sm transition-all duration-300",
                      event.status === "started" && "bg-[#00E5FF]/5 border-l-2 border-[#00E5FF]/50",
                      event.status === "completed" && "bg-green-500/5 border-l-2 border-green-500/50",
                      event.status === "workflow_complete" && "bg-purple-500/10 border-l-2 border-purple-400/50",
                      event.status === "connected" && "bg-white/5 border-l-2 border-white/20",
                      event.status === "error" && "bg-red-500/10 border-l-2 border-red-500/50",
                    )}
                  >
                    <span className="text-lg mt-0.5 shrink-0">{AGENT_ICONS[event.agent] || "🔹"}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-white/90 capitalize">{event.agent}</span>
                        <span className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded-full font-mono uppercase",
                          event.status === "started" && "bg-[#00E5FF]/20 text-[#00E5FF]",
                          event.status === "completed" && "bg-green-500/20 text-green-400",
                          event.status === "workflow_complete" && "bg-purple-500/20 text-purple-300",
                          event.status === "connected" && "bg-white/10 text-white/50",
                          event.status === "error" && "bg-red-500/20 text-red-400",
                        )}>
                          {event.status === "workflow_complete" ? "done" : event.status}
                        </span>
                      </div>
                      <p className="text-white/50 text-xs mt-0.5 truncate">{event.message}</p>
                    </div>
                    <span className="text-[10px] text-white/20 font-mono shrink-0 mt-1">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {workflowInfo!.status === "suspended" && !rerunInProgress && (
        <div className="mb-8 p-5 rounded-2xl border-2 border-[#00E5FF]/40 bg-[#00E5FF]/[0.02] shadow-[0_0_30px_rgba(0,229,255,0.05)] animate-fadeInUp">
          <div className="flex items-center gap-3 mb-4">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00E5FF] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-[#00E5FF]"></span>
            </span>
            <h2 className="text-xl font-bold text-white">Human-in-the-Loop Review</h2>
          </div>
          <p className="text-white/60 text-sm mb-6">The AI pipeline has completed its analysis and is awaiting your review. Approve the diagnosis or provide feedback to rerun the workflow with your guidance.</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-white/40 mb-2 uppercase tracking-wide">Doctor&apos;s Feedback</label>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Write your clinical observations, disagreements, or additional context for the AI to consider on rerun..."
                className="w-full bg-black/40 border border-white/[0.05] rounded-xl p-3 text-sm text-white resize-none h-[80px] placeholder:text-white/20 focus:border-[#00E5FF]/30 focus:outline-none transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-white/40 mb-2 uppercase tracking-wide">Suggested Diagnosis (optional)</label>
              <input
                type="text"
                value={correctDx}
                onChange={(e) => setCorrectDx(e.target.value)}
                placeholder="Your diagnosis if different from the AI's..."
                className="w-full bg-black/40 border border-white/[0.05] rounded-xl p-3 text-sm text-white mb-4 placeholder:text-white/20 focus:border-[#00E5FF]/30 focus:outline-none transition-colors"
              />
              <div className="flex gap-3">
                <button
                  onClick={() => handleFeedback("approve")}
                  disabled={isSubmittingFeedback}
                  className="flex-1 bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30 font-medium text-sm py-2.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmittingFeedback ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  Approve Diagnosis
                </button>
                <button
                  onClick={() => handleFeedback("reject")}
                  disabled={isSubmittingFeedback}
                  className="flex-1 bg-[#00E5FF]/20 text-[#00E5FF] border border-[#00E5FF]/30 hover:bg-[#00E5FF]/30 font-medium text-sm py-2.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmittingFeedback ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  Rerun with Feedback
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Diagnosis Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-6 mb-10 animate-fadeInUp-delay-1">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 mb-4 rounded-full border border-[#00E5FF]/20 bg-[#00E5FF]/[0.04] text-[11px] text-[#00E5FF] uppercase tracking-[0.15em] font-medium">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00E5FF] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#00E5FF]"></span>
            </span>
            Study {params.id.slice(0, 8)} &bull; {workflowInfo!.status === "completed" ? "Finalized" : "Pending Review"}
          </div>
          <h1 className="text-3xl md:text-5xl font-[var(--font-outfit)] font-bold text-white/90 leading-tight">
            <GradientText colors={["#00E5FF", "#64FFDA", "#00E5FF"]}>{finalDx}</GradientText>
          </h1>
        </div>

        {/* Metrics */}
        <div className="flex gap-3 w-full lg:w-auto">
          <div className="flex-1 lg:min-w-[140px] rounded-xl border border-white/[0.04] bg-white/[0.02] p-4 group hover:border-green-500/20 transition-colors">
            <p className="text-[11px] uppercase tracking-[0.15em] text-white/25 mb-1">Confidence</p>
            <span className="text-3xl font-bold text-green-400 font-[var(--font-outfit)]">{confidence}<span className="text-lg">%</span></span>
          </div>
          {safetyReport && (
            <div className="flex-1 lg:min-w-[140px] rounded-xl border border-white/[0.04] bg-white/[0.02] p-4 group hover:border-green-500/20 transition-colors">
              <p className="text-[11px] uppercase tracking-[0.15em] text-white/25 mb-1">Safety Score</p>
              <span className="text-3xl font-bold font-[var(--font-outfit)]" style={{ color: safetyColor(safetyReport.safety_score) }}>
                {Math.round(safetyReport.safety_score * 100)}<span className="text-lg">%</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Safety Alert Banner (if critical findings) */}
      {safetyReport && safetyReport.requires_immediate_action && (
        <div className="mb-6 p-4 rounded-xl border-2 border-red-500/40 bg-red-500/[0.06] animate-pulse">
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <span className="text-red-400 font-bold text-sm uppercase tracking-wide">
              Critical Finding — Immediate Action Required
            </span>
          </div>
          {safetyReport.critical_findings.map((cf, i) => (
            <p key={i} className="text-red-300/80 text-sm mt-2 ml-8">
              <strong>{cf.condition}</strong> [{cf.urgency}] — {cf.action}
            </p>
          ))}
        </div>
      )}

      {/* === System Entropy Cascade Chart === */}
      {uncertaintyHistory.length > 0 && (
        <div className="mb-8 rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6 animate-fadeInUp-delay-1">
          <div className="flex items-center gap-2 mb-5">
            <Activity className="h-4 w-4 text-[#00E5FF]" />
            <h3 className="text-sm font-semibold text-white/70 uppercase tracking-[0.12em]">System Entropy Cascade</h3>
            <span className="ml-auto text-[11px] text-white/25">per-agent MUC update</span>
          </div>
          <UncertaintyGraph data={uncertaintyHistory} />
        </div>
      )}

      {isRunning && (
        <div className="mb-8 p-6 rounded-2xl border border-[#00E5FF]/20 bg-black/40 shadow-inner">
          <h3 className="text-[#00E5FF] font-semibold flex items-center gap-2 mb-4">
            <Loader2 className="w-5 h-5 animate-spin" /> Live Agent Pipeline
          </h3>
          <div
            ref={feedRef}
            className="bg-black/60 rounded-xl p-4 max-h-64 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-white/10"
          >
            {liveEvents.length === 0 && <p className="text-white/30 text-sm py-4 text-center">Awaiting agent updates or polling backend...</p>}
            {liveEvents.map((event, i) => (
              <div key={i} className="flex items-start gap-3 py-2 px-3 rounded-lg text-sm bg-white/5 border-l-2 border-[#00E5FF]/30">
                <span className="text-lg mt-0.5 shrink-0">{AGENT_ICONS[event.agent] || "🔹"}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white/90 capitalize">{event.agent}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full font-mono uppercase bg-[#00E5FF]/20 text-[#00E5FF]">
                      {event.status === "workflow_complete" ? "done" : event.status}
                    </span>
                  </div>
                  <p className="text-white/50 text-xs mt-0.5">{event.message}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 animate-fadeInUp-delay-2">
        {/* Left Sidebar */}
        <div className="lg:col-span-4 space-y-5">

          {/* Radiologist Impression */}
          <div className="rounded-2xl border border-white/[0.04] bg-white/[0.02] p-5">
            <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
              <FileText className="h-4 w-4 text-[#00E5FF]" />
              Impression
            </h3>
            <blockquote className="text-sm text-[#00E5FF]/80 italic border-l-2 border-[#00E5FF]/30 pl-4 mb-4 leading-relaxed">
              &ldquo;{evidence?.visual?.impression || "Awaiting visual evidence... "}&rdquo;
            </blockquote>
            <p className="text-[13px] text-white/40 leading-relaxed">
              {evidence?.visual?.findings || "Processing structured findings..."}
            </p>
          </div>

          {/* Safety Summary Card */}
          {safetyReport && (
            <div className={cn(
              "rounded-2xl border p-5",
              safetyReport.passed
                ? "border-green-500/20 bg-green-500/[0.02]"
                : "border-red-500/20 bg-red-500/[0.02]"
            )}>
              <h3 className="text-sm font-semibold text-white/70 mb-3 flex items-center gap-2">
                <Shield className="h-4 w-4" style={{ color: safetyColor(safetyReport.safety_score) }} />
                Safety Guardrails
              </h3>
              <div className="flex items-center gap-3 mb-3">
                <span
                  className={cn(
                    "text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider",
                    safetyReport.passed
                      ? "bg-green-500/20 text-green-400"
                      : "bg-red-500/20 text-red-400"
                  )}
                >
                  {safetyReport.passed ? "PASSED" : "FAILED"}
                </span>
                <span className="text-xs text-white/30">
                  {safetyReport.red_flags.length} flag{safetyReport.red_flags.length !== 1 ? "s" : ""}
                </span>
              </div>
              <p className="text-[12px] text-white/40 leading-relaxed">{safetyReport.summary}</p>
            </div>
          )}
        </div>

        {/* Right Panel: Evidence Tabs */}
        <div className="lg:col-span-8 rounded-2xl border border-white/[0.04] bg-white/[0.015] overflow-hidden min-h-[500px] flex flex-col">
          {/* Tab Header */}
          <div className="flex border-b border-white/[0.04] bg-black/20 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center gap-1.5 px-5 py-3.5 text-[13px] font-medium transition-all whitespace-nowrap",
                  activeTab === tab.id
                    ? "text-[#00E5FF] border-b-2 border-[#00E5FF] bg-[#00E5FF]/[0.03]"
                    : "text-white/30 hover:text-white/50 hover:bg-white/[0.02]"
                )}
              >
                <tab.icon className="h-3.5 w-3.5" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-6 flex-1">
            {activeTab === "visual" && (
              <div className="space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.15em] text-white/25 mb-3">Original DICOM</p>
                    <div className="aspect-[4/3] bg-black/40 rounded-xl border border-white/[0.04] relative flex items-center justify-center overflow-hidden">
                      {workflowInfo!.current_state?.image_path ? (
                        <img src={`http://localhost:8000/${workflowInfo!.current_state.image_path}`} alt="Original View" className="object-contain w-full h-full" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-700/40 via-slate-800/60 to-black" />
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.15em] text-[#00E5FF]/50 mb-3 flex items-center gap-1">
                      Visual Heatmap
                    </p>
                    <div className="aspect-[4/3] bg-black/40 rounded-xl border border-[#00E5FF]/10 relative flex items-center justify-center overflow-hidden glow-cyan">
                      {workflowInfo!.current_state?.radiologist?.heatmap_paths && Object.keys(workflowInfo!.current_state.radiologist.heatmap_paths).length > 0 ? (
                        <img
                          src={`http://localhost:8000/${Object.values(workflowInfo!.current_state.radiologist.heatmap_paths)[0]}`}
                          alt="Heatmap"
                          className="object-contain w-full h-full mix-blend-screen"
                        />
                      ) : workflowInfo!.current_state?.image_path ? (
                        <img src={`http://localhost:8000/${workflowInfo!.current_state.image_path}`} alt="Original View Fallback" className="object-contain w-full h-full opacity-50 grayscale" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-700/40 via-slate-800/60 to-black" />
                      )}
                    </div>
                  </div>
                </div>
                {workflowInfo!.current_state?.radiologist?.findings && (
                  <div className="bg-[#00E5FF]/[0.04] border border-[#00E5FF]/10 rounded-xl p-4 text-[13px] text-[#00E5FF]/70 flex items-start gap-3">
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-[#00E5FF] mt-0.5" />
                    <div>{workflowInfo!.current_state.radiologist.findings}</div>
                  </div>
                )}
              </div>
            )}

            {activeTab === "clinical" && (
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-white/70 mb-3">Clinical Evidence Summary</h3>
                <div className="text-sm text-white/60 bg-black/20 p-4 rounded-lg border border-white/[0.03] leading-relaxed">
                  {workflowInfo!.current_state?.historian?.clinical_summary || "No clinical summary available yet."}
                </div>

                <h3 className="text-sm font-semibold text-green-400 mt-6 mb-3">Supporting Facts</h3>
                {workflowInfo!.current_state?.historian?.supporting_facts?.length > 0 ? (
                  workflowInfo!.current_state.historian.supporting_facts.map((fact: any, i: number) => (
                    <div key={`supp-${i}`} className="rounded-xl border border-white/[0.04] bg-black/20 p-4 border-l-2 border-l-green-400/40 transition-colors hover:bg-black/40">
                      <div className="text-sm text-white/60 leading-relaxed">{fact.description}</div>
                      <div className="text-[11px] text-white/30 font-mono mt-3 inline-block bg-white/[0.05] px-2 py-0.5 rounded">
                        {fact.fhir_resource_type} / {fact.fhir_resource_id}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-[13px] text-white/30 p-4 border border-white/[0.02] rounded-xl text-center">No supporting historical facts retrieved.</div>
                )}

                <h3 className="text-sm font-semibold text-red-500 mt-6 mb-3">Contradicting Facts</h3>
                {workflowInfo!.current_state?.historian?.contradicting_facts?.length > 0 ? (
                  workflowInfo!.current_state.historian.contradicting_facts.map((fact: any, i: number) => (
                    <div key={`cont-${i}`} className="rounded-xl border border-white/[0.04] bg-black/20 p-4 border-l-2 border-l-red-500/40 transition-colors hover:bg-black/40">
                      <div className="text-sm text-white/60 leading-relaxed">{fact.description}</div>
                      <div className="text-[11px] text-white/30 font-mono mt-3 inline-block bg-white/[0.05] px-2 py-0.5 rounded">
                        {fact.fhir_resource_type} / {fact.fhir_resource_id}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-[13px] text-white/30 p-4 border border-white/[0.02] rounded-xl text-center">No contradicting historical facts retrieved.</div>
                )}
              </div>
            )}

            {activeTab === "literary" && (
              <div className="space-y-4">
                {workflowInfo!.current_state?.literature?.citations && workflowInfo!.current_state.literature.citations.length > 0 ? (
                  workflowInfo!.current_state.literature.citations.map((c: any, i: number) => (
                    <div key={i} className="rounded-xl border border-white/[0.04] bg-black/20 p-5 hover:border-[#00E5FF]/10 transition-colors cursor-pointer group">
                      <h4 className="text-sm font-semibold text-white/80 mb-2 group-hover:text-[#00E5FF] transition-colors">
                        {c.title}
                      </h4>
                      <p className="text-[11px] text-[#00E5FF]/80 font-mono mb-3 flex items-center gap-1">
                        <BookOpen className="h-3 w-3" /> <a href={c.url} target="_blank" rel="noreferrer" className="hover:underline text-white/40">{c.url}</a>
                      </p>
                      <p className="text-[13px] text-white/50 leading-relaxed border-l-2 border-[#00E5FF]/20 pl-4">
                        {c.relevance_summary}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="p-5 text-[13px] text-white/40 text-center border rounded-xl border-white/[0.02] bg-white/[0.01]">
                    No literature citations retrieved for this context yet.
                  </div>
                )}
              </div>
            )}

            {/* NEW: Safety Tab */}
            {activeTab === "safety" && (
              <div className="space-y-4">
                {safetyLoading ? (
                  <div className="flex flex-col items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 text-[#00E5FF] animate-spin mb-4" />
                    <p className="text-white/40 text-sm">Running safety validation...</p>
                  </div>
                ) : safetyReport ? (
                  <>
                    {/* Safety Score Header */}
                    <div className={cn(
                      "rounded-xl border p-5",
                      safetyReport.passed
                        ? "border-green-500/15 bg-green-500/[0.03]"
                        : "border-red-500/15 bg-red-500/[0.03]"
                    )}>
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <Shield className="h-5 w-5" style={{ color: safetyColor(safetyReport.safety_score) }} />
                          <span className="text-lg font-bold" style={{ color: safetyColor(safetyReport.safety_score) }}>
                            {Math.round(safetyReport.safety_score * 100)}%
                          </span>
                          <span className={cn(
                            "text-[11px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider",
                            safetyReport.passed ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                          )}>
                            {safetyReport.passed ? "PASSED" : "FAILED"}
                          </span>
                        </div>
                        <span className="text-[11px] text-white/25 font-mono">
                          Hallucination risk: {safetyReport.hallucination_risk}
                        </span>
                      </div>
                      <p className="text-[13px] text-white/50">{safetyReport.summary}</p>
                    </div>

                    {/* Critical Findings */}
                    {safetyReport.critical_findings.length > 0 && (
                      <div>
                        <p className="text-[11px] uppercase tracking-[0.15em] text-red-400/80 mb-3 font-semibold">Critical Findings</p>
                        {safetyReport.critical_findings.map((cf, i) => (
                          <div key={i} className="rounded-xl border border-red-500/15 bg-red-500/[0.04] p-4 mb-2">
                            <div className="flex items-center gap-2 mb-2">
                              <AlertTriangle className="h-4 w-4 text-red-400" />
                              <span className="text-sm font-semibold text-red-300">{cf.condition}</span>
                              <span className="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 font-mono">{cf.urgency}</span>
                              {cf.icd10 && <span className="text-[10px] text-white/20 font-mono">ICD-10: {cf.icd10}</span>}
                            </div>
                            <p className="text-[13px] text-red-300/70">{cf.action}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Red Flags */}
                    {safetyReport.red_flags.length > 0 && (
                      <div>
                        <p className="text-[11px] uppercase tracking-[0.15em] text-white/40 mb-3 font-semibold">Red Flags</p>
                        {safetyReport.red_flags.map((flag, i) => (
                          <div key={i} className={cn(
                            "rounded-xl border p-4 mb-2",
                            flag.severity === "high" ? "border-red-500/15 bg-red-500/[0.03]" :
                              flag.severity === "medium" ? "border-yellow-500/15 bg-yellow-500/[0.03]" :
                                "border-green-500/15 bg-green-500/[0.03]"
                          )}>
                            <div className="flex items-start gap-3">
                              {severityIcon(flag.severity)}
                              <div className="flex-1">
                                <p className="text-sm font-medium text-white/70 mb-1">{flag.flag_type}</p>
                                <p className="text-[13px] text-white/40">{flag.description}</p>
                                <p className="text-[12px] text-[#00E5FF]/50 mt-2">💡 {flag.recommendation}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Safety Recommendations */}
                    {safetyReport.recommendations.length > 0 && (
                      <div>
                        <p className="text-[11px] uppercase tracking-[0.15em] text-white/40 mb-3 font-semibold">Recommendations</p>
                        {safetyReport.recommendations.map((rec, i) => (
                          <div key={i} className="rounded-lg border border-white/[0.04] bg-black/20 p-3 mb-2 text-[13px] text-white/50 flex items-start gap-2">
                            <CheckCircle2 className="h-3.5 w-3.5 text-[#00E5FF] shrink-0 mt-0.5" />
                            {rec}
                          </div>
                        ))}
                      </div>
                    )}

                    {safetyReport.red_flags.length === 0 && safetyReport.critical_findings.length === 0 && (
                      <div className="flex flex-col items-center py-12 text-center">
                        <CheckCircle2 className="h-12 w-12 text-green-400/60 mb-4" />
                        <p className="text-white/60 font-medium">All safety checks passed</p>
                        <p className="text-white/30 text-sm mt-1">No critical findings or red flags detected.</p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col items-center py-12 text-center">
                    <Shield className="h-12 w-12 text-white/20 mb-4" />
                    <p className="text-white/40 text-sm">Safety report unavailable</p>
                    <button onClick={fetchSafetyReport} className="mt-3 text-[#00E5FF] text-sm hover:underline">
                      Retry
                    </button>
                  </div>
                )}
              </div>
            )}

            {activeTab === "audit" && (
              <div className="space-y-6">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.15em] text-white/40 mb-3 font-semibold ml-1">Execution Trace</p>
                  <div className="space-y-3 relative pl-8 mt-2">
                    <div className="absolute left-3 top-0 bottom-0 w-px bg-gradient-to-b from-[#00E5FF]/30 via-white/5 to-transparent" />
                    {(workflowInfo!.current_state?.trace || workflowInfo!.final_result?.trace || [])?.map((traceStr: string, index: number) => (
                      <div key={index} className="relative flex items-start gap-4 group">
                        <div
                          className="absolute -left-5 w-6 h-6 rounded-full border-2 flex items-center justify-center bg-[#050507] z-10 group-hover:scale-110 transition-transform cursor-pointer"
                          style={{ borderColor: `#00E5FF50` }}
                        >
                          <Activity className="h-3 w-3 text-[#00E5FF]" />
                        </div>
                        <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 flex-1 group-hover:border-white/[0.08] transition-colors">
                          <p className="text-[13px] text-white/40 leading-relaxed">{traceStr}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Raw Output Dumps */}
                <div className="grid grid-cols-1 gap-4 pt-6 mt-6 border-t border-white/5">
                  <p className="text-[14px] uppercase tracking-[0.15em] text-white/60 font-semibold mb-2">Raw Agent Outputs</p>

                  {workflowInfo?.current_state?.radiologist && (
                    <div className="rounded-xl border border-blue-500/10 bg-blue-500/[0.02] overflow-hidden">
                      <div className="bg-blue-500/10 px-4 py-2 text-xs font-mono text-blue-400 font-semibold border-b border-blue-500/10">Radiologist Component</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.radiologist, null, 2)}
                      </pre>
                    </div>
                  )}

                  {workflowInfo?.current_state?.chexbert && (
                    <div className="rounded-xl border border-indigo-500/10 bg-indigo-500/[0.02] overflow-hidden">
                      <div className="bg-indigo-500/10 px-4 py-2 text-xs font-mono text-indigo-400 font-semibold border-b border-indigo-500/10">CheXbert Pathologies</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.chexbert, null, 2)}
                      </pre>
                    </div>
                  )}

                  {workflowInfo?.current_state?.historian && (
                    <div className="rounded-xl border border-emerald-500/10 bg-emerald-500/[0.02] overflow-hidden">
                      <div className="bg-emerald-500/10 px-4 py-2 text-xs font-mono text-emerald-400 font-semibold border-b border-emerald-500/10">Historian Inference</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.historian, null, 2)}
                      </pre>
                    </div>
                  )}

                  {workflowInfo?.current_state?.literature && (
                    <div className="rounded-xl border border-yellow-500/10 bg-yellow-500/[0.02] overflow-hidden">
                      <div className="bg-yellow-500/10 px-4 py-2 text-xs font-mono text-yellow-400 font-semibold border-b border-yellow-500/10">Literature Agent</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.literature, null, 2)}
                      </pre>
                    </div>
                  )}

                  {workflowInfo?.current_state?.critic && (
                    <div className="rounded-xl border border-pink-500/10 bg-pink-500/[0.02] overflow-hidden">
                      <div className="bg-pink-500/10 px-4 py-2 text-xs font-mono text-pink-400 font-semibold border-b border-pink-500/10">Critic Evaluation</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.critic, null, 2)}
                      </pre>
                    </div>
                  )}

                  {workflowInfo?.current_state?.debate && (
                    <div className="rounded-xl border border-orange-500/10 bg-orange-500/[0.02] overflow-hidden">
                      <div className="bg-orange-500/10 px-4 py-2 text-xs font-mono text-orange-400 font-semibold border-b border-orange-500/10">Debate Orchestrator</div>
                      <pre className="p-4 text-[11px] text-white/50 font-mono overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(workflowInfo.current_state.debate, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
