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
        } catch {}
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
        }
      } catch {}
    };

    poll();
    intervalId = setInterval(poll, 5000);

    return () => {
      clearInterval(intervalId);
      es?.close();
    };
  }, [params.id]);

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
      alert("Please provide the reason for rejection.");
      return;
    }

    setIsSubmittingFeedback(true);
    try {
      await submitHumanFeedback(params.id, action, feedbackText, correctDx);
      setTimeout(async () => {
        const data = await checkWorkflowStatus(params.id);
        setWorkflowInfo(data);
        setIsSubmittingFeedback(false);
      }, 1500);
    } catch (err) {
      console.error(err);
      alert("Failed to submit feedback.");
      setIsSubmittingFeedback(false);
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

  if (!workflowInfo || workflowInfo.status === "running") {
    return <div className="max-w-7xl mx-auto px-6 py-8">{renderLoadingState()}</div>;
  }

  // --- Real Logic variables for completed or suspended --- //
  let finalDx = "Pending Diagnosis";
  let confidence = 0;
  let uncertainty = 0;
  let evidence = null;

  if (workflowInfo.status === "completed" && workflowInfo.final_result) {
    finalDx = workflowInfo.final_result.diagnosis || "Undetermined";
    confidence = Math.round(workflowInfo.final_result.confidence * 100);
    evidence = workflowInfo.final_result.evidence_packet;
  } else if (workflowInfo.status === "suspended" && workflowInfo.pending_review_data) {
    finalDx = workflowInfo.pending_review_data.diagnosis;
    confidence = Math.round(workflowInfo.pending_review_data.confidence * 100);
    evidence = workflowInfo.pending_review_data.evidence;
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

      {workflowInfo.status === "suspended" && (
        <div className="mb-8 p-5 rounded-2xl border-2 border-[#00E5FF]/40 bg-[#00E5FF]/[0.02] shadow-[0_0_30px_rgba(0,229,255,0.05)] animate-fadeInUp">
          <div className="flex items-center gap-3 mb-4">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00E5FF] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-[#00E5FF]"></span>
            </span>
            <h2 className="text-xl font-bold text-white">Pending Radiologist Review</h2>
          </div>
          <p className="text-white/60 text-sm mb-6">The AI pipeline requires final sign-off. Please review the visual and clinical evidence below.</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-white/40 mb-2 uppercase tracking-wide">Doctor&apos;s Assessment</label>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Reason for rejection or additional notes..."
                className="w-full bg-black/40 border border-white/[0.05] rounded-xl p-3 text-sm text-white resize-none h-[80px]"
              />
            </div>
            <div>
              <label className="block text-xs text-white/40 mb-2 uppercase tracking-wide">Correction (if rejecting)</label>
              <input
                type="text"
                value={correctDx}
                onChange={(e) => setCorrectDx(e.target.value)}
                placeholder="Correct diagnosis..."
                className="w-full bg-black/40 border border-white/[0.05] rounded-xl p-3 text-sm text-white mb-4"
              />
              <div className="flex gap-3">
                <button
                  onClick={() => handleFeedback("approve")}
                  disabled={isSubmittingFeedback}
                  className="flex-1 bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30 font-medium text-sm py-2.5 rounded-lg transition-colors"
                >
                  Approve Diagnosis
                </button>
                <button
                  onClick={() => handleFeedback("reject")}
                  disabled={isSubmittingFeedback}
                  className="flex-1 bg-red-500/20 text-red-500 border border-red-500/30 hover:bg-red-500/30 font-medium text-sm py-2.5 rounded-lg transition-colors"
                >
                  Reject &amp; Rerun
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
            Study {params.id.slice(0, 8)} &bull; {workflowInfo.status === "completed" ? "Finalized" : "Pending Review"}
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
                    <div className="aspect-[4/3] bg-black/40 rounded-xl border border-white/[0.04] relative overflow-hidden">
                      <div className="w-full h-full bg-gradient-to-br from-slate-700/40 via-slate-800/60 to-black" />
                      <p className="absolute bottom-3 left-3 text-[10px] text-white/20 font-mono">AP View &bull; 14:02:55</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.15em] text-[#00E5FF]/50 mb-3 flex items-center gap-1">
                      Grad-CAM Heatmap
                    </p>
                    <div className="aspect-[4/3] bg-black/40 rounded-xl border border-[#00E5FF]/10 relative overflow-hidden glow-cyan">
                      <div className="w-full h-full bg-gradient-to-br from-slate-700/40 via-slate-800/60 to-black" />
                      <div className="absolute top-1/4 left-1/4 w-28 h-36 bg-red-500/30 rounded-[100%] blur-[25px] animate-pulse" />
                      <div className="absolute top-1/3 right-1/4 w-32 h-40 bg-orange-500/20 rounded-[100%] blur-[30px] animate-pulse" style={{ animationDelay: "1s" }} />
                    </div>
                  </div>
                </div>
                <div className="bg-[#00E5FF]/[0.04] border border-[#00E5FF]/10 rounded-xl p-4 text-[13px] text-[#00E5FF]/70 flex items-start gap-3">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#00E5FF] mt-0.5" />
                  Peak activation in perihilar regions bilaterally, consistent with PCP visual presentation.
                </div>
              </div>
            )}

            {activeTab === "clinical" && (
              <div className="space-y-4">
                <div className="rounded-xl border border-white/[0.04] bg-black/20 p-5 border-l-2 border-l-[#00E5FF]/40">
                  <div className="flex justify-between items-start mb-3">
                    <h4 className="text-sm font-medium text-white/70 flex items-center gap-2">
                      <FileText className="h-3.5 w-3.5 text-[#00E5FF]" /> FHIR Condition Resource
                    </h4>
                    <span className="text-[11px] text-white/20 font-mono bg-white/[0.03] px-2 py-1 rounded">3 days ago</span>
                  </div>
                  <div className="text-sm text-white/50 bg-black/30 p-4 rounded-lg font-mono leading-relaxed border border-white/[0.03]">
                    Patient presented with progressive dyspnea and dry cough. Known HIV, CD4 count 180 cells/&micro;L. On prophylactic TMP-SMX.
                  </div>
                  <div className="flex flex-wrap gap-2 mt-4">
                    <span className="text-[11px] px-2.5 py-1 rounded-full bg-[#00E5FF]/[0.06] text-[#00E5FF] border border-[#00E5FF]/15">Immunocompromised</span>
                    <span className="text-[11px] px-2.5 py-1 rounded-full bg-red-500/[0.06] text-red-400 border border-red-500/15">CD4 = 180</span>
                    <span className="text-[11px] px-2.5 py-1 rounded-full bg-white/[0.03] text-white/30 border border-white/[0.04]">Dyspnea</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "literary" && (
              <div className="space-y-4">
                <div className="rounded-xl border border-white/[0.04] bg-black/20 p-5 hover:border-[#00E5FF]/10 transition-colors cursor-pointer group">
                  <div className="flex items-center gap-2 mb-3 text-[11px] text-[#00E5FF] font-medium tracking-wide">
                    <CheckCircle2 className="h-3.5 w-3.5" /> HIGH RELEVANCE
                    <span className="bg-[#00E5FF]/10 px-1.5 py-0.5 rounded font-mono text-[10px]">0.92</span>
                  </div>
                  <h4 className="text-sm font-medium text-white/80 mb-2 group-hover:text-[#00E5FF] transition-colors">
                    Radiographic manifestations of Pneumocystis jirovecii pneumonia in HIV patients
                  </h4>
                  <p className="text-[11px] text-white/25 font-mono mb-3 flex items-center gap-1">
                    <BookOpen className="h-3 w-3" /> J Thoracic Imaging &bull; 2021 &bull; PMID: 33458291
                  </p>
                  <p className="text-[13px] text-white/40 leading-relaxed border-l-2 border-[#00E5FF]/20 pl-4">
                    Bilateral ground-glass opacities are the hallmark of PCP on chest radiography, occurring in up to 90% of cases...
                  </p>
                </div>
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
              <div className="space-y-5 relative pl-8">
                <div className="absolute left-3 top-0 bottom-0 w-px bg-gradient-to-b from-[#00E5FF]/30 via-white/5 to-transparent" />

                {(workflowInfo.status === "completed" ? workflowInfo.final_result?.trace : [])?.map((traceStr: string, index: number) => (
                  <div key={index} className="relative flex items-start gap-4 group">
                    <div
                      className="absolute -left-5 w-6 h-6 rounded-full border-2 flex items-center justify-center bg-[#050507] z-10 group-hover:scale-110 transition-transform"
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
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
