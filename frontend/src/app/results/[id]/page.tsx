"use client";
import { useState, useEffect } from "react";
import { CheckCircle2, AlertTriangle, FileText, ImageIcon, BookOpen, Activity, User, ShieldAlert, ChevronLeft, Download, Database, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { GradientText } from "@/components/GradientText";
import { checkWorkflowStatus, submitHumanFeedback, WorkflowStatusResponse } from "@/lib/api";

export default function ResultsPage({ params }: { params: { id: string } }) {
  const [activeTab, setActiveTab] = useState("visual");
  const [workflowInfo, setWorkflowInfo] = useState<WorkflowStatusResponse | null>(null);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [feedbackText, setFeedbackText] = useState("");
  const [correctDx, setCorrectDx] = useState("");

  // Polling Effect
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const poll = async () => {
      try {
        const data = await checkWorkflowStatus(params.id);
        setWorkflowInfo(data);

        // Stop polling if completed or failed
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    // Initial fetch
    poll();

    // Poll every 3 seconds if not completed/failed
    intervalId = setInterval(poll, 3000);

    return () => clearInterval(intervalId);
  }, [params.id]);

  const handleFeedback = async (action: "approve" | "reject") => {
    if (action === "reject" && !feedbackText.trim()) {
      alert("Please provide the reason for rejection.");
      return;
    }

    setIsSubmittingFeedback(true);
    try {
      await submitHumanFeedback(params.id, action, feedbackText, correctDx);
      // Wait a moment and then fetch status to trigger "running" again
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
    { id: "audit", label: "Audit Trail", icon: ShieldAlert },
  ];

  const renderLoadingState = () => {
    const state = workflowInfo?.current_state || {};
    const traceLength = state.trace?.length || 0;

    // Determine active stage
    let activeStage = "Initializing Multi-Agent Graph...";
    if (state.radiologist) activeStage = "Radiologist Analysis Complete";
    if (state.chexbert) activeStage = "CheXbert Semantic Extraction Complete";
    if (state.historian) activeStage = "Historian Clinical Anchoring Complete";
    if (state.literature) activeStage = "Literature Search Complete";
    if (state.critic) activeStage = "Critic Verification Complete";
    if (state.debate) activeStage = "Multi-Agent Debate Complete";

    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6">
        <Loader2 className="w-12 h-12 text-[#00E5FF] animate-spin" />
        <h2 className="text-2xl font-semibold text-white/80">Analyzing Study...</h2>
        <p className="text-[#00E5FF]/80 font-medium tracking-wide">{activeStage}</p>
        <p className="text-white/40 text-sm">Processed {traceLength} internal reasoning steps so far.</p>

        {/* Agent Checklist */}
        <div className="w-full max-w-sm bg-black/40 border border-white/[0.05] rounded-xl p-6 mt-6 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <span className={`text-[13px] font-medium transition-colors ${state.radiologist ? "text-white/80" : "text-white/40"}`}>1. Vision Backbone (Radiologist)</span>
            {state.radiologist ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <Loader2 className="w-4 h-4 text-[#00E5FF] animate-spin" />}
          </div>
          <div className="flex items-center justify-between">
            <span className={`text-[13px] font-medium transition-colors ${state.chexbert ? "text-white/80" : "text-white/40"}`}>2. Structured Pathology (CheXbert)</span>
            {state.chexbert ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : (state.radiologist ? <Loader2 className="w-4 h-4 text-[#00E5FF] animate-spin" /> : <span className="w-4 h-4 rounded-full border border-white/20" />)}
          </div>
          <div className="flex items-center justify-between">
            <span className={`text-[13px] font-medium transition-colors ${state.historian ? "text-white/80" : "text-white/40"}`}>3. EHR Context (Historian)</span>
            {state.historian ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : (state.chexbert ? <Loader2 className="w-4 h-4 text-[#00E5FF] animate-spin" /> : <span className="w-4 h-4 rounded-full border border-white/20" />)}
          </div>
          <div className="flex items-center justify-between">
            <span className={`text-[13px] font-medium transition-colors ${state.literature ? "text-white/80" : "text-white/40"}`}>4. Evidence Retrieval (Literature)</span>
            {state.literature ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : (state.historian ? <Loader2 className="w-4 h-4 text-[#00E5FF] animate-spin" /> : <span className="w-4 h-4 rounded-full border border-white/20" />)}
          </div>
          <div className="flex items-center justify-between">
            <span className={`text-[13px] font-medium transition-colors ${state.critic ? "text-white/80" : "text-white/40"}`}>5. Adversarial Validation (Critic)</span>
            {state.critic ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : (state.literature ? <Loader2 className="w-4 h-4 text-[#00E5FF] animate-spin" /> : <span className="w-4 h-4 rounded-full border border-white/20" />)}
          </div>
        </div>

        <div className="flex gap-2 text-[10px] text-[#00E5FF]/70 bg-[#00E5FF]/10 px-3 py-1 rounded-full font-mono uppercase tracking-widest mt-8">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-ping inline-block mt-[2px]"></span>
          Session {params.id.slice(0, 8)}
        </div>
      </div>
    );
  };

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
    // Human review payload mapped from Human Review Node
    finalDx = workflowInfo.pending_review_data.diagnosis;
    confidence = Math.round(workflowInfo.pending_review_data.confidence * 100);
    evidence = workflowInfo.pending_review_data.evidence;
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 mb-20 relative z-10">
      {/* Top Bar */}
      <div className="flex justify-between items-center mb-8 animate-fadeInUp">
        <Link href="/diagnose" className="text-white/30 hover:text-white/60 text-sm flex items-center gap-1 transition-colors group">
          <ChevronLeft className="h-4 w-4 group-hover:-translate-x-0.5 transition-transform" /> Back
        </Link>
        <button className="flex items-center gap-2 px-4 py-2 text-[13px] text-[#00E5FF] bg-[#00E5FF]/[0.06] border border-[#00E5FF]/15 rounded-lg hover:bg-[#00E5FF]/10 transition-all glow-cyan">
          <Download className="h-3.5 w-3.5" /> Export PDF
        </button>
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
              <label className="block text-xs text-white/40 mb-2 uppercase tracking-wide">Doctor's Assessment</label>
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
                  Reject & Rerun
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
        </div>
      </div>

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
        </div>

        {/* Right Panel: Evidence Tabs */}
        <div className="lg:col-span-8 rounded-2xl border border-white/[0.04] bg-white/[0.015] overflow-hidden min-h-[500px] flex flex-col">
          {/* Tab Header */}
          <div className="flex border-b border-white/[0.04] bg-black/20">
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
