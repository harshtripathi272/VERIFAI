"use client";
import { useState } from "react";
import { CheckCircle2, AlertTriangle, FileText, ImageIcon, BookOpen, Activity, User, ShieldAlert, ChevronLeft, Download, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { GradientText } from "@/components/GradientText";

export default function ResultsPage({ params }: { params: { id: string } }) {
  const [activeTab, setActiveTab] = useState("visual");

  const tabs = [
    { id: "visual", label: "Visual Proof", icon: ImageIcon },
    { id: "clinical", label: "Clinical", icon: User },
    { id: "literary", label: "Literature", icon: BookOpen },
    { id: "audit", label: "Audit Trail", icon: ShieldAlert },
  ];

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

      {/* Diagnosis Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-6 mb-10 animate-fadeInUp-delay-1">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 mb-4 rounded-full border border-[#00E5FF]/20 bg-[#00E5FF]/[0.04] text-[11px] text-[#00E5FF] uppercase tracking-[0.15em] font-medium">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00E5FF] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#00E5FF]"></span>
            </span>
            Study {params.id} &bull; Verified
          </div>
          <h1 className="text-3xl md:text-5xl font-[var(--font-outfit)] font-bold text-white/90 leading-tight">
            <GradientText colors={["#00E5FF", "#64FFDA", "#00E5FF"]}>Pneumocystis Pneumonia</GradientText>
          </h1>
          <p className="text-white/30 mt-3 text-sm flex items-center gap-2">
            <User className="h-3.5 w-3.5 text-[#00E5FF]/60" /> MRN-74892 &bull; 45 Y.O. Male &bull; Immunocompromised
          </p>
        </div>

        {/* Metrics */}
        <div className="flex gap-3 w-full lg:w-auto">
          <div className="flex-1 lg:min-w-[140px] rounded-xl border border-white/[0.04] bg-white/[0.02] p-4 group hover:border-green-500/20 transition-colors">
            <p className="text-[11px] uppercase tracking-[0.15em] text-white/25 mb-1">Confidence</p>
            <span className="text-3xl font-bold text-green-400 font-[var(--font-outfit)]">87<span className="text-lg">%</span></span>
          </div>
          <div className="flex-1 lg:min-w-[140px] rounded-xl border border-white/[0.04] bg-white/[0.02] p-4 group hover:border-yellow-500/20 transition-colors">
            <p className="text-[11px] uppercase tracking-[0.15em] text-white/25 mb-1 flex items-center gap-1">
              Uncertainty <AlertTriangle className="h-3 w-3 text-yellow-500/50" />
            </p>
            <span className="text-3xl font-bold text-yellow-500 font-[var(--font-outfit)]">25<span className="text-lg">%</span></span>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 animate-fadeInUp-delay-2">
        {/* Left Sidebar */}
        <div className="lg:col-span-4 space-y-5">
          {/* CheXbert Labels */}
          <div className="rounded-2xl border border-white/[0.04] bg-white/[0.02] p-5">
            <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#00E5FF]" />
              Structured Pathology
            </h3>
            <div className="space-y-2">
              {[
                { label: "Pneumonia", status: "present", color: "green" },
                { label: "Consolidation", status: "present", color: "green" },
                { label: "Pleural Effusion", status: "uncertain", color: "yellow" },
                { label: "Cardiomegaly", status: "absent", color: "neutral" },
              ].map((item) => (
                <div key={item.label} className="flex justify-between items-center p-3 rounded-lg bg-black/20 border border-white/[0.03]">
                  <span className={`text-sm ${item.color === "neutral" ? "text-white/25" : "text-white/70"}`}>{item.label}</span>
                  <span className={cn("text-[11px] px-2 py-0.5 rounded-full font-medium", {
                    "bg-green-500/10 text-green-400 border border-green-500/20": item.color === "green",
                    "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20": item.color === "yellow",
                    "bg-white/[0.03] text-white/20 border border-white/[0.03]": item.color === "neutral",
                  })}>
                    {item.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Radiologist Impression */}
          <div className="rounded-2xl border border-white/[0.04] bg-white/[0.02] p-5">
            <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
              <FileText className="h-4 w-4 text-[#00E5FF]" />
              Impression
            </h3>
            <blockquote className="text-sm text-[#00E5FF]/80 italic border-l-2 border-[#00E5FF]/30 pl-4 mb-4 leading-relaxed">
              &ldquo;Bilateral diffuse ground-glass opacities, predominantly perihilar.&rdquo;
            </blockquote>
            <p className="text-[13px] text-white/40 leading-relaxed">
              Highly suspicious for PCP given immunocompromised status. Recommend correlation with CD4 count and sputum analysis.
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

                {[
                  { agent: "Radiologist", time: "0ms", color: "#00E5FF", detail: "Generated findings. Token entropy: 0.12 (High Confidence).", icon: Activity },
                  { agent: "CheXbert", time: "+120ms", color: "#64FFDA", detail: "Extracted 3 labels: Pneumonia (present), Consolidation (present), Pleural Effusion (uncertain).", icon: Activity },
                  { agent: "Historian", time: "+450ms", color: "#7C4DFF", detail: "FHIR query returned CD4=180. Strongly supports PCP hypothesis.", icon: Database },
                  { agent: "Critic", time: "+1200ms", color: "#FFD740", detail: "Flagged Pleural Effusion as uncertain. Increased overall uncertainty by +0.15.", icon: ShieldAlert },
                ].map((item) => (
                  <div key={item.agent} className="relative flex items-start gap-4 group">
                    <div
                      className="absolute -left-5 w-6 h-6 rounded-full border-2 flex items-center justify-center bg-[#050507] z-10 group-hover:scale-110 transition-transform"
                      style={{ borderColor: `${item.color}50` }}
                    >
                      <item.icon className="h-3 w-3" style={{ color: item.color }} />
                    </div>
                    <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 flex-1 group-hover:border-white/[0.08] transition-colors">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-semibold" style={{ color: item.color }}>{item.agent}</span>
                        <span className="text-[10px] text-white/20 font-mono bg-white/[0.03] px-2 py-0.5 rounded">{item.time}</span>
                      </div>
                      <p className="text-[13px] text-white/40 leading-relaxed">{item.detail}</p>
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
