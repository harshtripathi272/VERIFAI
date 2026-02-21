"use client";
import { useState } from "react";
import { CheckCircle2, AlertTriangle, FileText, ImageIcon, BookOpen, Activity, User, ShieldAlert, ChevronLeft, Download, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

export default function ResultsPage({ params }: { params: { id: string } }) {
  const [activeTab, setActiveTab] = useState("visual");

  const tabs = [
    { id: "visual", label: "Visual Proof", icon: ImageIcon },
    { id: "clinical", label: "Clinical Proof", icon: User },
    { id: "literary", label: "Literary Proof", icon: BookOpen },
    { id: "audit", label: "Audit Trail", icon: ShieldAlert },
  ];

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-8 mb-20 animate-in fade-in zoom-in-95 duration-500">
      
      <div className="mb-6 flex justify-between items-center bg-background/50 sticky top-16 z-40 py-4 backdrop-blur-md border-b border-transparent">
        <Link href="/diagnose" className="text-muted-foreground hover:text-foreground inline-flex items-center text-sm transition-colors cursor-pointer group">
          <ChevronLeft className="h-4 w-4 mr-1 group-hover:-translate-x-1 transition-transform" /> Back to Dashboard
        </Link>
        <button className="inline-flex items-center px-4 py-2 border border-primary/20 bg-primary/5 rounded-md text-primary text-sm hover:bg-primary/10 transition-colors shadow-[0_0_10px_rgba(0,229,255,0.1)] hover:shadow-[0_0_15px_rgba(0,229,255,0.2)]">
          <Download className="h-4 w-4 mr-2" /> Export Evidence Packet
        </button>
      </div>

      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-6">
        <div>
          <div className="inline-flex items-center px-3 py-1 mb-4 rounded-full border border-primary/30 bg-primary/10 text-primary text-xs font-semibold tracking-wider uppercase shadow-[0_0_10px_rgba(0,229,255,0.1)]">
            <span className="flex h-1.5 w-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_rgba(0,229,255,1)]"></span>
            Study {params.id} • Verified
          </div>
          <h1 className="text-3xl md:text-5xl font-heading font-bold text-foreground">
            Pneumocystis Pneumonia <span className="text-muted-foreground font-normal">(PCP)</span>
          </h1>
          <p className="text-muted-foreground mt-3 flex items-center bg-white/5 inline-flex px-3 py-1 rounded-md text-sm border border-white/10">
            <User className="h-4 w-4 mr-2 text-primary" /> MRN-74892 • 45 Y.O. Male • Immunocompromised
          </p>
        </div>
        
        <div className="flex gap-4 w-full md:w-auto">
          <div className="bg-card backdrop-blur-md border border-white/10 rounded-xl p-4 min-w-[140px] flex-1 md:flex-none relative overflow-hidden group hover:border-green-500/50 transition-colors">
            <div className="absolute inset-0 bg-gradient-to-tr from-green-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <p className="text-xs text-muted-foreground uppercase tracking-widest mb-1">Confidence</p>
            <div className="flex items-end gap-2">
              <span className="text-4xl font-bold text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.5)]">87%</span>
            </div>
          </div>
          <div className="bg-card backdrop-blur-md border border-white/10 rounded-xl p-4 min-w-[140px] flex-1 md:flex-none relative overflow-hidden group hover:border-yellow-500/50 transition-colors">
            <div className="absolute inset-0 bg-gradient-to-tr from-yellow-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <p className="text-xs text-muted-foreground uppercase tracking-widest mb-1 flex items-center">
              Uncertainty <AlertTriangle className="h-3 w-3 ml-1 text-yellow-500" />
            </p>
            <div className="flex items-end gap-2">
              <span className="text-4xl font-bold text-yellow-500 drop-shadow-[0_0_10px_rgba(234,179,8,0.5)]">25%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: CheXbert Labels & Impression */}
        <div className="space-y-6 lg:col-span-4">
          <div className="glass p-6 rounded-xl border border-white/5 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:scale-110 transition-transform duration-500">
              <Activity className="h-24 w-24" />
            </div>
            <h3 className="font-heading font-semibold text-lg mb-4 flex items-center relative z-10">
              <Activity className="h-5 w-5 mr-2 text-primary" />
              Structured Pathology
            </h3>
            <div className="space-y-3 relative z-10">
              <div className="flex justify-between items-center p-3 rounded-lg bg-black/40 border border-white/5 hover:border-primary/30 transition-colors">
                <span className="text-sm font-medium">Pneumonia</span>
                <span className="text-xs px-2 py-1 rounded bg-green-500/10 text-green-400 border border-green-500/20 shadow-[0_0_10px_rgba(74,222,128,0.1)]">Present</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-black/40 border border-white/5 hover:border-primary/30 transition-colors">
                <span className="text-sm font-medium">Consolidation</span>
                <span className="text-xs px-2 py-1 rounded bg-green-500/10 text-green-400 border border-green-500/20 shadow-[0_0_10px_rgba(74,222,128,0.1)]">Present</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-black/40 border border-white/5 hover:border-yellow-500/30 transition-colors">
                <span className="text-sm font-medium">Pleural Effusion</span>
                <span className="text-xs px-2 py-1 rounded bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 shadow-[0_0_10px_rgba(234,179,8,0.1)] animate-pulse">Uncertain</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-black/40 border border-white/5">
                <span className="text-sm font-medium text-muted-foreground">Cardiomegaly</span>
                <span className="text-xs px-2 py-1 rounded bg-white/5 text-muted-foreground border border-white/5">Not Mentioned</span>
              </div>
            </div>
          </div>

          <div className="glass p-6 rounded-xl border border-white/5 group hover:border-white/10 transition-colors">
            <h3 className="font-heading font-semibold text-lg mb-4 flex items-center">
              <FileText className="h-5 w-5 mr-2 text-primary group-hover:rotate-12 transition-transform" />
              Radiologist Impression
            </h3>
            <div className="p-4 rounded-lg bg-primary/5 text-primary border border-primary/20 text-sm italic mb-4 shadow-[inset_0_0_20px_rgba(0,229,255,0.05)]">
              "Bilateral diffuse ground-glass opacities, predominantly perihilar."
            </div>
            <p className="text-sm text-foreground/80 leading-relaxed">
              Findings are highly suspicious for Pneumocystis jirovecii pneumonia given the patient&apos;s immunocompromised status. Mild pleural effusion cannot be entirely excluded. Recommend correlation with clinical markers and sputum analysis.
            </p>
          </div>
        </div>

        {/* Right Column: Evidence Tabs */}
        <div className="glass rounded-xl border border-white/5 lg:col-span-8 flex flex-col overflow-hidden min-h-[500px] shadow-2xl shadow-black/50">
          <div className="border-b border-white/5 flex overflow-x-auto no-scrollbar bg-black/20">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center px-6 py-4 text-sm font-medium outline-none transition-all whitespace-nowrap",
                  activeTab === tab.id 
                    ? "text-primary border-b-2 border-primary bg-primary/5" 
                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                )}
              >
                <tab.icon className={cn("h-4 w-4 mr-2", activeTab === tab.id ? "text-primary" : "text-muted-foreground")} />
                {tab.label}
              </button>
            ))}
          </div>

          <div className="p-6 md:p-8 flex-1">
            {activeTab === "visual" && (
              <div className="space-y-6 animate-in fade-in duration-300">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-muted-foreground uppercase tracking-wider font-semibold">Original DICOM</p>
                      <span className="text-xs bg-white/10 px-2 py-1 rounded text-foreground border border-white/5">AP View</span>
                    </div>
                    <div className="aspect-[4/3] bg-black rounded-lg border border-white/10 flex items-center justify-center overflow-hidden relative group shadow-lg">
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent z-10" />
                      <p className="absolute bottom-4 left-4 z-20 text-xs text-white/70 flex items-center font-mono">
                        <Activity className="w-3 h-3 mr-1 text-primary" /> 14:02:55
                      </p>
                      {/* Fake lung image placeholder - a dark gradient to simulate an X-Ray */}
                      <div className="w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-500 via-slate-800 to-[#050505] opacity-90 group-hover:scale-105 transition-transform duration-700" />
                      <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-10 mix-blend-overlay" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-primary uppercase tracking-wider font-semibold flex items-center group">
                        <AlertTriangle className="h-4 w-4 mr-1 group-hover:animate-bounce" /> Grad-CAM Heatmap
                      </p>
                      <span className="text-xs bg-primary/20 text-primary border border-primary/30 px-2 py-1 rounded shadow-[0_0_10px_rgba(0,229,255,0.2)]">Attention Map</span>
                    </div>
                    <div className="aspect-[4/3] bg-black rounded-lg border border-primary/30 flex items-center justify-center overflow-hidden relative shadow-[0_0_30px_rgba(0,229,255,0.15)] group">
                       <div className="w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-500 via-slate-800 to-[#050505] opacity-90 group-hover:scale-105 transition-transform duration-700" />
                       <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-10 mix-blend-overlay" />
                       {/* Fake heatmap spots representing Bilateral diffuse ground-glass opacities */}
                       <div className="absolute top-1/4 left-1/4 w-32 h-40 bg-red-500/40 rounded-[100%] blur-[25px] mix-blend-screen animate-pulse duration-[3000ms]" />
                       <div className="absolute top-1/3 right-1/4 w-40 h-48 bg-orange-500/30 rounded-[100%] blur-[35px] mix-blend-screen animate-pulse duration-[4000ms]" />
                       <div className="absolute top-1/2 left-1/3 w-20 h-20 bg-yellow-400/20 rounded-full blur-[20px] mix-blend-screen" />
                    </div>
                  </div>
                </div>
                <div className="bg-primary/5 border border-primary/20 rounded-lg p-5 text-sm text-primary flex items-start shadow-inner">
                  <CheckCircle2 className="h-5 w-5 mr-3 shrink-0 text-primary" />
                  <p className="leading-relaxed">Vision Encoder identifies highest activation in the perihilar regions bilaterally, consistent with the expected visual presentation of PCP.</p>
                </div>
              </div>
            )}

            {activeTab === "clinical" && (
              <div className="space-y-6 animate-in fade-in duration-300">
                <div className="p-5 rounded-xl bg-black/40 border border-white/5 border-l-4 border-l-primary relative overflow-hidden group hover:bg-black/60 transition-colors shadow-lg shadow-black/20">
                  <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                    <User className="h-16 w-16" />
                  </div>
                  <div className="flex justify-between items-start mb-3 relative z-10">
                    <h4 className="text-base font-semibold text-foreground flex items-center">
                      <FileText className="h-4 w-4 mr-2 text-primary" />
                      Condition snippet from FHIR Resource
                    </h4>
                    <span className="text-xs font-mono text-muted-foreground bg-white/5 border border-white/10 px-2 py-1 rounded shadow-sm">3 days ago</span>
                  </div>
                  <p className="text-sm text-foreground/80 font-mono bg-background/50 p-4 rounded-lg border border-white/5 leading-relaxed">
                    "Patient presented with progressive dyspnea and dry cough over the past week. Known history of HIV, recent CD4 count 180 cells/µL. Prescribed prophylactic trimethoprim-sulfamethoxazole."
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2 relative z-10">
                    <span className="text-xs px-3 py-1.5 rounded-full bg-primary/10 text-primary border border-primary/20 shadow-[0_0_10px_rgba(0,229,255,0.1)]">Immunocompromised</span>
                    <span className="text-xs px-3 py-1.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20 shadow-[0_0_10px_rgba(248,113,113,0.1)]">CD4 = 180 (&lt;200)</span>
                    <span className="text-xs px-3 py-1.5 rounded-full bg-white/5 text-muted-foreground border border-white/10">Dyspnea</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "literary" && (
              <div className="space-y-6 animate-in fade-in duration-300">
                <div className="p-5 rounded-xl bg-black/40 border border-white/5 hover:border-primary/30 transition-colors cursor-pointer group shadow-lg shadow-black/20 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="flex items-center gap-2 mb-3 text-xs text-primary font-medium tracking-wide relative z-10">
                    <CheckCircle2 className="h-4 w-4" /> PubMed: HIGH RELEVANCE SCORE <span className="text-white/80 bg-primary/20 px-1 rounded ml-1 font-mono">0.92</span>
                  </div>
                  <h4 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors relative z-10">
                    Radiographic manifestations of Pneumocystis jirovecii pneumonia in HIV patients
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4 font-mono relative z-10 flex items-center">
                    <BookOpen className="h-3 w-3 mr-1" /> Journal of Thoracic Imaging • 2021 • PMID: 33458291
                  </p>
                  <p className="text-sm text-foreground/80 leading-relaxed border-l-2 border-primary/50 pl-4 bg-primary/5 py-3 pr-3 rounded-r-lg relative z-10 shadow-inner">
                    Bilateral ground-glass opacities, which may be patchy or diffuse, are the hallmark of PCP on chest radiography and CT, occurring in up to 90% of cases...
                  </p>
                </div>
              </div>
            )}

            {activeTab === "audit" && (
              <div className="space-y-6 animate-in fade-in duration-300 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-px before:bg-gradient-to-b before:from-primary/50 before:via-white/10 before:to-transparent pt-4">
                
                {/* Timeline Item 1 */}
                <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full border-2 border-primary/50 bg-background text-primary shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-[0_0_15px_rgba(0,229,255,0.4)] z-10 ring-4 ring-background group-hover:scale-110 transition-transform">
                    <Activity className="h-5 w-5" />
                  </div>
                  <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] p-5 rounded-xl border border-primary/30 bg-primary/5 shadow-lg relative group-hover:border-primary/50 transition-colors">
                    {/* Directional Arrow */}
                    <div className="absolute top-5 -right-2 w-4 h-4 bg-primary/5 border-t border-r border-primary/30 rotate-45 hidden md:block group-hover:border-primary/50 transition-colors"></div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-bold text-primary text-base flex items-center font-heading">
                        Radiologist Agent
                      </div>
                      <time className="font-mono text-xs text-muted-foreground bg-black/40 border border-white/5 px-2 py-1 rounded shadow-inner">0ms</time>
                    </div>
                    <div className="text-sm text-foreground/80 leading-relaxed">
                      Generated Findings & Impression. Token entropy measured at <span className="text-green-400 font-mono bg-green-400/10 px-1 py-0.5 rounded ml-1 border border-green-400/20">0.12</span> (High Confidence).
                    </div>
                  </div>
                </div>

                {/* Timeline Item 2 */}
                <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active mt-8">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full border-2 border-purple-500/50 bg-background text-purple-400 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-[0_0_15px_rgba(168,85,247,0.3)] z-10 ring-4 ring-background group-hover:scale-110 transition-transform">
                    <Database className="h-5 w-5" />
                  </div>
                  <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] p-5 rounded-xl border border-purple-500/20 bg-purple-500/5 shadow-lg relative group-hover:border-purple-500/40 transition-colors">
                    {/* Directional Arrow */}
                    <div className="absolute top-5 -left-2 w-4 h-4 bg-purple-500/5 border-b border-l border-purple-500/20 rotate-45 hidden md:block md:group-odd:hidden md:group-even:block group-hover:border-purple-500/40 transition-colors"></div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-bold text-purple-400 text-base font-heading">Historian Agent</div>
                      <time className="font-mono text-xs text-muted-foreground bg-black/40 border border-white/5 px-2 py-1 rounded shadow-inner">+450ms</time>
                    </div>
                    <div className="text-sm text-foreground/80 leading-relaxed">
                      Queried FHIR. Found CD4 count = 180. Evidence strongly supports PCP hypothesis.
                    </div>
                  </div>
                </div>

                {/* Timeline Item 3 */}
                <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active mt-8">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full border-2 border-yellow-500/50 bg-background text-yellow-500 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-[0_0_15px_rgba(234,179,8,0.3)] z-10 ring-4 ring-background group-hover:scale-110 transition-transform">
                    <ShieldAlert className="h-5 w-5" />
                  </div>
                  <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] p-5 rounded-xl border border-yellow-500/30 bg-yellow-500/5 shadow-lg relative group-hover:border-yellow-500/50 transition-colors">
                    <div className="absolute top-5 -right-2 w-4 h-4 bg-yellow-500/5 border-t border-r border-yellow-500/30 rotate-45 hidden md:block group-hover:border-yellow-500/50 transition-colors"></div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-bold text-yellow-500 text-base flex items-center font-heading">
                        <AlertTriangle className="h-4 w-4 mr-1" /> Critic Head
                      </div>
                      <time className="font-mono text-xs text-muted-foreground bg-black/40 border border-white/5 px-2 py-1 rounded shadow-inner">+1200ms</time>
                    </div>
                    <div className="text-sm text-foreground/80 leading-relaxed">
                      Flagged CheXbert label 'Pleural Effusion' as <span className="text-yellow-500 font-mono bg-yellow-500/10 px-1 py-0.5 rounded ml-1 border border-yellow-500/20">Uncertain</span>. Increased overall uncertainty score by +0.15.
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
