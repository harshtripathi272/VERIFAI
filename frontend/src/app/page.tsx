import Link from "next/link";
import { ArrowRight, Cpu, ShieldCheck, FileSearch, Stethoscope, BookOpen, Brain } from "lucide-react";
import { GradientText } from "@/components/GradientText";
import { ShinyText } from "@/components/ShinyText";
import { AnimatedCard } from "@/components/AnimatedCard";

const agents = [
  {
    icon: Stethoscope,
    name: "Radiologist",
    desc: "MedGemma 4B + LoRA. Generates findings, impressions, and attention maps from DICOM studies.",
    color: "#00E5FF",
  },
  {
    icon: Cpu,
    name: "CheXbert",
    desc: "BERT-based labeler extracting 14 structured pathology conditions from free-text reports.",
    color: "#64FFDA",
  },
  {
    icon: FileSearch,
    name: "Historian",
    desc: "Queries FHIR R4 patient records. Correlates imaging findings with clinical history and lab results.",
    color: "#7C4DFF",
  },
  {
    icon: BookOpen,
    name: "Literature",
    desc: "PubMed RAG retrieval. Finds supporting or contradicting evidence from biomedical literature.",
    color: "#FF6E40",
  },
  {
    icon: ShieldCheck,
    name: "Critic",
    desc: "Adversarial validator. Flags contradictions, adjusts uncertainty, and ensures diagnostic safety.",
    color: "#FFD740",
  },
  {
    icon: ShieldCheck,
    name: "Validator",
    desc: "Final arbiter. Reviews evidence density, debate consensus, and safety to finalize or defer diagnosis to human review.",
    color: "#E040FB",
  },
];

export default function Home() {
  return (
    <div className="relative z-10">
      {/* Hero */}
      <section className="flex flex-col items-center justify-center min-h-[90vh] px-6 text-center">
        <div className="animate-fadeInUp">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-8 rounded-full border border-[#00E5FF]/20 bg-[#00E5FF]/[0.04] text-[13px] text-[#00E5FF] tracking-wide">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00E5FF] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00E5FF]"></span>
            </span>
            Sequential Debate Architecture
          </div>
        </div>

        <h1 className="text-5xl sm:text-6xl md:text-8xl font-[var(--font-outfit)] font-bold tracking-tight mb-6 animate-fadeInUp-delay-1 leading-[0.9]">
          <span className="text-white/90">Verified</span>
          <br />
          <GradientText className="mt-2" colors={["#00E5FF", "#2979FF", "#64FFDA", "#00E5FF"]}>
            Clinical AI
          </GradientText>
        </h1>

        <p className="max-w-xl text-base md:text-lg text-white/40 mb-12 animate-fadeInUp-delay-2 leading-relaxed font-light">
          Multi-agent diagnostic system with auditable evidence packets, calibrated uncertainty, and real-time literature retrieval.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 animate-fadeInUp-delay-3">
          <Link
            href="/diagnose"
            className="group inline-flex items-center justify-center gap-2 px-8 py-3.5 text-sm font-medium text-black bg-[#00E5FF] rounded-xl hover:bg-[#00E5FF]/90 transition-all duration-300 glow-cyan-strong hover:scale-[1.02] active:scale-[0.98]"
          >
            Start Diagnosis
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
          <Link
            href="/results/demo-123"
            className="inline-flex items-center justify-center gap-2 px-8 py-3.5 text-sm font-medium text-white/60 bg-white/[0.03] border border-white/[0.06] rounded-xl hover:bg-white/[0.06] hover:text-white/80 transition-all duration-300"
          >
            View Demo Report
          </Link>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 animate-bounce opacity-30">
          <div className="w-px h-12 bg-gradient-to-b from-white/30 to-transparent mx-auto" />
        </div>
      </section>

      {/* Agent Grid */}
      <section className="max-w-6xl mx-auto px-6 pb-32">
        <div className="text-center mb-16">
          <p className="text-[13px] uppercase tracking-[0.2em] text-[#00E5FF]/60 mb-3">The Council</p>
          <h2 className="text-3xl md:text-4xl font-[var(--font-outfit)] font-semibold text-white/90">
            Six Specialized Agents
          </h2>
          <p className="text-white/30 mt-4 max-w-lg mx-auto text-sm leading-relaxed">
            Each agent is independently trained and operates within a structured debate pipeline to ensure diagnostic accuracy.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent, i) => (
            <AnimatedCard key={agent.name} glowColor={`${agent.color}20`}>
              <div className="flex items-start gap-4">
                <div
                  className="flex items-center justify-center w-10 h-10 rounded-xl shrink-0"
                  style={{ backgroundColor: `${agent.color}10`, border: `1px solid ${agent.color}15` }}
                >
                  <agent.icon className="h-5 w-5" style={{ color: agent.color }} />
                </div>
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold text-white/90 mb-1">{agent.name}</h3>
                  <p className="text-[13px] text-white/35 leading-relaxed">{agent.desc}</p>
                </div>
              </div>
            </AnimatedCard>
          ))}
        </div>
      </section>

      {/* Architecture Flow */}
      <section className="max-w-4xl mx-auto px-6 pb-32">
        <div className="text-center mb-16">
          <p className="text-[13px] uppercase tracking-[0.2em] text-[#00E5FF]/60 mb-3">Pipeline</p>
          <h2 className="text-3xl md:text-4xl font-[var(--font-outfit)] font-semibold text-white/90">
            Sequential Debate Flow
          </h2>
        </div>

        <div className="relative">
          {/* Vertical connector */}
          <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-[#00E5FF]/40 via-white/10 to-transparent" />

          {[
            { step: "01", label: "Diagnosis", detail: "Radiologist generates findings and impressions from imaging", color: "#00E5FF" },
            { step: "02", label: "Labeling", detail: "CheXbert extracts 14 structured pathology conditions", color: "#64FFDA" },
            { step: "03", label: "Evidence", detail: "Historian and Literature agents gather context in parallel", color: "#7C4DFF" },
            { step: "04", label: "Critique", detail: "Critic validates consistency, flags contradictions, and triggers Debate", color: "#FFD740" },
            { step: "05", label: "Debate", detail: "Multi-round consensus building between all agents", color: "#FF6E40" },
            { step: "06", label: "Verdict", detail: "Validator evaluates confidence, safety, and consensus to finalize or request Human Review.", color: "#E040FB" },
          ].map((item) => (
            <div key={item.step} className="flex items-start gap-6 mb-8 relative group">
              <div
                className="relative z-10 w-12 h-12 rounded-full border-2 flex items-center justify-center shrink-0 text-xs font-bold bg-[#050507] transition-all duration-300 group-hover:scale-110"
                style={{ borderColor: `${item.color}40`, color: item.color }}
              >
                {item.step}
                <div
                  className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-md -z-10"
                  style={{ backgroundColor: `${item.color}30` }}
                />
              </div>
              <div className="pt-2.5 min-w-0">
                <h4 className="text-sm font-semibold text-white/80 mb-1">{item.label}</h4>
                <p className="text-[13px] text-white/30">{item.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
