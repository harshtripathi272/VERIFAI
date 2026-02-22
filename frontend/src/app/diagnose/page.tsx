"use client";
import { useState } from "react";
import { UploadCloud, FileType, ChevronDown, ChevronUp, Play, Stethoscope, Cpu, FileSearch, BookOpen, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { GradientText } from "@/components/GradientText";
import { uploadAndStartWorkflow } from "@/lib/api";

const agentToggles = [
  { id: "radiologist", label: "Radiologist", icon: Stethoscope, description: "MedGemma 4B vision analysis", defaultOn: true },
  { id: "chexbert", label: "CheXbert", icon: Cpu, description: "Structured pathology labels", defaultOn: true },
  { id: "historian", label: "Historian", icon: FileSearch, description: "FHIR patient history", defaultOn: true },
  { id: "literature", label: "Literature", icon: BookOpen, description: "PubMed evidence retrieval", defaultOn: true },
  { id: "critic", label: "Critic", icon: ShieldCheck, description: "Adversarial validation", defaultOn: true },
];

export default function DiagnosePage() {
  const router = useRouter();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [agents, setAgents] = useState(
    Object.fromEntries(agentToggles.map((a) => [a.id, a.defaultOn]))
  );

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("");
  const [errorMsgs, setErrorMsgs] = useState<string>("");

  const toggleAgent = (id: string) => {
    setAgents((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsgs("Please upload a DICOM or Image file.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMsgs("");

    try {
      const resp = await uploadAndStartWorkflow(selectedFile, patientId);
      router.push(`/results/${resp.session_id}`);
    } catch (err: any) {
      console.error(err);
      setErrorMsgs(err.message || "Failed to start workflow.");
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-12 mb-20 relative z-10">
      {/* Header */}
      <div className="mb-10 animate-fadeInUp">
        <p className="text-[13px] uppercase tracking-[0.2em] text-[#00E5FF]/60 mb-2">Diagnostic Input</p>
        <h1 className="text-3xl md:text-4xl font-[var(--font-outfit)] font-bold text-white/90">
          New <GradientText>Analysis</GradientText>
        </h1>
        <p className="text-white/30 mt-3 text-sm leading-relaxed">
          Upload a medical imaging study and provide clinical context to initiate the multi-agent diagnostic pipeline.
        </p>
      </div>

      <form onSubmit={handleAnalyze} className="space-y-6 animate-fadeInUp-delay-1">

        {errorMsgs && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-sm mb-6">
            {errorMsgs}
          </div>
        )}

        {/* Upload Zone */}
        <div
          className={`relative rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300 group ${dragActive
              ? "border-[#00E5FF]/50 bg-[#00E5FF]/[0.03]"
              : (selectedFile ? "border-green-500/50 bg-green-500/[0.03]" : "border-white/[0.06] hover:border-white/[0.12] bg-white/[0.01] hover:bg-white/[0.02]")
            }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              setSelectedFile(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept="image/*,.dcm,.zip"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setSelectedFile(e.target.files[0]);
              }
            }}
          />
          <UploadCloud className={`h-10 w-10 mx-auto mb-4 transition-all duration-300 ${dragActive ? "text-[#00E5FF] scale-110" : (selectedFile ? "text-green-400 scale-105" : "text-white/20 group-hover:text-white/40 group-hover:scale-105")
            }`} />
          {selectedFile ? (
            <p className="text-sm font-medium text-green-400 mb-1">
              {selectedFile.name}
            </p>
          ) : (
            <>
              <p className="text-sm font-medium text-white/60 mb-1">
                Drop DICOM files here or <span className="text-[#00E5FF] underline underline-offset-4 decoration-[#00E5FF]/30">browse</span>
              </p>
              <p className="text-xs text-white/25 flex items-center justify-center gap-1.5 mt-2">
                <FileType className="h-3.5 w-3.5" /> .dcm, .zip, .png, .jpg
              </p>
            </>
          )}
        </div>

        {/* Input Fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-[13px] font-medium text-white/40 mb-2">Patient ID</label>
            <input
              type="text"
              placeholder="e.g. MRN-74892"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full bg-white/[0.02] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-white/90 placeholder:text-white/20 transition-all duration-200"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-white/40 mb-2">Study UID</label>
            <input
              type="text"
              placeholder="e.g. 1.2.840.113..."
              required
              className="w-full bg-white/[0.02] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-white/90 placeholder:text-white/20 transition-all duration-200"
            />
          </div>
        </div>

        <div>
          <label className="block text-[13px] font-medium text-white/40 mb-2">Clinical Question</label>
          <textarea
            placeholder="What specifically should the diagnostic pipeline investigate?"
            required
            rows={3}
            className="w-full bg-white/[0.02] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-white/90 placeholder:text-white/20 resize-none transition-all duration-200"
          />
        </div>

        {/* Advanced Settings */}
        <div className="rounded-xl border border-white/[0.04] bg-white/[0.01] overflow-hidden">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-5 py-3.5 text-sm text-white/40 hover:text-white/60 transition-colors"
          >
            <span className="flex items-center gap-2">
              Agent Configuration
              <span className="text-[11px] text-[#00E5FF]/50 bg-[#00E5FF]/[0.06] px-2 py-0.5 rounded-full">
                {Object.values(agents).filter(Boolean).length}/5 active
              </span>
            </span>
            {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>

          {showAdvanced && (
            <div className="px-5 pb-5 space-y-2 border-t border-white/[0.04] pt-4">
              {agentToggles.map((agent) => (
                <div
                  key={agent.id}
                  onClick={() => toggleAgent(agent.id)}
                  className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all duration-200 ${agents[agent.id]
                      ? "bg-[#00E5FF]/[0.04] border border-[#00E5FF]/10"
                      : "bg-white/[0.01] border border-white/[0.03] opacity-50"
                    }`}
                >
                  <agent.icon className={`h-4 w-4 shrink-0 ${agents[agent.id] ? "text-[#00E5FF]" : "text-white/20"}`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white/80">{agent.label}</p>
                    <p className="text-[11px] text-white/25">{agent.description}</p>
                  </div>
                  <div className={`w-8 h-[18px] rounded-full relative transition-colors ${agents[agent.id] ? "bg-[#00E5FF]/30" : "bg-white/10"
                    }`}>
                    <div className={`absolute top-[2px] w-[14px] h-[14px] rounded-full transition-all duration-200 ${agents[agent.id] ? "right-[2px] bg-[#00E5FF] shadow-[0_0_8px_rgba(0,229,255,0.6)]" : "left-[2px] bg-white/30"
                      }`} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={isAnalyzing}
          className="w-full flex items-center justify-center gap-2.5 px-8 py-4 text-sm font-semibold text-black bg-[#00E5FF] rounded-xl hover:bg-[#00E5FF]/90 transition-all duration-300 glow-cyan-strong hover:scale-[1.01] active:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {isAnalyzing ? (
            <>
              <span className="animate-spin h-4 w-4 border-2 border-black/30 border-t-black rounded-full" />
              Initiating Multi-Agent Pipeline...
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Analyze Study
            </>
          )}
        </button>
      </form>
    </div>
  );
}
