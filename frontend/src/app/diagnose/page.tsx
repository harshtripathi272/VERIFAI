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

  const [selectedImages, setSelectedImages] = useState<{ file: File, view: string }[]>([]);
  const [selectedFhirFile, setSelectedFhirFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("");
  const [errorMsgs, setErrorMsgs] = useState<string>("");

  const toggleAgent = (id: string) => {
    setAgents((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedImages.length === 0) {
      setErrorMsgs("Please upload at least one DICOM or Image file.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMsgs("");

    try {
      const files = selectedImages.map(img => img.file);
      const views = selectedImages.map(img => img.view);
      const resp = await uploadAndStartWorkflow(files, views, patientId, selectedFhirFile);
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
            : (selectedImages.length > 0 ? "border-[#00E5FF]/20 bg-[#00E5FF]/[0.01]" : "border-white/[0.06] hover:border-white/[0.12] bg-white/[0.01] hover:bg-white/[0.02]")
            }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              const newFiles = Array.from(e.dataTransfer.files).map(f => ({ file: f, view: "AP" }));
              setSelectedImages(prev => [...prev, ...newFiles]);
            }
          }}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            multiple
            accept="image/*,.dcm,.zip"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                const newFiles = Array.from(e.target.files).map(f => ({ file: f, view: "AP" }));
                setSelectedImages(prev => [...prev, ...newFiles]);
              }
            }}
          />
          {selectedImages.length === 0 && (
            <UploadCloud className={`h-10 w-10 mx-auto mb-4 transition-all duration-300 ${dragActive ? "text-[#00E5FF] scale-110" : "text-white/20 group-hover:text-white/40 group-hover:scale-105"}`} />
          )}
          {selectedImages.length > 0 ? (
            <div className="flex flex-col gap-3 w-full max-w-md mx-auto">
              {selectedImages.map((img, idx) => (
                <div key={idx} className="flex items-center justify-between bg-white/[0.03] p-3 rounded-xl border border-white/[0.06]" onClick={(e) => e.stopPropagation()}>
                  <span className="text-sm font-medium text-[#00E5FF]/90 truncate max-w-[200px]">{img.file.name}</span>
                  <div className="flex items-center gap-3">
                    <select
                      value={img.view}
                      onChange={(e) => {
                        const newImgs = [...selectedImages];
                        newImgs[idx].view = e.target.value;
                        setSelectedImages(newImgs);
                      }}
                      className="bg-black/50 text-white/90 text-xs px-2.5 py-1.5 rounded-lg border border-white/[0.08] outline-none hover:border-[#00E5FF]/40 transition-colors"
                    >
                      <option value="AP">AP</option>
                      <option value="PA">PA</option>
                      <option value="LATERAL">LATERAL</option>
                    </select>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        const newImgs = [...selectedImages];
                        newImgs.splice(idx, 1);
                        setSelectedImages(newImgs);
                      }}
                      className="text-white/30 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </div>
                </div>
              ))}
              <div
                className="text-xs text-[#00E5FF]/60 mt-2 font-medium hover:text-[#00E5FF] transition-colors py-2 border border-dashed border-[#00E5FF]/20 rounded-xl bg-[#00E5FF]/[0.02]"
                onClick={(e) => {
                  e.stopPropagation();
                  document.getElementById('file-upload')?.click();
                }}
              >
                + Add another image
              </div>
            </div>
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
            <label className="block text-[13px] font-medium text-white/40 mb-2">Patient ID (Optional)</label>
            <input
              type="text"
              placeholder="e.g. MRN-74892"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full bg-white/[0.02] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-white/90 placeholder:text-white/20 transition-all duration-200"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-white/40 mb-2">Current FHIR Report (Optional)</label>
            <label className="flex items-center justify-between w-full bg-white/[0.02] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-white/90 transition-all duration-200 cursor-pointer hover:border-white/[0.12] hover:bg-white/[0.03]">
              <span className="truncate mr-4 text-white/60">
                {selectedFhirFile ? selectedFhirFile.name : "Select FHIR JSON..."}
              </span>
              <UploadCloud className="h-4 w-4 text-white/40 shrink-0" />
              <input
                type="file"
                className="hidden"
                accept=".json"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedFhirFile(e.target.files[0]);
                  }
                }}
              />
            </label>
          </div>
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
