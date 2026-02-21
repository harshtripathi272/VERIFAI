"use client";
import { useState } from "react";
import { UploadCloud, FileType, Settings, Play } from "lucide-react";
import { useRouter } from "next/navigation";

export default function DiagnosePage() {
  const router = useRouter();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  
  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    setIsAnalyzing(true);
    // Simulate API call and redirect
    setTimeout(() => {
      router.push("/results/demo-123");
    }, 2000);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 md:p-12 mb-20 mt-4 relative z-10">
      <div className="mb-8">
        <h1 className="text-3xl md:text-4xl font-heading font-bold mb-2 text-foreground">New Diagnosis</h1>
        <p className="text-muted-foreground">Upload imaging studies and provide clinical context to initiate the sequential debate.</p>
      </div>

      <form onSubmit={handleAnalyze} className="space-y-8">
        {/* Upload Zone */}
        <div 
          className={`glass p-8 rounded-xl border-dashed relative overflow-hidden group transition-all cursor-pointer text-center ${dragActive ? 'border-primary bg-primary/5' : 'border-white/10 hover:border-primary/50'}`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => { e.preventDefault(); setDragActive(false); }}
        >
          <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
          <UploadCloud className={`h-12 w-12 mx-auto mb-4 transition-all duration-300 ${dragActive ? 'text-primary scale-110' : 'text-muted-foreground group-hover:text-primary group-hover:scale-110'}`} />
          <h3 className="text-lg font-medium text-foreground mb-1">Upload DICOM Study</h3>
          <p className="text-sm text-muted-foreground mb-4">Drag and drop files here, or click to browse.</p>
          <div className="inline-flex items-center px-4 py-2 rounded-md bg-background/50 border border-white/5 text-sm text-foreground">
            <FileType className="h-4 w-4 mr-2 text-primary" /> Supported: .dcm, .zip
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1">Patient ID (FHIR reference)</label>
              <input 
                type="text" 
                placeholder="e.g. MRN-74892" 
                className="w-full bg-background/50 border border-white/10 rounded-lg px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1">Study UID</label>
              <input 
                type="text" 
                placeholder="e.g. 1.2.840.113..." 
                className="w-full bg-background/50 border border-white/10 rounded-lg px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                required
              />
            </div>
          </div>
          
          <div className="space-y-4 h-full">
            <label className="block text-sm font-medium text-muted-foreground mb-1">Clinical Question</label>
            <textarea 
              placeholder="What specifically should the radiologist agent look for?" 
              className="w-full h-[124px] bg-background/50 border border-white/10 rounded-lg px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none"
              required
            />
          </div>
        </div>

        {/* Settings Toggle */}
        <div className="glass p-6 rounded-xl border border-white/5 flex items-start justify-between">
          <div>
            <div className="flex items-center mb-1">
              <Settings className="h-4 w-4 mr-2 text-primary" />
              <h4 className="text-sm font-medium text-foreground">Sequential Debate Flow</h4>
            </div>
            <p className="text-xs text-muted-foreground">Enable all agents (Radiologist, CheXbert, Historian, Literature, Critic) for maximum evidence rigor.</p>
          </div>
          <div className="ml-4 pt-1">
            <div className="w-12 h-6 bg-primary/20 rounded-full relative cursor-pointer border border-primary/30">
              <div className="absolute right-1 top-[1px] w-5 h-5 rounded-full bg-primary shadow-[0_0_10px_rgba(0,229,255,0.8)]" />
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-white/5">
          <button 
            type="submit" 
            disabled={isAnalyzing}
            className="w-full flex items-center justify-center px-8 py-4 text-base font-semibold text-background bg-primary rounded-lg hover:bg-primary/90 transition-all shadow-[0_0_20px_rgba(0,229,255,0.4)] hover:shadow-[0_0_30px_rgba(0,229,255,0.6)] disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? (
              <span className="flex items-center">
                <span className="animate-spin h-5 w-5 mr-3 border-2 border-background border-t-transparent rounded-full"></span>
                Initiating Multi-Agent Debate...
              </span>
            ) : (
              <span className="flex items-center">
                <Play className="h-5 w-5 mr-2" />
                Analyze Patient Study
              </span>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
