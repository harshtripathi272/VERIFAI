import Link from "next/link";
import { ArrowRight, BrainCircuit, ShieldCheck, Database } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] p-8 text-center mt-[-4rem]">
      <div className="inline-flex items-center px-3 py-1 mb-8 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium">
        <span className="flex h-2 w-2 rounded-full bg-primary mr-2 animate-pulse"></span>
        System Online • Ready for Diagnostics
      </div>
      
      <h1 className="text-5xl md:text-7xl font-heading font-bold tracking-tight mb-6 mt-4">
        Verified <span className="text-primary text-glow">Clinical AI</span>
      </h1>
      
      <p className="max-w-2xl text-lg md:text-xl text-muted-foreground mb-10 leading-relaxed">
        A hierarchical multi-agent diagnostic system providing auditable evidence packets with visual proof, literature citations, and calibrated uncertainty quantification.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4 mb-20 z-10 relative">
        <Link href="/diagnose" className="inline-flex items-center justify-center px-8 py-4 text-base font-semibold text-background bg-primary rounded-lg hover:bg-primary/90 transition-all shadow-[0_0_20px_rgba(0,229,255,0.4)] hover:shadow-[0_0_30px_rgba(0,229,255,0.6)] hover:scale-105 active:scale-95">
          Start Diagnosis
          <ArrowRight className="ml-2 h-5 w-5" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full text-left relative z-10">
        <div className="glass p-6 rounded-xl border border-white/5 hover:border-primary/30 transition-all hover:-translate-y-1">
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
            <BrainCircuit className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-heading font-semibold mb-2 text-foreground">Sequential Debate</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">Multiple specialized agents gather evidence, critique, and build consensus before concluding.</p>
        </div>
        <div className="glass p-6 rounded-xl border border-white/5 hover:border-primary/30 transition-all hover:-translate-y-1">
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
            <Database className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-heading font-semibold mb-2 text-foreground">FHIR & Literature</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">Context-aware diagnosis grounded in live patient history and real-time PubMed retrieval.</p>
        </div>
        <div className="glass p-6 rounded-xl border border-white/5 hover:border-primary/30 transition-all hover:-translate-y-1">
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
            <ShieldCheck className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-heading font-semibold mb-2 text-foreground">Uncertainty Calibrated</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">Avoids the black-box and overconfidence problems with epistemic uncertainty modeling.</p>
        </div>
      </div>
    </div>
  );
}
