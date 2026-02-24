"use client";
import { useState, useEffect } from "react";
import { Activity, BarChart3, Shield, Clock, AlertTriangle, CheckCircle2, ArrowUpRight, RefreshCw, Loader2, ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { GradientText } from "@/components/GradientText";
import { getMetricsSummary, MetricsSummary } from "@/lib/api";

export default function ObservabilityPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetricsSummary();
      setMetrics(data);
      setLastRefresh(new Date());
    } catch (err: any) {
      setError(err.message || "Failed to fetch metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    // Auto-refresh every 15 seconds
    const interval = setInterval(fetchMetrics, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !metrics) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12 flex flex-col items-center justify-center min-h-[60vh]">
        <Loader2 className="w-10 h-10 text-[#00E5FF] animate-spin mb-4" />
        <p className="text-white/40 text-sm">Loading observability metrics...</p>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12 flex flex-col items-center justify-center min-h-[60vh]">
        <AlertTriangle className="w-10 h-10 text-yellow-500/60 mb-4" />
        <p className="text-white/60 mb-2">Unable to reach metrics endpoint</p>
        <p className="text-white/30 text-sm mb-4">Ensure the backend is running at your configured API URL.</p>
        <button onClick={fetchMetrics} className="text-[#00E5FF] text-sm hover:underline flex items-center gap-1">
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      </div>
    );
  }

  const m = metrics!;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 mb-20 relative z-10">
      {/* Header */}
      <div className="flex justify-between items-start mb-10">
        <div>
          <Link
            href="/diagnose"
            className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 text-sm text-white/50 hover:text-white transition-colors bg-white/5 hover:bg-white/10 rounded-lg border border-white/10"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to Workspace
          </Link>
          <div className="flex items-center gap-2 px-3 py-1 mb-4 rounded-full border border-[#E040FB]/20 bg-[#E040FB]/[0.04] text-[11px] text-[#E040FB] uppercase tracking-[0.15em] font-medium w-fit">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E040FB] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#E040FB]"></span>
            </span>
            Live Monitoring
          </div>
          <h1 className="text-3xl md:text-4xl font-[var(--font-outfit)] font-bold text-white/90 leading-tight">
            <GradientText colors={["#E040FB", "#00E5FF", "#64FFDA"]}>Observability Dashboard</GradientText>
          </h1>
          <p className="text-white/30 text-sm mt-2">Real-time production metrics — auto-refreshes every 15s</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-white/20 font-mono">
            Last: {lastRefresh.toLocaleTimeString()}
          </span>
          <button
            onClick={fetchMetrics}
            className="flex items-center gap-2 px-4 py-2 text-[13px] text-[#00E5FF] bg-[#00E5FF]/[0.06] border border-[#00E5FF]/15 rounded-lg hover:bg-[#00E5FF]/10 transition-all"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} /> Refresh
          </button>
        </div>
      </div>

      {/* System Health Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Total Workflows"
          value={m.system.total_workflows}
          icon={<Activity className="h-4 w-4 text-[#00E5FF]" />}
          color="#00E5FF"
        />
        <MetricCard
          label="Active Now"
          value={m.system.active_workflows}
          icon={<span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-400"></span>
          </span>}
          color="#22c55e"
        />
        <MetricCard
          label="Deferrals"
          value={m.system.deferrals}
          icon={<Shield className="h-4 w-4 text-yellow-400" />}
          color="#eab308"
        />
        <MetricCard
          label="Critical Findings"
          value={m.system.critical_findings}
          icon={<AlertTriangle className="h-4 w-4 text-red-400" />}
          color="#ef4444"
        />
      </div>

      {/* Diagnostic Quality Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        <HistogramCard
          title="Confidence Distribution"
          data={m.diagnostics.confidence}
          unit="%"
          multiplier={100}
          color="#22c55e"
        />
        <HistogramCard
          title="Uncertainty Distribution"
          data={m.diagnostics.uncertainty}
          unit="%"
          multiplier={100}
          color="#00E5FF"
        />
        <HistogramCard
          title="Safety Score"
          data={m.diagnostics.safety_score}
          unit="%"
          multiplier={100}
          color="#E040FB"
        />
      </div>

      {/* Performance & Agents Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-8">
        {/* Agent Performance */}
        <div className="rounded-2xl border border-white/[0.04] bg-white/[0.015] p-6">
          <h3 className="text-sm font-semibold text-white/60 mb-5 flex items-center gap-2">
            <Clock className="h-4 w-4 text-[#00E5FF]" />
            Agent Performance
          </h3>
          {Object.keys(m.agents.duration).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(m.agents.duration).map(([label, stats]) => (
                <div key={label} className="flex items-center gap-3">
                  <span className="w-28 text-[12px] text-white/30 font-mono truncate">{label.replace(/agent_name="([^"]+)"/, "$1")}</span>
                  <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[#00E5FF] to-[#64FFDA] rounded-full transition-all"
                      style={{ width: `${Math.min(100, (stats.mean / 30) * 100)}%` }}
                    />
                  </div>
                  <span className="text-[11px] text-white/40 font-mono w-16 text-right">
                    {stats.mean.toFixed(1)}s avg
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-white/20 text-sm italic">No agent data yet — run a diagnosis first.</p>
          )}
        </div>

        {/* Workflow Duration */}
        <div className="rounded-2xl border border-white/[0.04] bg-white/[0.015] p-6">
          <h3 className="text-sm font-semibold text-white/60 mb-5 flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-[#E040FB]" />
            Workflow Duration
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 text-center">
              <p className="text-[10px] uppercase tracking-[0.15em] text-white/20 mb-1">Mean</p>
              <p className="text-2xl font-bold text-[#E040FB] font-[var(--font-outfit)]">
                {m.system.workflow_duration.mean.toFixed(1)}<span className="text-sm text-white/20">s</span>
              </p>
            </div>
            <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 text-center">
              <p className="text-[10px] uppercase tracking-[0.15em] text-white/20 mb-1">p95</p>
              <p className="text-2xl font-bold text-[#00E5FF] font-[var(--font-outfit)]">
                {m.system.workflow_duration.p95.toFixed(1)}<span className="text-sm text-white/20">s</span>
              </p>
            </div>
            <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 text-center">
              <p className="text-[10px] uppercase tracking-[0.15em] text-white/20 mb-1">p99</p>
              <p className="text-2xl font-bold text-[#64FFDA] font-[var(--font-outfit)]">
                {m.system.workflow_duration.p99.toFixed(1)}<span className="text-sm text-white/20">s</span>
              </p>
            </div>
            <div className="rounded-xl border border-white/[0.04] bg-black/20 p-4 text-center">
              <p className="text-[10px] uppercase tracking-[0.15em] text-white/20 mb-1">Total</p>
              <p className="text-2xl font-bold text-white/50 font-[var(--font-outfit)]">
                {m.system.workflow_duration.count}
              </p>
            </div>
          </div>
          <div className="mt-5 flex items-center gap-2 text-[11px] text-white/20">
            <span>Debate Rounds Avg: {m.diagnostics.debate_rounds.mean.toFixed(1)}</span>
            <span>&bull;</span>
            <span>Total Cases: {m.diagnostics.debate_rounds.count}</span>
          </div>
        </div>
      </div>

      {/* Safety Breakdown */}
      <div className="rounded-2xl border border-white/[0.04] bg-white/[0.015] p-6">
        <h3 className="text-sm font-semibold text-white/60 mb-5 flex items-center gap-2">
          <Shield className="h-4 w-4 text-red-400" />
          Safety &amp; Error Breakdown
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <p className="text-[11px] uppercase tracking-[0.12em] text-white/25 mb-3">Safety Flags</p>
            {Object.keys(m.safety.flags).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(m.safety.flags).map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/[0.04] bg-black/20 text-[12px]">
                    <span className="text-white/40 font-mono">{label}</span>
                    <span className="text-red-400 font-bold">{count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-white/20 text-sm italic">No safety flags raised yet.</p>
            )}
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.12em] text-white/25 mb-3">Errors</p>
            {Object.keys(m.safety.errors).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(m.safety.errors).map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/[0.04] bg-black/20 text-[12px]">
                    <span className="text-white/40 font-mono">{label}</span>
                    <span className="text-yellow-400 font-bold">{count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-green-400/60 text-sm">
                <CheckCircle2 className="h-4 w-4" /> No errors recorded
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prometheus Link */}
      <div className="mt-8 text-center">
        <a
          href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/metrics`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[#00E5FF]/50 text-[11px] hover:text-[#00E5FF] transition-colors inline-flex items-center gap-1"
        >
          Raw Prometheus endpoint <ArrowUpRight className="h-3 w-3" />
        </a>
      </div>
    </div>
  );
}


// ─── SUB-COMPONENTS ───

function MetricCard({ label, value, icon, color }: { label: string; value: number; icon: React.ReactNode; color: string }) {
  return (
    <div className="rounded-2xl border border-white/[0.04] bg-white/[0.02] p-5 hover:border-white/[0.08] transition-colors group">
      <div className="flex items-center gap-2 mb-2 text-white/30">
        {icon}
        <span className="text-[10px] uppercase tracking-[0.15em]">{label}</span>
      </div>
      <p className="text-3xl font-bold font-[var(--font-outfit)]" style={{ color }}>{value}</p>
    </div>
  );
}

function HistogramCard({ title, data, unit, multiplier, color }: {
  title: string;
  data: { count: number; mean: number; p50?: number; p95?: number };
  unit: string;
  multiplier: number;
  color: string;
}) {
  return (
    <div className="rounded-2xl border border-white/[0.04] bg-white/[0.015] p-5 hover:border-white/[0.08] transition-colors">
      <p className="text-[11px] uppercase tracking-[0.12em] text-white/25 mb-3">{title}</p>
      <p className="text-3xl font-bold font-[var(--font-outfit)] mb-3" style={{ color }}>
        {(data.mean * multiplier).toFixed(0)}
        <span className="text-sm text-white/20 ml-0.5">{unit}</span>
      </p>
      <div className="flex gap-3 text-[10px] text-white/30">
        <span>n={data.count}</span>
        {data.p50 !== undefined && <span>p50={(data.p50 * multiplier).toFixed(0)}{unit}</span>}
        {data.p95 !== undefined && <span>p95={(data.p95 * multiplier).toFixed(0)}{unit}</span>}
      </div>
    </div>
  );
}
