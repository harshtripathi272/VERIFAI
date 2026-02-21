"use client";
import { cn } from "@/lib/utils";

interface AnimatedCardProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}

export function AnimatedCard({ children, className = "", glowColor = "rgba(0,229,255,0.15)" }: AnimatedCardProps) {
  return (
    <div
      className={cn(
        "relative group rounded-2xl bg-[#0d0d10]/80 backdrop-blur-xl border border-white/[0.06] p-6 overflow-hidden transition-all duration-500 hover:border-white/[0.12] hover:-translate-y-1",
        className
      )}
    >
      {/* Gradient glow on hover */}
      <div
        className="absolute -inset-[1px] rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm -z-10"
        style={{ background: `radial-gradient(circle at 50% 50%, ${glowColor}, transparent 70%)` }}
      />
      {/* Inner content shine line */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      {children}
    </div>
  );
}
