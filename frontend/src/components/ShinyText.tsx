"use client";
import { cn } from "@/lib/utils";

interface ShinyTextProps {
  text: string;
  className?: string;
  shimmerWidth?: number;
  speed?: string;
}

export function ShinyText({ text, className = "", shimmerWidth = 100, speed = "3s" }: ShinyTextProps) {
  return (
    <span
      className={cn("inline-block bg-clip-text text-transparent", className)}
      style={{
        backgroundImage: `linear-gradient(
          120deg,
          rgba(255,255,255,0.4) 0%,
          rgba(255,255,255,1) 33%,
          rgba(255,255,255,0.4) 66%
        )`,
        backgroundSize: `${shimmerWidth}% 100%`,
        animation: `shiny-text ${speed} linear infinite`,
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
      }}
    >
      {text}
      <style jsx>{`
        @keyframes shiny-text {
          0% {
            background-position: 100% 0;
          }
          100% {
            background-position: -100% 0;
          }
        }
      `}</style>
    </span>
  );
}
