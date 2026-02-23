"use client";
import Link from "next/link";
import { Activity, Menu, X } from "lucide-react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useState } from "react";

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const links = [
    { href: "/diagnose", label: "New Diagnosis" },
    { href: "/observability", label: "Observability" },
    { href: "/results/demo-123", label: "Demo Results" },
  ];

  return (
    <nav className="fixed top-0 w-full z-50 glass-strong">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="relative">
              <Activity className="h-5 w-5 text-[#00E5FF] transition-transform group-hover:scale-110" />
              <div className="absolute inset-0 bg-[#00E5FF] blur-[10px] opacity-30 group-hover:opacity-50 transition-opacity" />
            </div>
            <span className="font-[var(--font-outfit)] font-semibold text-lg tracking-[0.08em] text-white/90">
              VERIF<span className="text-[#00E5FF]">AI</span>
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "px-4 py-2 text-[13px] font-medium rounded-lg transition-all duration-200",
                  pathname === link.href
                    ? "text-[#00E5FF] bg-[#00E5FF]/[0.08]"
                    : "text-white/50 hover:text-white/80 hover:bg-white/[0.03]"
                )}
              >
                {link.label}
              </Link>
            ))}
            <div className="w-px h-5 bg-white/[0.06] mx-2" />
            <Link
              href="/diagnose"
              className="ml-1 px-5 py-2 text-[13px] font-medium rounded-lg bg-[#00E5FF]/10 text-[#00E5FF] border border-[#00E5FF]/20 hover:bg-[#00E5FF]/15 hover:border-[#00E5FF]/30 transition-all duration-200 glow-cyan"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile Toggle */}
          <button onClick={() => setMobileOpen(!mobileOpen)} className="md:hidden text-white/60 hover:text-white p-2">
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="md:hidden glass-strong border-t border-white/[0.04] px-6 py-4 space-y-2">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-3 text-sm text-white/60 hover:text-white hover:bg-white/[0.03] rounded-lg transition-colors"
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
