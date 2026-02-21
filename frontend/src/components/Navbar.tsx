"use client";
import Link from "next/link";
import { Activity } from "lucide-react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 w-full z-50 glass border-b-0 border-white/5 bg-background/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center space-x-2">
            <Activity className="h-6 w-6 text-primary" />
            <span className="font-heading font-bold text-xl tracking-wide text-foreground">
              VERI<span className="text-primary">FAI</span>
            </span>
          </Link>
          <div className="hidden md:block">
            <div className="flex items-baseline space-x-8">
              <Link 
                href="/diagnose" 
                className={cn(
                  "transition-colors px-3 py-2 text-sm font-medium",
                  pathname === "/diagnose" ? "text-primary" : "text-muted-foreground hover:text-foreground"
                )}>
                New Diagnosis
              </Link>
              <Link 
                href="#" 
                className="text-muted-foreground hover:text-foreground transition-colors px-3 py-2 text-sm font-medium">
                Architecture
              </Link>
              <Link 
                href="#" 
                className="bg-primary/10 text-primary border border-primary/20 hover:bg-primary/20 px-4 py-2 rounded-md text-sm font-medium transition-all shadow-[0_0_10px_rgba(0,229,255,0.2)]">
                Connect PACS
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
