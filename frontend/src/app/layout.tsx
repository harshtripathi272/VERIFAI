import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "VERIFAI | Verified Evidence-Based Clinical AI",
  description: "Hierarchical Multi-Agent Diagnostic System with Sequential Debate Architecture",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${outfit.variable} font-sans antialiased text-foreground bg-background min-h-screen flex flex-col relative`}>
        <div className="fixed inset-0 z-[-1] bg-hero-glow animate-slow-pan opacity-50" />
        <Navbar />
        <main className="flex-1 mt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
