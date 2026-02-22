import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";
import dynamic from "next/dynamic";

const ParticleField = dynamic(() => import("@/components/ParticleField"), {
  ssr: false,
  loading: () => null,
});

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit", weight: ["300", "400", "500", "600", "700", "800"] });

export const metadata: Metadata = {
  title: "VERIFAI | Verified Evidence-Based Clinical AI",
  description: "Hierarchical Multi-Agent Diagnostic System with Sequential Debate Architecture. Auditable evidence packets for every diagnostic decision.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${outfit.variable} font-sans min-h-screen flex flex-col relative noise-overlay`}>
        <ParticleField />
        <Navbar />
        <main className="flex-1 relative z-10 mt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
