import type { Metadata, Viewport } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { Shell } from "@/components/Shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "PhishLens — URL Threat Analysis",
  description: "Explainable machine-learning phishing URL detection.",
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = { themeColor: "#080a0d" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="font-sans">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
