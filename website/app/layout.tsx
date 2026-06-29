import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: {
    default: "Pushing vector search QPS & latency to the ceiling",
    template: "%s",
  },
  description:
    "A first-principles study of how far one CPU box can push in-memory top-k vector search — the binary funnel, and what turned out to be the best.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="text-body antialiased">
        <main className="mx-auto max-w-3xl px-6 py-16 sm:py-24">{children}</main>
      </body>
    </html>
  );
}
