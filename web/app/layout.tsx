import type { Metadata } from "next";
import { Geist, Geist_Mono, Newsreader } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin"],
  style: ["italic", "normal"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "A 1-bit vector search engine",
  description:
    "What I built optimizing in-memory top-k vector search from first principles — the binary funnel, and what turned out to be the best.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} ${newsreader.variable}`}>
      <body className="text-body antialiased">
        <main className="mx-auto max-w-3xl px-6 py-16 sm:py-24">{children}</main>
      </body>
    </html>
  );
}
