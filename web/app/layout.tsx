import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AA DFW Crew Risk | EPPS Data Challenge",
  description: "Weather-driven crew sequence risk scoring for American Airlines A→DFW→B routes",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-aa-navy antialiased">
        {children}
      </body>
    </html>
  );
}
