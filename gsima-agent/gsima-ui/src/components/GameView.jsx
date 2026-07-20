import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

export default function GameView({ frameData }) {
  const containerRef = useRef(null);

  useGSAP(() => {
    // Macro animation: Cinematic fade in on mount and subtle pulse on border
    gsap.from(containerRef.current, {
      opacity: 0,
      y: 20,
      duration: 1,
      ease: "power3.out"
    });

    gsap.to(containerRef.current, {
      boxShadow: "0 0 20px 5px rgba(255, 255, 255, 0.4)",
      duration: 2,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut"
    });
  }, { scope: containerRef });

  return (
    <div
      ref={containerRef}
      className="relative flex-1 rounded-xl overflow-hidden border-2 border-white bg-black min-h-[400px] flex items-center justify-center shadow-[0_0_15px_rgba(255,255,255,0.3)]"
    >
      <div className="absolute top-4 left-4 bg-white text-black px-3 py-1 rounded-full text-sm font-bold shadow-lg z-10">
        Live Feed
      </div>

      {frameData ? (
        <div className="w-full max-w-2xl aspect-video bg-[#1a1a1a] border border-[#4a4a4a] rounded-lg overflow-hidden flex items-center justify-center p-2 shadow-2xl">
          <img
            src={`data:image/jpeg;base64,${frameData}`}
            alt="Live game frame"
            className="w-full h-full object-contain"
          />
        </div>
      ) : (
        <div className="text-slate-400 animate-pulse flex flex-col items-center">
          <svg className="w-12 h-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
          <span className="text-lg">Waiting for video stream...</span>
        </div>
      )}
    </div>
  );
}
