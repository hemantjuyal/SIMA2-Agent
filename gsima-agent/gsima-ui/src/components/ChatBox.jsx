import React, { useState, useEffect, useRef } from 'react';
import { Virtuoso } from 'react-virtuoso';

export default function ChatBox({ messages, onSendInstruction }) {
  const [input, setInput] = useState('');
  const virtuosoRef = useRef(null);

  // Force scroll to bottom when messages change
  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => {
        virtuosoRef.current?.scrollToIndex({
          index: messages.length - 1,
          behavior: 'smooth',
          align: 'end'
        });
      }, 50);
    }
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim()) {
      onSendInstruction(input.trim());
      setInput('');
    }
  };

  const renderMessage = (index, msg) => {
    const isUser = msg.sender === 'User Command';

    if (isUser) {
      return (
        <div className="flex w-full mb-6 justify-start">
          <div className="flex flex-col items-start max-w-[95%]">
            <div className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-amber-400 bg-amber-950/60 border border-amber-700/50 px-2.5 py-0.5 rounded-md mb-1.5 ml-1">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
              Operator Instruction
            </div>
            <div className="rounded-2xl px-5 py-3.5 transition-all duration-300 bg-[#262626] text-white rounded-tl-none shadow-xl border border-slate-700/60">
              <div className="text-sm font-medium tracking-wide">
                {msg.text}
              </div>
            </div>
          </div>
        </div>
      );
    }

    // S1-SIMA Closed-Loop Agent Decision (System One + Action Execution Layer)
    if (msg.decision) {
      const decisionModelLabel = msg.decision.decision_model || 'System One Decision Model';
      const executionLayerLabel = msg.decision.execution_layer || 'Action Execution Layer';

      return (
        <div className="flex flex-col w-full mb-6 items-end space-y-3">
          {/* System One Decision Model (Jev) - Reasoning Block */}
          <div className="w-full max-w-[96%] text-left">
            <div className="flex items-center justify-between mb-1.5 px-1">
              <div className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-indigo-300 bg-indigo-950/70 border border-indigo-700/60 px-2.5 py-0.5 rounded-md">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse"></span>
                {decisionModelLabel}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">TypeSafe Jev</span>
            </div>
            <div className="rounded-xl px-4 py-3 bg-[#18181b]/90 text-slate-300 text-xs border border-indigo-900/40 shadow-inner">
              <div className="leading-relaxed whitespace-pre-wrap font-mono text-[11px] text-slate-200">
                {msg.decision.thought}
              </div>
            </div>
          </div>

          {/* Action Execution Layer - Deterministic Software Actuation */}
          <div className="flex flex-col items-end max-w-[96%]">
            <div className="flex items-center gap-1.5 mb-1.5 mr-1">
              <span className="text-[10px] text-slate-400 font-mono">Deterministic Software</span>
              <div className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-emerald-300 bg-emerald-950/70 border border-emerald-700/60 px-2.5 py-0.5 rounded-md">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                {executionLayerLabel}
              </div>
            </div>
            <div className="rounded-2xl px-5 py-3.5 transition-all duration-300 bg-gradient-to-r from-[#242424] to-[#2d2d2d] text-white rounded-tr-none shadow-xl border border-slate-700/60 flex flex-col gap-1.5 items-end">
              <div className="text-sm font-medium tracking-wide text-slate-100 text-right">
                {msg.decision.narrative || (
                  <span>Executing tactical action</span>
                )}
              </div>
              {msg.decision.action && (
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Actuator Command:</span>
                  <span className="px-2.5 py-0.5 text-xs font-mono font-bold bg-sky-950/90 text-sky-300 border border-sky-600/60 rounded shadow-sm">
                    {msg.decision.action}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    // Fallback for System / Error messages
    const isError = msg.sender.includes('Error');
    return (
      <div className="flex w-full mb-6 justify-end">
        <div className="flex flex-col items-end max-w-[95%]">
          <div className={`text-xs opacity-60 mb-1 font-bold uppercase tracking-wider ${isError ? 'text-red-400' : 'text-slate-300'} mr-2`}>
            {msg.sender}
          </div>
          <div className={`rounded-3xl px-5 py-4 transition-all duration-300 bg-[#2a2a2a] text-white rounded-tr-none shadow-xl border ${isError ? 'border-red-900/50 bg-[#3a2a2a]' : 'border-slate-700/50'}`}>
            <div className="text-base font-medium tracking-wide">
              {msg.text}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-[#1e1e1e] rounded-xl border-2 border-white shadow-[0_0_15px_rgba(255,255,255,0.1)] overflow-hidden">
      <div className="bg-[#181818] px-4 py-3 border-b border-white flex justify-between items-center">
        <div>
          <h2 className="text-white font-bold tracking-wide text-sm flex items-center gap-2">
            S1-SIMA Closed-Loop Agent
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              System One + Deterministic Software
            </span>
          </h2>
          <p className="text-[11px] text-slate-400 font-medium">TypeSafe Jev Strategic Gating · Action Execution Layer</p>
        </div>
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-hidden bg-gradient-to-b from-[#1a1a1a] to-[#222222]">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-400 text-sm font-medium tracking-wide gap-2">
            <div className="w-10 h-10 rounded-full border border-slate-700 flex items-center justify-center text-slate-400 font-mono text-xs bg-slate-900 shadow">
              S1
            </div>
            <span>Agent initialized and standing by for instructions...</span>
          </div>
        ) : (
          <Virtuoso
            ref={virtuosoRef}
            style={{ height: '100%', width: '100%' }}
            data={messages}
            itemContent={renderMessage}
            followOutput="smooth"
            components={{ Footer: () => <div className="h-12" /> }}
          />
        )}
      </div>

      <div className="p-4 bg-[#1a1a1a] border-t border-white">
        <form onSubmit={handleSubmit} className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Command the agent..."
            className="w-full bg-[#2a2a2a] text-white placeholder-slate-400 rounded-full py-3 px-5 pr-12 focus:outline-none focus:ring-2 focus:ring-white transition-all border border-[#4a4a4a]"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="absolute right-2 top-2 bottom-2 bg-[#4a4a4a] text-white rounded-full p-2 hover:bg-[#5a5a5a] disabled:opacity-50 transition-colors"
          >
            <svg className="w-4 h-4 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19V6m0 0l-7 7m7-7l7 7" /></svg>
          </button>
        </form>
      </div>
    </div>
  );
}
