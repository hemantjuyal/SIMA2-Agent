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
            <div className="text-xs opacity-60 mb-1 font-bold uppercase tracking-wider text-slate-300 ml-2">
              {msg.sender}
            </div>
            <div className="rounded-3xl px-5 py-4 transition-all duration-300 bg-[#2a2a2a] text-white rounded-tl-none shadow-xl border border-slate-700/50">
              <div className="text-base font-medium tracking-wide">
                {msg.text}
              </div>
            </div>
          </div>
        </div>
      );
    }

    // SIMA 2 Agent Message (with decision object)
    if (msg.decision) {
      return (
        <div className="flex flex-col w-full mb-6 items-end">
          {/* Reasoning block (raw text, no bubble) */}
          <div className="text-slate-400 text-sm mb-4 max-w-[95%] text-left px-2">
            <div className="font-semibold text-slate-300 mb-1">Reasoning:</div>
            <div className="leading-relaxed whitespace-pre-wrap opacity-90">
              {/* {msg.decision.perception && `Perception: ${msg.decision.perception}\n`} */}
              {msg.decision.thought}
            </div>
          </div>

          {/* Action bubble */}
          <div className="flex flex-col items-end max-w-[95%]">
            <div className="text-xs opacity-60 mb-1 font-bold uppercase tracking-wider text-slate-300 mr-2">
              SIMA 2
            </div>
            <div className="rounded-3xl px-5 py-4 transition-all duration-300 bg-[#2a2a2a] text-white rounded-tr-none shadow-xl border border-slate-700/50">
              <div className="text-base font-medium tracking-wide">
                {msg.decision.narrative ? (
                  msg.decision.narrative
                ) : (
                  <>Action: <span className="text-sky-400 font-bold ml-1">{msg.decision.action}</span></>
                )}
              </div>
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
      <div className="bg-[#1a1a1a] px-4 py-3 border-b border-white flex justify-between items-center">
        <h2 className="text-slate-300 font-semibold tracking-wide">SIMA2 Agent</h2>
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-hidden bg-gradient-to-b from-[#1a1a1a] to-[#222222]">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-400 text-sm font-medium tracking-wide">
            Agent is waiting for instructions...
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
