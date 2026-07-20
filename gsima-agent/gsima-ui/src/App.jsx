import { useState, useEffect, useRef } from 'react'
import GameView from './components/GameView'
import ChatBox from './components/ChatBox'
import { useRecorder } from './components/Recorder'

function App() {
  const [frameData, setFrameData] = useState(null);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('Disconnected');
  const [envName, setEnvName] = useState('Unknown Environment');
  const [defaultInstruction, setDefaultInstruction] = useState('Type a new instruction...');
  const wsRef = useRef(null);
  
  const { isRecording, startRecording, stopRecording } = useRecorder();

  useEffect(() => {
    // Connect to FastAPI WebSocket backend
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;

    ws.onopen = () => setStatus('Connected');
    ws.onclose = () => setStatus('Disconnected');
    ws.onerror = (error) => console.error('WebSocket Error:', error);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'init') {
          setEnvName(payload.env_name);
        } else if (payload.type === 'frame') {
          setFrameData(payload.data);
        } else if (payload.type === 'thought') {
          const decision = payload.data;
          
          setMessages(prev => [...prev, {
            sender: 'SIMA 2',
            decision: decision
          }]);
        } else if (payload.type === 'status') {
          console.log("Agent Status:", payload.data);
          if (payload.data === 'finished') {
             // Silently ignore to keep UI clean, as requested by user
          } else if (payload.data === 'stopped') {
             setMessages(prev => [...prev, { sender: 'System', text: 'SIMA 2 Agent execution manually halted.' }]);
          }
        } else if (payload.type === 'error') {
          setMessages(prev => [...prev, { sender: 'System Error', text: payload.data }]);
        }
      } catch (e) {
        console.error("Message parsing error:", e);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  // Listen for Escape key to stop recording
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isRecording) {
        stopRecording();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRecording, stopRecording]);

  const handleSendInstruction = (instruction) => {
    // Add user message to UI
    setMessages(prev => [...prev, { sender: 'User Command', text: instruction }]);
    
    // Send to backend
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'start',
        instruction: instruction
      }));
    }
  };

  const handleStop = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
  };

  return (
    <div className="w-full h-screen bg-slate-950 p-6 flex flex-col font-sans text-slate-200 overflow-hidden">
      
      {/* Header - Hidden during recording for Presentation Mode */}
      {!isRecording && (
        <header className="flex justify-between items-center mb-6 px-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            SIMA 2 Explorer
          </h1>
          <p className="text-sm text-slate-400 font-medium">Interactive Multimodal Agent Interface</p>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Backend</span>
            <div className={`px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2 ${
              status === 'Connected' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${status === 'Connected' ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              {status}
            </div>
          </div>
          
          <div className="flex items-center gap-3 border-l border-slate-700 pl-6">
            {isRecording && (
              <div className="flex items-center text-red-500 animate-pulse font-bold bg-black/50 px-3 py-1 rounded-full">
                <div className="w-2 h-2 rounded-full bg-red-500 mr-2"></div>
                REC
              </div>
            )}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`px-4 py-2 rounded-md font-semibold text-sm transition-colors ${
                isRecording 
                  ? 'bg-red-600 hover:bg-red-500 text-white'
                  : 'bg-slate-700 hover:bg-slate-600 text-white border border-slate-500'
              }`}
            >
              {isRecording ? 'Stop Recording' : 'Record UI'}
            </button>

            <button 
              onClick={handleStop}
              className="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-4 py-2 rounded-md font-semibold text-sm transition-colors text-white"
            >
              Halt Agent
            </button>
          </div>
        </div>
      </header>
      )}
        
      {/* Main Content Grid */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-8 min-h-0">
          {/* Left Column: Chat and Instructions */}
          <div className="col-span-2 h-[90%] overflow-hidden">
            <ChatBox 
              messages={messages} 
              onSendInstruction={handleSendInstruction} 
            />
          </div>
          
          {/* Right Column: Game View */}
          <div className="col-span-3 h-[90%] flex flex-col">
            <GameView frameData={frameData} />
            <div className="mt-4 flex justify-center">
              <div className="px-6 py-2 rounded-full border-2 border-white bg-[#2a2a2a] text-white font-bold tracking-wider uppercase text-sm shadow-[0_0_10px_rgba(255,255,255,0.2)]">
                {envName}
              </div>
            </div>
          </div>
        </div>
      </div>
  )
}

export default App
