import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { GoogleGenAI } from '@google/genai';
import { Eraser, PenTool, Move, Calculator, Trash2, Save, Palette, Activity, Loader2, Info, Bug, CameraOff, Sparkles } from 'lucide-react';

type Point = { x: number; y: number };
type Stroke = { points: Point[]; color: string; thickness: number };
type FloatingText = { text: string; x: number; y: number; color: string };
type Mode = 'DRAW' | 'MOVE' | 'ERASE' | 'NONE';

const COLORS = ['#FFFF00', '#00FF00', '#00FFFF', '#FF00FF', '#FF0000', '#FFFFFF'];

export default function AirWriter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [mode, setMode] = useState<Mode>('NONE');
  const [color, setColor] = useState(COLORS[0]);
  const [thickness, setThickness] = useState(5);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [fingerStates, setFingerStates] = useState({ thumb: false, index: false, middle: false, ring: false, pinky: false });
  const [isSolving, setIsSolving] = useState(false);
  const [solution, setSolution] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const strokesRef = useRef<Stroke[]>([]);
  const textsRef = useRef<FloatingText[]>([]);
  const currentStrokeRef = useRef<Stroke | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const requestRef = useRef<number>(0);

  // Refs for state accessed in the animation loop to prevent stale closures
  const colorRef = useRef(color);
  const thicknessRef = useRef(thickness);
  const showLandmarksRef = useRef(showLandmarks);

  useEffect(() => {
    colorRef.current = color;
    thicknessRef.current = thickness;
    showLandmarksRef.current = showLandmarks;
  }, [color, thickness, showLandmarks]);
  
  // Initialize MediaPipe
  useEffect(() => {
    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        handLandmarkerRef.current = handLandmarker;
        setIsLoaded(true);
        startCamera();
      } catch (error) {
        console.error("Error initializing MediaPipe:", error);
      }
    };
    initMediaPipe();
    
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (handLandmarkerRef.current) handLandmarkerRef.current.close();
    };
  }, []);

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 }, 
          facingMode: "user" 
        } 
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().then(() => {
          if (requestRef.current) cancelAnimationFrame(requestRef.current);
          predictWebcam();
        }).catch(e => console.error("Video play error:", e));
      };
      setCameraError(null);
    } catch (err: any) {
      console.error("Error accessing webcam:", err);
      if (err.name === 'NotAllowedError' || err.message === 'Permission denied') {
        setCameraError("Camera access was denied. If you are viewing this in an iframe or preview window, you may need to open the app in a new tab to grant permissions.");
      } else if (err.name === 'NotFoundError') {
        setCameraError("No camera found. Please connect a webcam.");
      } else {
        setCameraError(`Camera error: ${err.message || 'Unknown error'}`);
      }
    }
  };

  const predictWebcam = () => {
    try {
      if (!videoRef.current || !canvasRef.current || !drawingCanvasRef.current || !handLandmarkerRef.current) return;
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const drawingCanvas = drawingCanvasRef.current;
      const drawingCtx = drawingCanvas.getContext("2d");
      
      if (!ctx || !drawingCtx) return;

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          drawingCanvas.width = video.videoWidth;
          drawingCanvas.height = video.videoHeight;
        }

        let startTimeMs = performance.now();
        if (lastVideoTimeRef.current !== video.currentTime) {
          lastVideoTimeRef.current = video.currentTime;
          const results = handLandmarkerRef.current.detectForVideo(video, startTimeMs);
          
          ctx.save();
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          if (results && results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            
            if (showLandmarksRef.current) {
              const drawingUtils = new DrawingUtils(ctx);
              drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 2
              });
              drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 1, radius: 3 });
              
              // Draw landmark numbers for debug
              ctx.fillStyle = "#FFFFFF";
              ctx.font = "10px JetBrains Mono";
              landmarks.forEach((landmark, index) => {
                ctx.fillText(index.toString(), landmark.x * canvas.width + 5, landmark.y * canvas.height + 5);
              });
            }

            const wrist = landmarks[0];
            const thumbTip = landmarks[4];
            const thumbIp = landmarks[3];
            const indexTip = landmarks[8];
            const indexPip = landmarks[6];
            const middleTip = landmarks[12];
            const middlePip = landmarks[10];
            const ringTip = landmarks[16];
            const ringPip = landmarks[14];
            const pinkyTip = landmarks[20];
            const pinkyPip = landmarks[18];

            const getDist = (p1: any, p2: any) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

            const isThumbUp = getDist(thumbTip, wrist) > getDist(thumbIp, wrist);
            const isIndexUp = getDist(indexTip, wrist) > getDist(indexPip, wrist);
            const isMiddleUp = getDist(middleTip, wrist) > getDist(middlePip, wrist);
            const isRingUp = getDist(ringTip, wrist) > getDist(ringPip, wrist);
            const isPinkyUp = getDist(pinkyTip, wrist) > getDist(pinkyPip, wrist);

            setFingerStates(prev => {
              if (prev.thumb !== isThumbUp || prev.index !== isIndexUp || prev.middle !== isMiddleUp || prev.ring !== isRingUp || prev.pinky !== isPinkyUp) {
                return { thumb: isThumbUp, index: isIndexUp, middle: isMiddleUp, ring: isRingUp, pinky: isPinkyUp };
              }
              return prev;
            });

            let detectedMode: Mode = 'NONE';
            
            if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
              detectedMode = 'DRAW';
            } else if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
              detectedMode = 'MOVE';
            } else if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
              detectedMode = 'ERASE';
            }

            // Only update state if changed to avoid excessive re-renders
            setMode((prev) => {
              if (prev !== detectedMode) return detectedMode;
              return prev;
            });

            const px = indexTip.x * drawingCanvas.width;
            const py = indexTip.y * drawingCanvas.height;

            if (detectedMode === 'DRAW') {
              if (!currentStrokeRef.current) {
                // Start new stroke
                currentStrokeRef.current = { points: [], color: colorRef.current, thickness: thicknessRef.current };
                strokesRef.current.push(currentStrokeRef.current);
              }
              
              const pts = currentStrokeRef.current.points;
              if (pts.length === 0) {
                pts.push({ x: px, y: py });
              } else {
                const lastPt = pts[pts.length - 1];
                const dist = Math.hypot(px - lastPt.x, py - lastPt.y);
                if (dist > 2) {
                   const alpha = 0.6;
                   const smoothedX = alpha * px + (1 - alpha) * lastPt.x;
                   const smoothedY = alpha * py + (1 - alpha) * lastPt.y;
                   pts.push({ x: smoothedX, y: smoothedY });
                }
              }
            } else {
              currentStrokeRef.current = null;
            }

            if (detectedMode === 'ERASE') {
              strokesRef.current = [];
            }
            
            // Draw cursor
            ctx.beginPath();
            ctx.arc(px, py, thicknessRef.current, 0, 2 * Math.PI);
            ctx.fillStyle = detectedMode === 'DRAW' ? colorRef.current : (detectedMode === 'MOVE' ? '#00AFFF' : '#FF0000');
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();
          } else {
            setMode('NONE');
            currentStrokeRef.current = null;
          }
          ctx.restore();
          
          renderStrokes(drawingCtx, drawingCanvas.width, drawingCanvas.height);
        }
      }
    } catch (error) {
      console.error("Error in predictWebcam:", error);
    }
    
    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  // Update current stroke color/thickness when state changes
  useEffect(() => {
    if (currentStrokeRef.current) {
      currentStrokeRef.current.color = color;
      currentStrokeRef.current.thickness = thickness;
    }
  }, [color, thickness]);

  const renderStrokes = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.clearRect(0, 0, width, height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    strokesRef.current.forEach(stroke => {
      if (stroke.points.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.thickness;
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) {
        ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
      ctx.stroke();
    });

    textsRef.current.forEach(t => {
      ctx.save();
      // The canvas is flipped horizontally via CSS.
      // To make text readable, we must flip it horizontally in the canvas context before drawing.
      ctx.translate(t.x, t.y);
      ctx.scale(-1, 1);
      
      ctx.font = "bold 80px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = t.color;
      ctx.shadowColor = t.color;
      ctx.shadowBlur = 20;
      ctx.fillText(t.text, 0, 0);
      ctx.shadowBlur = 0;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.strokeText(t.text, 0, 0);
      
      ctx.restore();
    });
  };

  const clearCanvas = () => {
    strokesRef.current = [];
    textsRef.current = [];
    currentStrokeRef.current = null;
  };

  const saveDrawing = () => {
    if (!drawingCanvasRef.current) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = drawingCanvasRef.current.width;
    tempCanvas.height = drawingCanvasRef.current.height;
    const tCtx = tempCanvas.getContext('2d');
    if (!tCtx) return;
    
    tCtx.fillStyle = '#0f0f0f';
    tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    tCtx.translate(tempCanvas.width, 0);
    tCtx.scale(-1, 1);
    tCtx.drawImage(drawingCanvasRef.current, 0, 0);
    
    const link = document.createElement('a');
    link.download = 'air-writing.png';
    link.href = tempCanvas.toDataURL('image/png');
    link.click();
  };

  const askAI = async () => {
    if (!drawingCanvasRef.current) return;
    setIsSolving(true);
    setSolution(null);
    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = drawingCanvasRef.current.width;
      tempCanvas.height = drawingCanvasRef.current.height;
      const tCtx = tempCanvas.getContext('2d');
      if (!tCtx) return;
      
      tCtx.fillStyle = '#000000';
      tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
      
      tCtx.translate(tempCanvas.width, 0);
      tCtx.scale(-1, 1);
      tCtx.drawImage(drawingCanvasRef.current, 0, 0);
      
      const base64Image = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: {
          parts: [
            { inlineData: { data: base64Image, mimeType: 'image/jpeg' } },
            { text: 'Read the handwritten text or drawing in this image. If it is a greeting (like "hi", "hello"), reply with a friendly greeting. If it is a question, answer it. If it is a math equation, solve it. Keep your response very short (maximum 5 words) so it can be displayed as large text on a screen. If there is no text, say "Draw something!"' }
          ]
        }
      });
      
      const reply = response.text?.trim() || 'Could not read.';
      
      textsRef.current.push({
        text: reply,
        x: drawingCanvasRef.current.width / 2,
        y: drawingCanvasRef.current.height / 2 + (textsRef.current.length * 100),
        color: colorRef.current
      });
      
      setSolution(reply);
    } catch (err) {
      console.error(err);
      setSolution('Error contacting AI.');
    } finally {
      setIsSolving(false);
    }
  };

  return (
    <div className="relative w-full h-screen bg-[#050505] overflow-hidden font-sans text-white flex items-center justify-center">
      {cameraError && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-[#050505] text-white p-6 text-center">
          <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center mb-6 border border-red-500/20">
            <CameraOff className="w-10 h-10 text-red-400" />
          </div>
          <h2 className="text-2xl font-bold tracking-tight mb-3">Camera Access Required</h2>
          <p className="text-white/60 max-w-md mb-8 leading-relaxed">
            {cameraError}
          </p>
          <button 
            onClick={() => {
              window.open(window.location.href, '_blank');
            }}
            className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-colors font-medium mb-4"
          >
            Open in New Tab
          </button>
          <button 
            onClick={() => {
              setCameraError(null);
              startCamera();
            }}
            className="px-6 py-3 bg-transparent hover:bg-white/5 text-white/70 hover:text-white rounded-xl transition-colors font-medium text-sm"
          >
            Try Again Here
          </button>
        </div>
      )}

      {!isLoaded && !cameraError && (
        <div className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-[#050505] text-white">
          <Loader2 className="w-12 h-12 animate-spin text-yellow-400 mb-4" />
          <h2 className="text-xl font-medium tracking-tight">Initializing AI Models...</h2>
          <p className="text-white/50 text-sm mt-2">Loading MediaPipe Hand Tracking</p>
        </div>
      )}

      <div className="relative w-full max-w-6xl aspect-video rounded-2xl overflow-hidden shadow-2xl shadow-black/50 border border-white/10 bg-black/50">
        <video ref={videoRef} className="hidden" playsInline autoPlay muted={true} />
        
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover -scale-x-100 opacity-80" />
        <canvas ref={drawingCanvasRef} className="absolute inset-0 w-full h-full object-cover -scale-x-100 drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]" />
        
        {/* Top Toolbar */}
        <div className="absolute top-6 left-1/2 -translate-x-1/2 flex items-center gap-2 p-2 bg-black/60 backdrop-blur-xl rounded-2xl border border-white/10 shadow-xl">
          <div className="flex items-center gap-1 px-2 border-r border-white/10">
            {COLORS.map(c => (
              <button
                key={c}
                onClick={() => setColor(c)}
                className={`w-6 h-6 rounded-full border-2 transition-transform ${color === c ? 'scale-110 border-white' : 'border-transparent hover:scale-110'}`}
                style={{ backgroundColor: c }}
              />
            ))}
          </div>
          
          <div className="flex items-center gap-3 px-4 border-r border-white/10">
            <span className="text-xs text-white/50 font-mono uppercase tracking-wider">Thickness</span>
            <input 
              type="range" 
              min="2" max="20" 
              value={thickness} 
              onChange={(e) => setThickness(parseInt(e.target.value))}
              className="w-24 accent-yellow-400"
            />
          </div>

          <div className="flex items-center gap-1 px-2 border-r border-white/10">
            <button onClick={clearCanvas} className="p-2 hover:bg-white/10 rounded-xl transition-colors text-white/70 hover:text-white" title="Clear Canvas">
              <Trash2 className="w-5 h-5" />
            </button>
            <button onClick={saveDrawing} className="p-2 hover:bg-white/10 rounded-xl transition-colors text-white/70 hover:text-white" title="Save Drawing">
              <Save className="w-5 h-5" />
            </button>
            <button onClick={() => setShowLandmarks(!showLandmarks)} className={`p-2 rounded-xl transition-colors ${showLandmarks ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-white/70 hover:text-white'}`} title="Toggle Landmarks">
              <Activity className="w-5 h-5" />
            </button>
            <button onClick={() => setShowDebug(!showDebug)} className={`p-2 rounded-xl transition-colors ${showDebug ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-white/70 hover:text-white'}`} title="Toggle Debug Mode">
              <Bug className="w-5 h-5" />
            </button>
          </div>

          <div className="px-2">
            <button 
              onClick={askAI}
              disabled={isSolving}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 text-white rounded-xl transition-colors font-medium text-sm"
            >
              {isSolving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
              Ask AI
            </button>
          </div>
        </div>
        
        {/* Gesture Instructions Panel */}
        <div className="absolute bottom-6 left-6 p-5 bg-black/60 backdrop-blur-xl rounded-2xl border border-white/10 w-72 shadow-xl">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-white/50 mb-4 flex items-center gap-2">
            <Info className="w-4 h-4" /> Gestures
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10">
                  <PenTool className="w-4 h-4 text-yellow-400" />
                </div>
                <span className="text-sm font-medium">Index Finger</span>
              </div>
              <span className="text-xs font-mono text-white/50">WRITE</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10">
                  <Move className="w-4 h-4 text-blue-400" />
                </div>
                <span className="text-sm font-medium">Two Fingers</span>
              </div>
              <span className="text-xs font-mono text-white/50">MOVE</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10">
                  <Eraser className="w-4 h-4 text-red-400" />
                </div>
                <span className="text-sm font-medium">Open Palm</span>
              </div>
              <span className="text-xs font-mono text-white/50">ERASE</span>
            </div>
          </div>
        </div>
        
        {/* Status Indicator */}
        <div className="absolute top-6 right-6 flex flex-col items-end gap-2">
          <div className="px-4 py-2 bg-black/60 backdrop-blur-xl rounded-xl border border-white/10 shadow-xl flex items-center gap-3">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-widest text-white/50 font-mono">Current Mode</span>
              <span className={`text-sm font-bold tracking-wider ${
                mode === 'DRAW' ? 'text-yellow-400' : 
                mode === 'MOVE' ? 'text-blue-400' : 
                mode === 'ERASE' ? 'text-red-400' : 'text-white'
              }`}>
                {mode}
              </span>
            </div>
            <div className={`w-3 h-3 rounded-full ${
                mode === 'DRAW' ? 'bg-yellow-400 shadow-[0_0_10px_rgba(250,204,21,0.5)]' : 
                mode === 'MOVE' ? 'bg-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.5)]' : 
                mode === 'ERASE' ? 'bg-red-400 shadow-[0_0_10px_rgba(248,113,113,0.5)]' : 'bg-white/20'
              }`} 
            />
          </div>
          
          {/* Debug Panel */}
          {showDebug && (
            <div className="p-4 bg-black/80 backdrop-blur-xl rounded-xl border border-white/20 shadow-xl font-mono text-xs text-white/80 w-48 mt-2 animate-in fade-in slide-in-from-top-2">
              <h3 className="text-white font-bold mb-2 border-b border-white/20 pb-1 flex items-center gap-2">
                <Bug className="w-3 h-3" /> DEBUG MODE
              </h3>
              <div className="flex justify-between py-1"><span>Thumb:</span> <span className={fingerStates.thumb ? 'text-green-400' : 'text-red-400'}>{fingerStates.thumb ? 'UP' : 'DOWN'}</span></div>
              <div className="flex justify-between py-1"><span>Index:</span> <span className={fingerStates.index ? 'text-green-400' : 'text-red-400'}>{fingerStates.index ? 'UP' : 'DOWN'}</span></div>
              <div className="flex justify-between py-1"><span>Middle:</span> <span className={fingerStates.middle ? 'text-green-400' : 'text-red-400'}>{fingerStates.middle ? 'UP' : 'DOWN'}</span></div>
              <div className="flex justify-between py-1"><span>Ring:</span> <span className={fingerStates.ring ? 'text-green-400' : 'text-red-400'}>{fingerStates.ring ? 'UP' : 'DOWN'}</span></div>
              <div className="flex justify-between py-1"><span>Pinky:</span> <span className={fingerStates.pinky ? 'text-green-400' : 'text-red-400'}>{fingerStates.pinky ? 'UP' : 'DOWN'}</span></div>
              <div className="mt-2 pt-2 border-t border-white/20 flex justify-between font-bold">
                <span>Mode:</span> <span className="text-yellow-400">{mode}</span>
              </div>
            </div>
          )}
        </div>
        
        {/* Solution Overlay */}
        {solution && (
          <div className="absolute bottom-6 right-6 p-6 bg-indigo-950/90 backdrop-blur-xl rounded-2xl border border-indigo-500/30 max-w-sm shadow-2xl animate-in slide-in-from-bottom-4 fade-in duration-300">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-indigo-300 mb-2 flex items-center gap-2">
              <Calculator className="w-4 h-4" /> AI Solution
            </h3>
            <div className="text-lg font-medium text-white leading-relaxed">
              {solution}
            </div>
            <button 
              onClick={() => setSolution(null)}
              className="mt-4 text-xs text-indigo-300 hover:text-white transition-colors"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
