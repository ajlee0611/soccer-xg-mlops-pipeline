import React, { useState, useEffect, useRef, useCallback } from "react";

const ATTACKING_HALF_MIN_X = 60.0;
const PITCH_MAX_X = 120.0;
const PITCH_MAX_Y = 80.0;

export default function SoccerXgPlayground() {
  // Shot State in StatsBomb Pitch Coordinates
  const [coords, setCoords] = useState({ x: 104.5, y: 40.0 });
  const [isDragging, setIsDragging] = useState(false);
  const [bodyPart, setBodyPart] = useState("Right Foot");
  const [playPattern, setPlayPattern] = useState("Regular Play");
  const [underPressure, setUnderPressure] = useState(false);
  const [firstTime, setFirstTime] = useState(false);

  // Prediction Response State
  const [prediction, setPrediction] = useState({
    xg: 0.1245,
    distance_yards: 15.5,
    distance_meters: 14.17,
    angle_degrees: 29.35,
  });
  const [loading, setLoading] = useState(false);

  // Backend Health & Cold-Start State
  const [serviceStatus, setServiceStatus] = useState("checking"); // 'checking' | 'active'
  const [pollCount, setPollCount] = useState(0);

  const pitchRef = useRef(null);
  const abortControllerRef = useRef(null);
  const debounceTimerRef = useRef(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Send request to FastAPI /predict endpoint with abort controller
  const fetchPrediction = useCallback(
    async (x, y) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            match_id: 1,
            minute: 75,
            player_name: "Selected Striker",
            location_x: parseFloat(x.toFixed(1)),
            location_y: parseFloat(y.toFixed(1)),
            body_part: bodyPart,
            play_pattern: playPattern,
            under_pressure: underPressure,
            first_time: firstTime,
          }),
          signal: controller.signal,
        });

        if (response.ok) {
          const data = await response.json();
          setPrediction(data);
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("Inference server offline or unreachable:", err);
        }
      } finally {
        setLoading(false);
      }
    },
    [API_BASE_URL, bodyPart, playPattern, underPressure, firstTime]
  );

  // Health probe: polls /health until the backend & model are ready
  useEffect(() => {
    let isMounted = true;
    let timerId = null;

    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);

        const res = await fetch(`${API_BASE_URL}/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (res.ok) {
          const data = await res.json();
          if (data.status === "healthy" && data.model_loaded) {
            if (isMounted) {
              setServiceStatus("active");
              fetchPrediction(coords.x, coords.y);
            }
            return;
          }
        }
      } catch {
        // Cold start still in progress
      }

      if (isMounted) {
        setPollCount((prev) => prev + 1);
        timerId = setTimeout(checkHealth, 3000);
      }
    };

    checkHealth();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [API_BASE_URL, fetchPrediction, coords.x, coords.y]);

  // Debounced inference dispatcher during drag
  const queueInference = useCallback(
    (newX, newY) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      debounceTimerRef.current = setTimeout(() => {
        fetchPrediction(newX, newY);
      }, 60);
    },
    [fetchPrediction]
  );

  // Convert client pointer position into StatsBomb pitch dimensions
  const updateCoordsFromPointer = useCallback((e) => {
    if (!pitchRef.current) return null;
    const rect = pitchRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));

    const mappedX = ATTACKING_HALF_MIN_X + normX * (PITCH_MAX_X - ATTACKING_HALF_MIN_X);
    const mappedY = normY * PITCH_MAX_Y;

    return { x: mappedX, y: mappedY };
  }, []);

  // Pointer event handlers for drag interactions
  const handlePointerDown = (e) => {
    e.preventDefault();
    e.target.setPointerCapture(e.pointerId);
    setIsDragging(true);

    const newCoords = updateCoordsFromPointer(e);
    if (newCoords) {
      setCoords(newCoords);
      queueInference(newCoords.x, newCoords.y);
    }
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    const newCoords = updateCoordsFromPointer(e);
    if (newCoords) {
      setCoords(newCoords);
      queueInference(newCoords.x, newCoords.y);
    }
  };

  const handlePointerUp = (e) => {
    if (isDragging) {
      setIsDragging(false);
      try {
        e.target.releasePointerCapture(e.pointerId);
      } catch {
        // Safe ignore if pointer capture was lost
      }
      fetchPrediction(coords.x, coords.y);
    }
  };

  // Re-fetch when categorical controls change (only if backend is awake)
  useEffect(() => {
    if (serviceStatus === "active") {
      fetchPrediction(coords.x, coords.y);
    }
  }, [bodyPart, playPattern, underPressure, firstTime, serviceStatus, fetchPrediction, coords.x, coords.y]);

  // Teardown timers on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) abortControllerRef.current.abort();
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  // Pitch element mappings
  const ballSvgX = ((coords.x - ATTACKING_HALF_MIN_X) / (PITCH_MAX_X - ATTACKING_HALF_MIN_X)) * 100;
  const ballSvgY = (coords.y / PITCH_MAX_Y) * 100;

  const postLeftY = (36.0 / PITCH_MAX_Y) * 100;
  const postRightY = (44.0 / PITCH_MAX_Y) * 100;
  const goalCenterY = (40.0 / PITCH_MAX_Y) * 100;

  // Semantic color mapping: High xG is green, low xG is red
  const getXgColor = (val) => {
    if (val >= 0.35) return "text-emerald-400";
    if (val >= 0.15) return "text-amber-400";
    return "text-rose-500";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 flex flex-col items-center">
      <header className="max-w-6xl w-full mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
          Soccer Expected Goals (xG) Engine
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Real-time inference microservice backed by XGBoost, StatsBomb telemetry, and ZenML. Drag or click anywhere on the pitch to simulate a shot.
        </p>
      </header>

      {/* Cold-Start Warning Banner */}
      {serviceStatus === "checking" && (
        <div className="max-w-6xl w-full mb-6 bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-center justify-between gap-4 animate-pulse">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-amber-400 animate-ping" />
            <div>
              <p className="text-sm font-semibold text-amber-200">
                Waking up inference container on Render...
              </p>
              <p className="text-xs text-amber-300/70 mt-0.5">
                Free-tier instances spin down during inactivity and take ~30–45s to boot. The simulator will unlock automatically once warm.
              </p>
            </div>
          </div>
          <div className="text-xs font-mono text-amber-400/80 shrink-0 bg-amber-950/60 px-2.5 py-1 rounded border border-amber-500/20">
            Ping attempt #{pollCount + 1}
          </div>
        </div>
      )}

      <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pitch Canvas (Left 2 Columns) */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-col">
          <div className="flex justify-between items-center mb-3 px-1 text-xs text-slate-400 font-mono">
            <span>ATTACKING HALF (60m ➔ 120m)</span>
            <span>GOAL LINE ➔</span>
          </div>

          <div
            ref={pitchRef}
            onPointerDown={serviceStatus === "active" ? handlePointerDown : undefined}
            onPointerMove={serviceStatus === "active" ? handlePointerMove : undefined}
            onPointerUp={serviceStatus === "active" ? handlePointerUp : undefined}
            onPointerCancel={serviceStatus === "active" ? handlePointerUp : undefined}
            className={`relative w-full aspect-[4/3] bg-emerald-900/60 border-2 border-emerald-500/40 rounded-lg overflow-hidden select-none touch-none transition-all duration-300 ${
              serviceStatus === "active"
                ? "cursor-grab active:cursor-grabbing opacity-100"
                : "cursor-not-allowed opacity-60"
            }`}
            style={{
              backgroundImage:
                "radial-gradient(ellipse at center, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.4) 100%)",
            }}
          >
            {/* SVG Pitch Markings, Trajectory & Cone */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {/* Penalty Box (102 to 120 x, 18 to 62 y) */}
              <rect
                x={((102 - 60) / 60) * 100}
                y={(18 / 80) * 100}
                width={(18 / 60) * 100}
                height={(44 / 80) * 100}
                fill="none"
                stroke="rgba(255,255,255,0.3)"
                strokeWidth="0.8"
              />
              {/* 6-Yard Box */}
              <rect
                x={((114 - 60) / 60) * 100}
                y={(30 / 80) * 100}
                width={(6 / 60) * 100}
                height={(20 / 80) * 100}
                fill="none"
                stroke="rgba(255,255,255,0.3)"
                strokeWidth="0.8"
              />
              {/* Goal Line & Goal Frame */}
              <line x1="100" y1="0" x2="100" y2="100" stroke="rgba(255,255,255,0.5)" strokeWidth="1" />
              <line x1="100" y1={postLeftY} x2="100" y2={postRightY} stroke="#ffffff" strokeWidth="3" />

              {/* Subtended Shot Cone */}
              <polygon
                points={`${ballSvgX},${ballSvgY} 100,${postLeftY} 100,${postRightY}`}
                fill="rgba(32, 32, 30, 0.56)"
                stroke="rgba(0, 0, 0, 0.65)"
                strokeWidth="0.85"
                strokeDasharray="2,2"
              />

              {/* Faint Trajectory Center Line to Middle of Goal */}
              <line
                x1={ballSvgX}
                y1={ballSvgY}
                x2="100"
                y2={goalCenterY}
                stroke={
                  prediction.xg >= 0.35
                    ? "rgba(52, 211, 153, 0.60)"
                    : prediction.xg >= 0.15
                    ? "rgba(251, 191, 36, 0.55)"
                    : "rgba(244, 63, 94, 0.50)"
                }
                strokeWidth="0.35"
                strokeDasharray="1.5 2.5"
              />
            </svg>

            {/* Ball Marker */}
            <div
              className={`absolute w-6 h-6 -ml-3 -mt-3 rounded-full bg-white shadow-lg shadow-amber-400/50 flex items-center justify-center pointer-events-none transition-transform duration-75 ${
                isDragging ? "scale-125" : "scale-100"
              }`}
              style={{ left: `${ballSvgX}%`, top: `${ballSvgY}%` }}
            >
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            </div>
          </div>

          <div className="mt-3 flex justify-between text-xs text-slate-400 px-1 font-mono">
            <span>
              Coordinates: X={coords.x.toFixed(1)}y, Y={coords.y.toFixed(1)}y
            </span>
            <span>
              {serviceStatus === "checking"
                ? "Connecting..."
                : loading
                ? "Calculating..."
                : "Ready"}
            </span>
          </div>
        </div>

        {/* Inference HUD & Controls (Right Column) */}
        <div className="flex flex-col gap-4">
          {/* Main xG Readout Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl text-center flex flex-col justify-center items-center">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-1">
              Calculated Probability
            </span>
            <div
              className={`text-6xl font-black tracking-tight font-mono my-2 ${getXgColor(
                prediction.xg
              )}`}
            >
              {prediction.xg.toFixed(3)}
            </div>
            <span className="text-xs text-slate-400">
              {prediction.xg >= 0.35
                ? "High Chance Opportunity"
                : prediction.xg >= 0.15
                ? "Moderate Chance"
                : "Difficult Angle / Low Probability"}
            </span>

            {/* Sub-Metrics Breakdown */}
            <div className="w-full grid grid-cols-2 gap-2 mt-5 pt-4 border-t border-slate-800 text-left">
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                <p className="text-[10px] text-slate-500 uppercase font-mono">Goal Distance</p>
                <p className="text-base font-bold text-slate-200 mt-0.5">{prediction.distance_yards} yds</p>
                <p className="text-[11px] text-slate-400">({prediction.distance_meters} m)</p>
              </div>
              <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                <p className="text-[10px] text-slate-500 uppercase font-mono">Post Angle</p>
                <p className="text-base font-bold text-slate-200 mt-0.5">{prediction.angle_degrees}°</p>
                <p className="text-[11px] text-slate-400">Subtended</p>
              </div>
            </div>
          </div>

          {/* Context Controls */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-3.5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Shot Parameters</h3>

            <div>
              <label className="text-xs text-slate-300 block mb-1">Body Part</label>
              <select
                value={bodyPart}
                onChange={(e) => setBodyPart(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="Right Foot">Right Foot</option>
                <option value="Left Foot">Left Foot</option>
                <option value="Head">Head</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-slate-300 block mb-1">Play Pattern</label>
              <select
                value={playPattern}
                onChange={(e) => setPlayPattern(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="Regular Play">Regular Play</option>
                <option value="From Counter">From Counter</option>
                <option value="From Free Kick">From Free Kick</option>
                <option value="From Corner">From Corner</option>
              </select>
            </div>

            <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
              <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 select-none">
                <input
                  type="checkbox"
                  checked={underPressure}
                  onChange={(e) => setUnderPressure(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0"
                />
                Defensive Pressure Present
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 select-none">
                <input
                  type="checkbox"
                  checked={firstTime}
                  onChange={(e) => setFirstTime(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0"
                />
                First-Time Shot (One Touch)
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}