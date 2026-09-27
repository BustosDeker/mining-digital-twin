"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { SimulationSnapshot } from "@/lib/types";

const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_BASE_URL || "ws://localhost:8000";

interface UseSimulationSocketResult {
  snapshot: SimulationSnapshot | null;
  connected: boolean;
  error: string | null;
}

/**
 * Se suscribe al estado del gemelo digital en tiempo real de una sesión.
 * Reintenta la conexión automáticamente si el servidor la cierra mientras
 * la sesión sigue existiendo (p.ej. tras un reinicio del backend en dev).
 */
export function useSimulationSocket(
  sessionId: string | null
): UseSimulationSocketResult {
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback((sid: string) => {
    const ws = new WebSocket(`${WS_BASE_URL}/ws/simulations/${sid}`);
    socketRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as SimulationSnapshot;
        setSnapshot(data);
      } catch {
        // mensaje no-JSON inesperado: se ignora, no rompe la UI
      }
    };

    ws.onerror = () => {
      setError("Error de conexión con el gemelo digital");
    };

    ws.onclose = (event) => {
      setConnected(false);
      socketRef.current = null;
      // 4004 = sesión no encontrada (ver backend/api/websocket.py): no reintentar
      if (event.code !== 4004) {
        retryTimeoutRef.current = setTimeout(() => connect(sid), 1500);
      }
    };
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setSnapshot(null);
      setConnected(false);
      return;
    }

    connect(sessionId);

    return () => {
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return { snapshot, connected, error };
}
