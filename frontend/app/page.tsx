"use client";

import dynamic from "next/dynamic";
import "@/components/scene3d/twin.css";
import { useCallback, useEffect, useRef, useState } from "react";
import { TopBar, type DashboardTab } from "@/components/dashboard/TopBar";
import { ControlBar } from "@/components/dashboard/ControlBar";
import { ScenarioLauncherPanel } from "@/components/dashboard/ScenarioLauncherPanel";
import { StressPanel } from "@/components/dashboard/StressPanel";
import { EventLogPanel, LegendPanel } from "@/components/dashboard/EventLogPanel";
import { MLResultsPanel } from "@/components/dashboard/MLResultsPanel";
import {
  SessionHistoryPanel,
  ReportsPanel,
} from "@/components/dashboard/SessionsAndReportsPanel";
import { useSimulationSocket } from "@/hooks/useSimulationSocket";
import { useI18n } from "@/contexts/I18nContext";
import { useTheme } from "@/contexts/ThemeContext";
import { api } from "@/lib/api";
import type { CreateSessionInput, SimulationSnapshot } from "@/lib/types";

// react-three-fiber usa WebGL/DOM: debe cargarse solo en cliente.
const MineGraphScene = dynamic(
  () => import("@/components/scene3d/MineGraphScene").then((m) => m.MineGraphScene),
  { ssr: false }
);

export default function DashboardPage() {
  const { t } = useI18n();
  const { theme } = useTheme();

  const [activeTab, setActiveTab] = useState<DashboardTab>("twin");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [launcherOpen, setLauncherOpen] = useState(false);
  const [followedAgentId, setFollowedAgentId] = useState<string | null>(null);
  const [tunnelOpacity, setTunnelOpacity] = useState(0.85);
  const [showLabels, setShowLabels] = useState(true);
  const [performanceMode, setPerformanceMode] = useState(false);
  const [panMode, setPanMode] = useState(false);

  const { snapshot: streamSnapshot, connected } = useSimulationSocket(sessionId);
  const [lastSnapshot, setLastSnapshot] = useState<SimulationSnapshot | null>(null);
  const activeSession = useRef(sessionId);
  activeSession.current = sessionId;
  const [controlError, setControlError] = useState<string | null>(null);
  // Pause/stop/reset do not broadcast. Read their authoritative result through
  // the existing REST API; the next snapshot for this session resumes live updates.
  useEffect(() => {
    if (streamSnapshot?.session_id === sessionId) setLastSnapshot(streamSnapshot);
  }, [streamSnapshot, sessionId]);
  const snapshot = lastSnapshot?.session_id === sessionId ? lastSnapshot : null;

  const cancelFollow = useCallback(() => setFollowedAgentId(null), []);

  const handleLaunch = useCallback(async (payload: CreateSessionInput) => {
    const session = await api.createSession(payload);
    setSessionId(session.session_id);
    setFollowedAgentId(null);
    setControlError(null);
  }, []);

  const withSession = (fn: (sid: string) => Promise<unknown>) => async () => {
    if (!sessionId) return;
    setControlError(null);
    try {
      await fn(sessionId);
      const refreshed = await api.getSessionState(sessionId);
      if (activeSession.current === sessionId) setLastSnapshot(refreshed);
    } catch (cause) {
      setControlError(cause instanceof Error ? cause.message : t.common.error);
    }
  };

  const handleStart = withSession(api.startSession);
  const handlePause = withSession(api.pauseSession);
  const handleStop = withSession(api.stopSession);
  const handleReset = withSession(api.resetSession);
  const handleStep = withSession(api.stepSession);

  const connectionLabel = sessionId
    ? connected
      ? t.status.connected
      : t.status.disconnected
    : null;

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        connectionLabel={connectionLabel}
        connected={connected}
      />

      {activeTab === "twin" && (
        <div className="twin-workspace flex min-h-0 flex-1 flex-col" data-theme={theme}>
          <ControlBar
            status={snapshot?.status ?? null}
            onStart={handleStart}
            onPause={handlePause}
            onStop={handleStop}
            onReset={handleReset}
            onStep={handleStep}
            onNewScenario={() => setLauncherOpen(true)}
            disabled={!sessionId}
            panMode={panMode}
            onPanModeChange={setPanMode}
            tunnelOpacity={tunnelOpacity}
            onTunnelOpacityChange={setTunnelOpacity}
            showLabels={showLabels}
            onShowLabelsChange={setShowLabels}
            performanceMode={performanceMode}
            onPerformanceModeChange={setPerformanceMode}
          />

          {controlError && <p role="alert" className="border-b border-hairline bg-panel px-4 py-2 text-xs text-red">{controlError}</p>}
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden md:flex-row">
            <div className="relative min-h-0 min-w-0 flex-1">
              {snapshot ? (
                <MineGraphScene
                  key={snapshot.session_id}
                  nodes={snapshot.nodes}
                  edges={snapshot.edges}
                  agents={snapshot.agents}
                  hazards={snapshot.active_hazards}
                  onCancelFollow={cancelFollow}
                  tunnelOpacity={tunnelOpacity}
                  panMode={panMode}
                  showLabels={showLabels}
                  performanceMode={performanceMode}
                  followedAgentId={followedAgentId}
                  onSelectAgent={setFollowedAgentId}
                  theme={theme}
                />
              ) : (
                <div className="twin-empty">
                  <span className="twin-eyebrow">{t.twin.operations}</span>
                  <div className="twin-empty-icon" aria-hidden="true">⌑</div>
                  <h1>{sessionId ? t.common.loading : t.twin.emptyTitle}</h1>
                  <p>{t.twin.emptyDescription}</p>
                  {!sessionId && <button onClick={() => setLauncherOpen(true)}>+ {t.controls.newScenario}</button>}
                </div>
              )}

              <ScenarioLauncherPanel
                open={launcherOpen}
                onClose={() => setLauncherOpen(false)}
                onLaunch={handleLaunch}
              />
            </div>

            <aside className="panel-scroll h-52 w-full shrink-0 overflow-y-auto border-t border-hairline bg-panel md:h-auto md:w-72 xl:w-80 md:border-l md:border-t-0">
              <StressPanel agents={snapshot?.agents ?? {}} step={snapshot?.step ?? 0} />
              <EventLogPanel hazards={snapshot?.active_hazards ?? []} />
              <LegendPanel />
            </aside>
          </div>
        </div>
      )}

      {activeTab === "ml" && (
        <div className="flex-1 overflow-hidden">
          <MLResultsPanel />
        </div>
      )}

      {activeTab === "sessions" && (
        <div className="flex-1 overflow-hidden">
          <SessionHistoryPanel />
        </div>
      )}

      {activeTab === "reports" && (
        <div className="flex-1 overflow-hidden">
          <ReportsPanel />
        </div>
      )}
    </div>
  );
}
