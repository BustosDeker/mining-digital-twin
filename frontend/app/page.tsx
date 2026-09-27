"use client";

import dynamic from "next/dynamic";
import { useCallback, useState } from "react";
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
import type { CreateSessionInput } from "@/lib/types";

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

  const { snapshot, connected } = useSimulationSocket(sessionId);

  const handleLaunch = useCallback(async (payload: CreateSessionInput) => {
    const session = await api.createSession(payload);
    setSessionId(session.session_id);
    setFollowedAgentId(null);
  }, []);

  const withSession = (fn: (sid: string) => Promise<unknown>) => async () => {
    if (!sessionId) return;
    await fn(sessionId);
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
        <>
          <ControlBar
            status={snapshot?.status ?? null}
            onStart={handleStart}
            onPause={handlePause}
            onStop={handleStop}
            onReset={handleReset}
            onStep={handleStep}
            onNewScenario={() => setLauncherOpen(true)}
            disabled={!sessionId}
            tunnelOpacity={tunnelOpacity}
            onTunnelOpacityChange={setTunnelOpacity}
            showLabels={showLabels}
            onShowLabelsChange={setShowLabels}
            performanceMode={performanceMode}
            onPerformanceModeChange={setPerformanceMode}
          />

          <div className="relative flex flex-1 overflow-hidden">
            <div className="relative flex-1">
              {snapshot ? (
                <MineGraphScene
                  nodes={snapshot.nodes}
                  edges={snapshot.edges}
                  agents={snapshot.agents}
                  tunnelOpacity={tunnelOpacity}
                  showLabels={showLabels}
                  performanceMode={performanceMode}
                  followedAgentId={followedAgentId}
                  onSelectAgent={setFollowedAgentId}
                  theme={theme}
                />
              ) : (
                <div className="flex h-full items-center justify-center text-[13px] text-steel">
                  {sessionId ? t.common.loading : t.controls.newScenario}
                </div>
              )}

              <ScenarioLauncherPanel
                open={launcherOpen}
                onClose={() => setLauncherOpen(false)}
                onLaunch={handleLaunch}
              />
            </div>

            <aside className="panel-scroll w-72 shrink-0 overflow-y-auto border-l border-hairline bg-panel">
              <StressPanel agents={snapshot?.agents ?? {}} step={snapshot?.step ?? 0} />
              <EventLogPanel hazards={snapshot?.active_hazards ?? []} />
              <LegendPanel />
            </aside>
          </div>
        </>
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
