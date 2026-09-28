"use client";

import { Component, ReactNode, useMemo, useState, useEffect, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import type { AgentSnapshot, HazardEvent, MineEdge, MineNode, MineViewMode } from "@/lib/types";
import { useI18n } from "@/contexts/I18nContext";
import { Tunnels } from "./Tunnels";
import { NodeMarkers } from "./NodeMarkers";
import { Agents } from "./Agents";
import { CameraRig } from "./CameraRig";
import { HazardEffects } from "./HazardEffects";
import { SceneLabels, type SceneLabel } from "./SceneLabels";
import { backendToThreePosition } from "./geometry";
import { SCENE_COLORS as C } from "./colors";

interface MineGraphSceneProps {
  nodes: MineNode[]; edges: MineEdge[]; agents: Record<string, AgentSnapshot>;
  hazards: HazardEvent[]; tunnelOpacity: number; panMode: boolean; showLabels: boolean;
  performanceMode: boolean; followedAgentId: string | null;
  onSelectAgent: (id: string) => void; onCancelFollow: () => void; theme: "dark" | "light";
}

class WebGLErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { hasError: boolean }> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() { return this.state.hasError ? this.props.fallback : this.props.children; }
}

type Selection = { kind: "node" | "edge" | "agent"; id: string } | null;

export function MineGraphScene(props: MineGraphSceneProps) {
  const { t } = useI18n();
  const { edges, agents, hazards, performanceMode, showLabels, followedAgentId, onCancelFollow } = props;
  const [view, setView] = useState<MineViewMode>("general");
  const [level, setLevel] = useState<number | null>(null);
  const [fitRequest, setFitRequest] = useState(0);
  const [selection, setSelection] = useState<Selection>(null);
  const [retry, setRetry] = useState(0);
  // Snapshots contain new arrays even when infrastructure is unchanged.
  const nodesKey = JSON.stringify(props.nodes);
  const nodes = useMemo(() => JSON.parse(nodesKey) as MineNode[], [nodesKey]);
  const levels = useMemo(() => [...new Set(nodes.map(n => n.level))].sort((a, b) => a - b), [nodes]);
  const visibleNodes = useMemo(() => nodes.filter(n => level === null || n.level === level), [nodes, level]);
  const ids = useMemo(() => new Set(visibleNodes.map(n => n.node_id)), [visibleNodes]);
  const visibleEdges = useMemo(() => edges.filter(e => ids.has(e.source) || ids.has(e.target)), [edges, ids]);
  const visibleHazards = useMemo(() => hazards.filter(h => ids.has(h.origin_node_id)), [hazards, ids]);
  const visibleAgents = useMemo(() => Object.fromEntries(Object.entries(agents).filter(([, a]) => ids.has(a.node_id) || ids.has(a.next_node_id))), [agents, ids]);
  const affectedEdges = useMemo(() => new Set(hazards.flatMap(h => h.affected_edges)), [hazards]);
  const occupiedEdges = useMemo(() => new Set(Object.values(agents).filter(a => a.status !== "evacuated" && a.edge_id).map(a => a.edge_id!)), [agents]);
  const incidentNodes = useMemo(() => visibleNodes.filter(n => visibleHazards.some(h => h.origin_node_id === n.node_id)), [visibleNodes, visibleHazards]);
  const focusNodes = view === "incidents" && incidentNodes.length ? incidentNodes : visibleNodes;
  const onSelectEdge = useCallback((id: string) => setSelection({ kind: "edge", id }), []);
  const onSelectNode = useCallback((id: string) => setSelection({ kind: "node", id }), []);
  const onSelectWorker = useCallback((id: string) => setSelection({ kind: "agent", id }), []);
  useEffect(() => {
    if (followedAgentId && (!agents[followedAgentId] || agents[followedAgentId].status === "evacuated")) onCancelFollow();
  }, [agents, followedAgentId, onCancelFollow]);

  const hazardName = (h: HazardEvent) => h.hazard_type === "fire" ? t.scenario.hazardFire : h.hazard_type === "collapse" ? t.scenario.hazardCollapse : t.scenario.hazardGasLeak;
  const nodeName = (n: MineNode) => n.node_type === "exit" ? t.legend.exit : n.node_type === "refuge_chamber" ? t.legend.refuge : n.node_type === "risk_zone" ? t.legend.riskZone : t.scenario.galleries;
  const labels: SceneLabel[] = showLabels ? visibleNodes.filter(n => !["gallery", "intersection"].includes(n.node_type)).map(n => {
    const p = backendToThreePosition(n.position);
    return { id: n.node_id, position: [p[0], p[1] + 0.87, p[2]], text: `${nodeName(n)} ${n.node_id}`, color: n.node_type === "exit" ? C.exit : n.node_type === "refuge_chamber" ? C.refuge : C.riskZone, priority: n.node_type === "exit" ? 3 : 2 };
  }) : [];
  if (showLabels) visibleHazards.forEach(h => {
    const node = nodes.find(n => n.node_id === h.origin_node_id);
    if (!node) return;
    const p = backendToThreePosition(node.position);
    labels.push({ id: h.event_id, position: [p[0], p[1] + 1.15, p[2]], text: `${hazardName(h)} · ${Math.round(h.intensity * 100)}%`, color: C.blocked, priority: 5 });
  });
  const selectedNode = selection?.kind === "node" ? nodes.find(n => n.node_id === selection.id) : undefined;
  const selectedEdge = selection?.kind === "edge" ? edges.find(e => e.edge_id === selection.id) : undefined;
  const selectedAgent = selection?.kind === "agent" ? agents[selection.id] : undefined;
  const agentStatus = (a: AgentSnapshot) => a.status === "moving" ? t.stressPanel.moving : a.status === "waiting" ? t.twin.waiting : a.status === "sheltered" ? t.stressPanel.sheltered : a.status === "evacuated" ? t.stressPanel.evacuated : t.stressPanel.lost;
  const background = props.theme === "dark" ? C.background : C.backgroundLight;
  const bounds = useMemo(() => {
    const p = nodes.map(n => backendToThreePosition(n.position));
    const minX = Math.min(...p.map(v => v[0])), maxX = Math.max(...p.map(v => v[0]));
    const minZ = Math.min(...p.map(v => v[2])), maxZ = Math.max(...p.map(v => v[2]));
    return { x: (minX + maxX) / 2, z: (minZ + maxZ) / 2, y: Math.min(...p.map(v => v[1])) - 0.8,
      span: Math.max(maxX - minX, maxZ - minZ, 4) * 1.25 };
  }, [nodes]);

  return <div className="twin-scene flex h-full min-h-0 flex-col">
    <div className="twin-viewbar" role="toolbar" aria-label={t.mapView.toolbar}>
      {(["general", "top", "lateral", "incidents", "routes"] as MineViewMode[]).map(mode => <button key={mode} aria-pressed={view === mode}
        title={mode === "routes" ? t.twin.routesHint : t.mapView[mode]}
        onClick={() => { onCancelFollow(); setView(mode); setFitRequest(v => v + 1); }}>{t.mapView[mode]}</button>)}
      <span className="twin-divider" />
      <label>{t.mapView.level} <select value={level ?? "all"} onChange={e => { onCancelFollow(); setLevel(e.target.value === "all" ? null : Number(e.target.value)); }}>
        <option value="all">{t.mapView.allLevels}</option>{levels.map(l => <option key={l} value={l}>{l}</option>)}
      </select></label>
      <button className="ml-auto" onClick={() => { onCancelFollow(); setView("general"); setLevel(null); setFitRequest(v => v + 1); } }>⊞ {t.twin.fit}</button>
    </div>
    <div className="relative min-h-0 flex-1">
      <WebGLErrorBoundary key={retry} fallback={<div className="twin-empty"><h3>{t.twin.webglError}</h3><p>{t.twin.webglHelp}</p><button onClick={() => setRetry(v => v + 1)}>{t.twin.retry}</button></div>}>
        <Canvas camera={{ position: [12, 12, 12], fov: 42, near: 0.03, far: 1000 }} dpr={performanceMode ? 1 : [1, 1.6]} shadows={false}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}>
          <color attach="background" args={[background]} />
          <fog attach="fog" args={[background, bounds.span * 2.5, bounds.span * 6]} />
          <hemisphereLight args={["#dce8ed", "#343027", 1.4]} />
          <directionalLight position={[10, 18, 6]} color="#fff0d1" intensity={2.2} />
          {!performanceMode && <directionalLight position={[-10, 3, -8]} color="#a8cde0" intensity={0.7} />}
          <gridHelper args={[bounds.span, 24, props.theme === "dark" ? "#293d48" : "#afbcc3", props.theme === "dark" ? "#1c2b35" : "#c4cfd4"]} position={[bounds.x, bounds.y, bounds.z]} />
          <Tunnels nodes={nodes} edges={visibleEdges} opacity={props.tunnelOpacity} performanceMode={performanceMode}
            affectedEdges={affectedEdges} occupiedEdges={occupiedEdges} routes={view === "routes"} selectedEdge={selectedEdge?.edge_id ?? null} onSelectEdge={onSelectEdge} />
          <NodeMarkers nodes={visibleNodes} edges={edges} onSelectNode={onSelectNode} />
          <HazardEffects nodes={nodes} hazards={visibleHazards} performanceMode={performanceMode} />
          <Agents nodes={nodes} agents={visibleAgents} followedAgentId={followedAgentId} onSelectAgent={onSelectWorker} performanceMode={performanceMode} />
          <SceneLabels labels={labels} />
          <CameraRig nodes={nodes} agents={agents} followedAgentId={followedAgentId} panMode={props.panMode}
            view={view} fitRequest={fitRequest} focusNodes={focusNodes} onCancelFollow={onCancelFollow} />
        </Canvas>
      </WebGLErrorBoundary>
      <div className="twin-map-caption"><span className="twin-eyebrow">{t.twin.infrastructure}</span><strong>{t.twin.cutaway}</strong></div>
      <div className="twin-levels">{levels.map(l => {
        const elevations = nodes.filter(n => n.level === l).map(n => n.position[2]);
        const min = Math.min(...elevations), max = Math.max(...elevations);
        return <button key={l} aria-pressed={level === l} onClick={() => { onCancelFollow(); setLevel(level === l ? null : l); }}>
          <span>{t.mapView.level} {l}</span><small>{min === max ? min : `${min}…${max}`} m</small></button>;
      })}</div>
      {followedAgentId && <button className="twin-follow" onClick={onCancelFollow}>{t.twin.worker} #{followedAgentId} · {t.twin.cancelFollow} ×</button>}
      <div className="twin-inspector">
        <div className="flex items-center justify-between"><span className="twin-eyebrow">{t.twin.inspect}</span>{selection && <button aria-label={t.common.close} onClick={() => setSelection(null)}>×</button>}</div>
        {selectedNode ? <><strong>{nodeName(selectedNode)} {selectedNode.node_id}</strong><p>{selectedNode.label}</p><dl><dt>{t.mapView.level}</dt><dd>{selectedNode.level}</dd><dt>{t.twin.depth}</dt><dd>{selectedNode.position[2]} m</dd><dt>{t.twin.capacity}</dt><dd>{selectedNode.capacity}</dd></dl></>
          : selectedEdge ? <><strong>{selectedEdge.edge_id} · {selectedEdge.source} → {selectedEdge.target}</strong><dl>
            <dt>{t.twin.status}</dt><dd>{t.legend[selectedEdge.status]} ({selectedEdge.status})</dd>
            <dt>{t.twin.width}</dt><dd>{selectedEdge.width_m.toFixed(2)} m</dd><dt>{t.twin.length}</dt><dd>{selectedEdge.length_m.toFixed(2)} m</dd>
            <dt>{t.twin.slope}</dt><dd>{selectedEdge.slope_pct.toFixed(1)}%</dd><dt>{t.twin.risk}</dt><dd>{(selectedEdge.current_risk * 100).toFixed(1)}%</dd></dl></>
          : selectedAgent ? <><strong>{t.twin.worker} #{selectedAgent.agent_id}</strong><dl><dt>{t.twin.status}</dt><dd>{agentStatus(selectedAgent)}</dd><dt>{t.twin.panic}</dt><dd>{Math.round(selectedAgent.panic_level * 100)}%</dd><dt>{t.twin.distance}</dt><dd>{selectedAgent.cumulative_distance_m.toFixed(1)} m</dd></dl>
            <button className="twin-action" disabled={selectedAgent.status === "evacuated"} onClick={() => { setLevel(null); props.onSelectAgent(String(selectedAgent.agent_id)); }}>{t.twin.follow}</button></>
          : <p>{t.twin.noSelection}</p>}
        <label className="twin-worker-select"><span>{t.twin.selectWorker}</span><select value={selectedAgent?.agent_id ?? ""} onChange={e => { if (e.target.value) onSelectWorker(e.target.value); }}>
          <option value="">—</option>{Object.values(agents).map(a => <option key={a.agent_id} value={a.agent_id}>#{a.agent_id} · {agentStatus(a)}</option>)}
        </select></label>
      </div>
      <div className="twin-scene-footer"><span>{view === "routes" ? t.twin.routesHint : t.twin.navigation}</span><span>{nodes.length} N · {edges.length} E</span></div>
    </div>
  </div>;
}
