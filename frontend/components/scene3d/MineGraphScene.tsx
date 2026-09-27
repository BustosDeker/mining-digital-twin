"use client";

import { Canvas } from "@react-three/fiber";
import { Grid } from "@react-three/drei";
import type { AgentSnapshot, MineEdge, MineNode } from "@/lib/types";
import { Tunnels } from "./Tunnels";
import { NodeMarkers } from "./NodeMarkers";
import { Agents } from "./Agents";
import { CameraRig } from "./CameraRig";
import { SCENE_COLORS } from "./colors";

interface MineGraphSceneProps {
  nodes: MineNode[];
  edges: MineEdge[];
  agents: Record<string, AgentSnapshot>;
  tunnelOpacity: number;
  showLabels: boolean;
  performanceMode: boolean;
  followedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  theme: "dark" | "light";
}

export function MineGraphScene({
  nodes,
  edges,
  agents,
  tunnelOpacity,
  showLabels,
  performanceMode,
  followedAgentId,
  onSelectAgent,
  theme,
}: MineGraphSceneProps) {
  const background =
    theme === "dark" ? SCENE_COLORS.background : SCENE_COLORS.backgroundLight;

  return (
    <Canvas
      camera={{ position: [10, 9, 10], fov: 50 }}
      dpr={performanceMode ? 1 : [1, 1.8]}
      shadows={false}
    >
      <color attach="background" args={[background]} />
      <fog attach="fog" args={[background, 20, 70]} />

      <ambientLight intensity={0.55} />
      <directionalLight position={[8, 12, 6]} intensity={0.6} />

      <Grid
        args={[80, 80]}
        cellColor={theme === "dark" ? "#1E2724" : "#C7CAC3"}
        sectionColor={theme === "dark" ? "#2A3532" : "#B4B8AF"}
        fadeDistance={45}
        fadeStrength={1.5}
        position={[0, -0.02, 0]}
        infiniteGrid
      />

      <Tunnels nodes={nodes} edges={edges} opacity={tunnelOpacity} />
      <NodeMarkers nodes={nodes} showLabels={showLabels} />
      <Agents
        nodes={nodes}
        agents={agents}
        followedAgentId={followedAgentId}
        onSelectAgent={onSelectAgent}
        performanceMode={performanceMode}
      />
      <CameraRig nodes={nodes} agents={agents} followedAgentId={followedAgentId} />
    </Canvas>
  );
}
