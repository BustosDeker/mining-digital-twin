"use client";

import { Component, ReactNode } from "react";
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

class WebGLErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full items-center justify-center bg-panel p-8 text-center">
          <div className="max-w-md">
            <h3 className="mb-3 text-lg font-semibold text-foreground">
              Error al inicializar WebGL
            </h3>
            <p className="mb-3 text-sm text-muted-foreground">
              {this.state.error?.message || "No se pudo crear el contexto WebGL"}
            </p>
            <p className="text-sm text-muted-foreground">
              El browser preview puede no soportar WebGL. Intenta abrir la aplicación directamente en tu navegador:
            </p>
            <a
              href="http://localhost:3001"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-block rounded bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
            >
              Abrir en navegador
            </a>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function SceneContent({
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
      gl={{
        antialias: !performanceMode,
        alpha: false,
        powerPreference: "high-performance",
        failIfMajorPerformanceCaveat: false,
        preserveDrawingBuffer: true,
      }}
    >
      <color attach="background" args={[background]} />
      <fog attach="fog" args={[background, 15, 60]} />

      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 15, 8]} intensity={0.8} />
      <directionalLight position={[-5, 10, -5]} intensity={0.3} />
      <pointLight position={[0, 5, 0]} intensity={0.2} distance={30} />

      <Grid
        args={[80, 80]}
        cellColor={theme === "dark" ? "#1E2724" : "#C7CAC3"}
        sectionColor={theme === "dark" ? "#2A3532" : "#B4B8AF"}
        fadeDistance={40}
        fadeStrength={1.8}
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

export function MineGraphScene(props: MineGraphSceneProps) {
  return (
    <WebGLErrorBoundary>
      <SceneContent {...props} />
    </WebGLErrorBoundary>
  );
}
