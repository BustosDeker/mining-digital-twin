"use client";

import { useMemo } from "react";
import type { AgentSnapshot, MineNode } from "@/lib/types";
import { backendToThreePosition, lerpPosition } from "./geometry";
import { SCENE_COLORS } from "./colors";

interface AgentsProps {
  nodes: MineNode[];
  agents: Record<string, AgentSnapshot>;
  followedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  performanceMode: boolean;
}

const PANIC_HIGHLIGHT_THRESHOLD = 0.6;

const COLOR_BY_STATUS: Record<string, string> = {
  moving: SCENE_COLORS.agentMoving,
  waiting: SCENE_COLORS.agentWaiting,
  sheltered: SCENE_COLORS.agentSheltered,
  evacuated: SCENE_COLORS.agentEvacuated,
  lost: SCENE_COLORS.agentLost,
};

// Componente para representar un minero de forma más detallada
function MinerFigure({ color, isPanicked, isFollowed }: { color: string; isPanicked: boolean; isFollowed: boolean }) {
  return (
    <group>
      {/* Cuerpo del minero */}
      <mesh position={[0, 0, 0]}>
        <capsuleGeometry args={[0.12, 0.35, 4, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.4}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>
      
      {/* Cabeza del minero */}
      <mesh position={[0, 0.28, 0]}>
        <sphereGeometry args={[0.09, 12, 12]} />
        <meshStandardMaterial
          color="#FFE4C4"
          emissive="#FFE4C4"
          emissiveIntensity={0.2}
          roughness={0.6}
          metalness={0.05}
        />
      </mesh>
      
      {/* Casco de seguridad */}
      <mesh position={[0, 0.32, 0]}>
        <sphereGeometry args={[0.095, 12, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial
          color="#FFD700"
          emissive="#FFD700"
          emissiveIntensity={0.3}
          roughness={0.3}
          metalness={0.3}
        />
      </mesh>
      
      {/* Luz del casco */}
      <mesh position={[0, 0.35, 0.08]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.03, 0.08, 8]} />
        <meshBasicMaterial
          color="#FFFF00"
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Halo de pánico */}
      {isPanicked && (
        <mesh scale={[1.6, 1.6, 1.6]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshBasicMaterial
            color={SCENE_COLORS.agentPanicHalo}
            transparent
            opacity={0.3}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Indicador de seguimiento */}
      {isFollowed && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.35, 0]}>
          <ringGeometry args={[0.25, 0.32, 32]} />
          <meshBasicMaterial 
            color={SCENE_COLORS.agentEvacuated}
            transparent
            opacity={0.8}
          />
        </mesh>
      )}
    </group>
  );
}

export function Agents({
  nodes,
  agents,
  followedAgentId,
  onSelectAgent,
  performanceMode,
}: AgentsProps) {
  const nodePositions = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const node of nodes) {
      map.set(node.node_id, backendToThreePosition(node.position));
    }
    return map;
  }, [nodes]);

  const visibleAgents = Object.values(agents).filter(
    (a) => a.status !== "evacuated"
  );

  return (
    <group>
      {visibleAgents.map((agent) => {
        const from = nodePositions.get(agent.node_id);
        const to = nodePositions.get(agent.next_node_id) ?? from;
        if (!from || !to) return null;

        const position = lerpPosition(from, to, agent.progress);
        const color = COLOR_BY_STATUS[agent.status] ?? SCENE_COLORS.agentMoving;
        const isPanicked = agent.panic_level >= PANIC_HIGHLIGHT_THRESHOLD;
        const isFollowed = followedAgentId === String(agent.agent_id);

        return (
          <group
            key={agent.agent_id}
            position={position}
            onClick={(e) => {
              e.stopPropagation();
              onSelectAgent(String(agent.agent_id));
            }}
          >
            {!performanceMode && (
              <MinerFigure color={color} isPanicked={isPanicked} isFollowed={isFollowed} />
            )}
            {performanceMode && (
              <mesh>
                <sphereGeometry args={[0.18, 8, 8]} />
                <meshBasicMaterial color={color} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}
