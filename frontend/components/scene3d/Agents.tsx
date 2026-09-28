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
      {/* Torso */}
      <mesh position={[0, 0.05, 0]}>
        <boxGeometry args={[0.08, 0.15, 0.05]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>
      
      {/* Cabeza */}
      <mesh position={[0, 0.15, 0]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <meshStandardMaterial
          color="#FFE4C4"
          emissive="#FFE4C4"
          emissiveIntensity={0.15}
          roughness={0.6}
          metalness={0.05}
        />
      </mesh>
      
      {/* Casco de seguridad */}
      <mesh position={[0, 0.17, 0]}>
        <sphereGeometry args={[0.045, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial
          color="#FFD700"
          emissive="#FFD700"
          emissiveIntensity={0.25}
          roughness={0.3}
          metalness={0.4}
        />
      </mesh>
      
      {/* Linterna del casco */}
      <mesh position={[0, 0.18, 0.035]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.012, 0.035, 8]} />
        <meshBasicMaterial
          color="#FFFF00"
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Brazo izquierdo */}
      <mesh position={[-0.06, 0.05, 0]}>
        <capsuleGeometry args={[0.015, 0.1, 4, 8]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>
      
      {/* Brazo derecho */}
      <mesh position={[0.06, 0.05, 0]}>
        <capsuleGeometry args={[0.015, 0.1, 4, 8]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>
      
      {/* Pierna izquierda */}
      <mesh position={[-0.025, -0.08, 0]}>
        <capsuleGeometry args={[0.018, 0.12, 4, 8]} />
        <meshStandardMaterial
          color="#2C3E50"
          emissive="#2C3E50"
          emissiveIntensity={0.1}
          roughness={0.6}
          metalness={0.05}
        />
      </mesh>
      
      {/* Pierna derecha */}
      <mesh position={[0.025, -0.08, 0]}>
        <capsuleGeometry args={[0.018, 0.12, 4, 8]} />
        <meshStandardMaterial
          color="#2C3E50"
          emissive="#2C3E50"
          emissiveIntensity={0.1}
          roughness={0.6}
          metalness={0.05}
        />
      </mesh>

      {/* Halo de pánico */}
      {isPanicked && (
        <mesh scale={[1.4, 1.4, 1.4]}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial
            color={SCENE_COLORS.agentPanicHalo}
            transparent
            opacity={0.25}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Indicador de seguimiento */}
      {isFollowed && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.15, 0]}>
          <ringGeometry args={[0.12, 0.16, 32]} />
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
