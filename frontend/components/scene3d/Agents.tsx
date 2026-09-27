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
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[0.14, 0.32, 4, 8]} />
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={0.6}
                  roughness={0.5}
                />
              </mesh>
            )}
            {performanceMode && (
              <mesh>
                <sphereGeometry args={[0.16, 6, 6]} />
                <meshBasicMaterial color={color} />
              </mesh>
            )}

            {isPanicked && (
              <mesh scale={1.8}>
                <sphereGeometry args={[0.18, 12, 12]} />
                <meshBasicMaterial
                  color={SCENE_COLORS.agentPanicHalo}
                  transparent
                  opacity={0.25}
                  depthWrite={false}
                />
              </mesh>
            )}

            {isFollowed && (
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.3, 0]}>
                <ringGeometry args={[0.28, 0.36, 24]} />
                <meshBasicMaterial color={SCENE_COLORS.agentEvacuated} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}
