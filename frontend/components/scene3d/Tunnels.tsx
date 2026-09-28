"use client";

import { useMemo } from "react";
import type { MineEdge, MineNode } from "@/lib/types";
import { backendToThreePosition, cylinderBetween } from "./geometry";
import { SCENE_COLORS } from "./colors";

interface TunnelsProps {
  nodes: MineNode[];
  edges: MineEdge[];
  opacity: number;
}

const RADIUS_BY_WIDTH = (widthM: number) => Math.max(0.15, widthM * 0.07);

export function Tunnels({ nodes, edges, opacity }: TunnelsProps) {
  const nodePositions = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const node of nodes) {
      map.set(node.node_id, backendToThreePosition(node.position));
    }
    return map;
  }, [nodes]);

  return (
    <group>
      {edges.map((edge) => {
        const a = nodePositions.get(edge.source);
        const b = nodePositions.get(edge.target);
        if (!a || !b) return null;

        const { position, quaternion, length } = cylinderBetween(a, b);
        const color =
          edge.status === "blocked"
            ? SCENE_COLORS.blocked
            : edge.status === "degraded"
              ? SCENE_COLORS.degraded
              : SCENE_COLORS.clear;

        const emissiveIntensity = edge.status === "clear" ? 0.25 : 0.5;
        const tunnelOpacity = edge.status === "clear" ? opacity * 0.7 : opacity * 0.85;

        return (
          <group key={edge.edge_id} position={position} quaternion={quaternion}>
            {/* Túnel exterior - más translúcido */}
            <mesh>
              <cylinderGeometry
                args={[RADIUS_BY_WIDTH(edge.width_m), RADIUS_BY_WIDTH(edge.width_m), length, 16]}
              />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={emissiveIntensity}
                transparent
                opacity={tunnelOpacity}
                roughness={0.3}
                metalness={0.2}
                side={2}
              />
            </mesh>
            
            {/* Túnel interior - más brillante para dar profundidad */}
            <mesh>
              <cylinderGeometry
                args={[RADIUS_BY_WIDTH(edge.width_m) * 0.85, RADIUS_BY_WIDTH(edge.width_m) * 0.85, length, 16]}
              />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={emissiveIntensity * 1.5}
                transparent
                opacity={tunnelOpacity * 0.6}
                roughness={0.4}
                metalness={0.1}
                side={2}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
