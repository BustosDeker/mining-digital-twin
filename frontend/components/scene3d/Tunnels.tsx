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

const RADIUS_BY_WIDTH = (widthM: number) => Math.max(0.12, widthM * 0.06);

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

        const emissiveIntensity = edge.status === "clear" ? 0.35 : 0.6;

        return (
          <mesh
            key={edge.edge_id}
            position={position}
            quaternion={quaternion}
          >
            <cylinderGeometry
              args={[RADIUS_BY_WIDTH(edge.width_m), RADIUS_BY_WIDTH(edge.width_m), length, 10]}
            />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={emissiveIntensity}
              transparent
              opacity={opacity}
              roughness={0.55}
              metalness={0.1}
            />
          </mesh>
        );
      })}
    </group>
  );
}
