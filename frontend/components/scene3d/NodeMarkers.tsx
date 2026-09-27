"use client";

import { useMemo } from "react";
import { Html } from "@react-three/drei";
import type { MineNode } from "@/lib/types";
import { backendToThreePosition } from "./geometry";
import { SCENE_COLORS } from "./colors";

interface NodeMarkersProps {
  nodes: MineNode[];
  showLabels: boolean;
}

const SIZE_BY_TYPE: Record<string, number> = {
  exit: 0.42,
  refuge_chamber: 0.36,
  risk_zone: 0.3,
  intersection: 0.18,
  gallery: 0.1,
};

const COLOR_BY_TYPE: Record<string, string> = {
  exit: SCENE_COLORS.exit,
  refuge_chamber: SCENE_COLORS.refuge,
  risk_zone: SCENE_COLORS.riskZone,
  intersection: SCENE_COLORS.intersection,
  gallery: SCENE_COLORS.gallery,
};

export function NodeMarkers({ nodes, showLabels }: NodeMarkersProps) {
  const notable = useMemo(
    () => nodes.filter((n) => n.node_type !== "gallery"),
    [nodes]
  );

  return (
    <group>
      {notable.map((node) => {
        const pos = backendToThreePosition(node.position);
        const size = SIZE_BY_TYPE[node.node_type] ?? 0.15;
        const color = COLOR_BY_TYPE[node.node_type] ?? SCENE_COLORS.intersection;

        return (
          <group key={node.node_id} position={pos}>
            <mesh>
              <sphereGeometry args={[size, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.5}
                roughness={0.4}
              />
            </mesh>
            {showLabels && (node.label || node.node_type !== "intersection") && (
              <Html distanceFactor={12} position={[0, size + 0.3, 0]} center>
                <div className="pointer-events-none whitespace-nowrap rounded-sm bg-void/80 px-1.5 py-0.5 font-mono text-[9px] text-steel2">
                  {node.label || node.node_id}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}
