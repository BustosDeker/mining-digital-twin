"use client";

import { useMemo } from "react";
import type { MineNode, MineEdge } from "@/lib/types";
import { backendToThreePosition } from "./geometry";
import { SCENE_COLORS as C } from "./colors";

interface NodeMarkersProps {
  nodes: MineNode[]; edges: MineEdge[]; onSelectNode: (id: string) => void;
}

export function NodeMarkers({ nodes, edges, onSelectNode }: NodeMarkersProps) {
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, backendToThreePosition(n.position)])), [nodes]);
  return <group>{nodes.filter(n => !["gallery", "intersection"].includes(n.node_type)).map(node => {
    const position = positions.get(node.node_id)!;
    const edge = edges.find(e => e.source === node.node_id || e.target === node.node_id);
    const other = edge ? positions.get(edge.source === node.node_id ? edge.target : edge.source) : null;
    const angle = other ? Math.atan2(other[0] - position[0], other[2] - position[2]) : 0;
    const isExit = node.node_type === "exit", isRefuge = node.node_type === "refuge_chamber";
    const color = isExit ? C.exit : isRefuge ? C.refuge : C.riskZone;
    return <group key={node.node_id} position={position} rotation={[0, angle, 0]}
      onClick={e => { e.stopPropagation(); onSelectNode(node.node_id); }}>
      {(isExit || isRefuge) ? <>
        <mesh position={[0, 0.015, 0]}><boxGeometry args={[0.7, 0.03, isRefuge ? 0.7 : 0.3]} /><meshStandardMaterial color="#596972" /></mesh>
        {isRefuge && <mesh position={[0, 0.25, -0.19]}><boxGeometry args={[0.65, 0.5, 0.4]} /><meshStandardMaterial color="#476374" metalness={0.35} roughness={0.65} /></mesh>}
        {[-1, 1].map(side => <mesh key={side} position={[side * 0.26, 0.27, 0.06]}>
          <boxGeometry args={[0.065, 0.54, 0.09]} /><meshStandardMaterial color="#b3bfc0" metalness={0.5} roughness={0.5} />
        </mesh>)}
        <mesh position={[0, 0.55, 0.06]}><boxGeometry args={[0.6, 0.14, 0.1]} /><meshBasicMaterial color={color} /></mesh>
        {isRefuge ? <>
          <mesh position={[0, 0.25, 0.025]}><boxGeometry args={[0.39, 0.44, 0.045]} /><meshStandardMaterial color="#193749" /></mesh>
          <mesh position={[0, 0.33, 0.057]}><boxGeometry args={[0.05, 0.18, 0.015]} /><meshBasicMaterial color="#edf6f3" /></mesh>
          <mesh position={[0, 0.33, 0.057]}><boxGeometry args={[0.18, 0.05, 0.018]} /><meshBasicMaterial color="#edf6f3" /></mesh>
        </> : <mesh position={[0, 0.55, 0.125]} rotation={[0, 0, -Math.PI / 2]}><coneGeometry args={[0.055, 0.13, 3]} /><meshBasicMaterial color="#17382d" /></mesh>}
      </> : <>
        <mesh position={[0, 0.2, 0]}><boxGeometry args={[0.025, 0.4, 0.025]} /><meshStandardMaterial color="#b0b4ab" /></mesh>
        <mesh position={[0, 0.43, 0]} rotation={[0, 0, 0]}><circleGeometry args={[0.18, 3, Math.PI / 2]} /><meshBasicMaterial color={color} side={2} /></mesh>
        <mesh position={[0, 0.43, 0.006]}><boxGeometry args={[0.018, 0.09, 0.014]} /><meshBasicMaterial color="#33291b" /></mesh>
        <mesh position={[0, 0.36, 0.007]}><boxGeometry args={[0.021, 0.018, 0.014]} /><meshBasicMaterial color="#33291b" /></mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.015, 0]}><ringGeometry args={[0.28, 0.3, 16]} /><meshBasicMaterial color={color} side={2} /></mesh>
      </>}
    </group>;
  })}</group>;
}
