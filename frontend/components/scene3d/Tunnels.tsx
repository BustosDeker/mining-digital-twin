"use client";

import { memo, useEffect, useMemo } from "react";
import * as THREE from "three";
import type { MineEdge, MineNode } from "@/lib/types";
import { backendToThreePosition, galleryBetween, WORLD_SCALE } from "./geometry";
import { SCENE_COLORS as C } from "./colors";

interface TunnelsProps {
  nodes: MineNode[];
  edges: MineEdge[];
  opacity: number;
  performanceMode: boolean;
  affectedEdges: Set<string>;
  occupiedEdges: Set<string>;
  routes: boolean;
  selectedEdge: string | null;
  onSelectEdge: (id: string) => void;
}

// An open-bottom arched section, in local XY. Width uses the SAME scale as positions.
function profile(width: number): THREE.Vector2[] {
  const r = width / 2;
  return [new THREE.Vector2(-r, 0), new THREE.Vector2(-r, width * 0.48),
    ...Array.from({ length: 9 }, (_, i) => {
      const angle = Math.PI - i * Math.PI / 8;
      return new THREE.Vector2(Math.cos(angle) * r, width * 0.48 + Math.sin(angle) * r);
    }), new THREE.Vector2(r, 0)];
}

function archGeometry(width: number, length: number, frames: boolean, simple: boolean) {
  const outline = profile(width);
  const vertices: number[] = [];
  const quad = (a: number[], b: number[], c: number[], d: number[]) => vertices.push(...a, ...b, ...c, ...a, ...c, ...d);
  const count = frames ? Math.max(2, Math.min(simple ? 4 : 18, Math.ceil(length / (simple ? 1.5 : 0.65)))) : 1;
  for (let j = 0; j < count; j++) {
    const center = frames ? -length / 2 + j * length / (count - 1) : 0;
    const half = frames ? 0.014 : length / 2;
    for (let i = 0; i < outline.length - 1; i++) {
      const a = outline[i], b = outline[i + 1];
      quad([a.x, a.y, center - half], [b.x, b.y, center - half], [b.x, b.y, center + half], [a.x, a.y, center + half]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  return geometry;
}

const Gallery = memo(function Gallery({ edge, a, b, opacity, simple, affected, occupied, selected, onSelect }: {
  edge: MineEdge; a: [number, number, number]; b: [number, number, number];
  opacity: number; simple: boolean; affected: boolean; occupied: boolean;
  selected: boolean; onSelect: (id: string) => void;
}) {
  const transform = useMemo(() => galleryBetween(a, b), [a, b]);
  const width = Math.max(0.01, edge.width_m * WORLD_SCALE);
  const shell = useMemo(() => archGeometry(width, transform.length, false, simple), [width, transform.length, simple]);
  const ribs = useMemo(() => archGeometry(width * 0.99, transform.length, true, simple), [width, transform.length, simple]);
  useEffect(() => () => { shell.dispose(); ribs.dispose(); }, [shell, ribs]);
  const signal = C[edge.status];
  const length = transform.length;
  if (length < 1e-6) return null;
  return (
    <group position={transform.position} quaternion={transform.quaternion} onClick={(e) => { e.stopPropagation(); onSelect(edge.edge_id); }}>
      <mesh position={[0, -0.025, 0]}>
        <boxGeometry args={[width, 0.05, length]} />
        <meshStandardMaterial color={selected ? "#748ea1" : C.floor} roughness={0.95} />
      </mesh>
      {/* Cutaway roof: opacity controls the rock envelope, never the safety signals. */}
      <mesh geometry={shell} raycast={() => undefined}>
        <meshStandardMaterial color={C.rock} side={THREE.DoubleSide} transparent opacity={opacity * 0.3} depthWrite={false} roughness={1} />
      </mesh>
      <mesh geometry={ribs}>
        <meshStandardMaterial color={C.support} side={THREE.DoubleSide} metalness={0.45} roughness={0.65} />
      </mesh>
      {[-1, 1].map(side => (
        <mesh key={side} position={[side * width * 0.43, 0.035, 0]}>
          <boxGeometry args={[0.014, 0.018, length]} />
          <meshBasicMaterial color={signal} />
        </mesh>
      ))}
      {!simple && <mesh position={[0, width * 0.97, 0]}>
        <boxGeometry args={[0.018, 0.016, length * 0.92]} />
        <meshBasicMaterial color={edge.status === "clear" ? "#e1d8b9" : signal} />
      </mesh>}
      {(affected || occupied || selected) && <mesh position={[0, 0.012, 0]}>
        <boxGeometry args={[width * 0.14, 0.018, length]} />
        <meshBasicMaterial color={occupied || selected ? "#a8d7ef" : signal} transparent opacity={0.4 + Math.min(1, Math.max(0, edge.current_risk)) * 0.5} />
      </mesh>}
      {edge.status !== "clear" && <group position={[0, width * 0.4, 0]}>
        <mesh rotation={[0, 0, edge.status === "blocked" ? Math.PI / 5 : 0]}>
          <boxGeometry args={[width * 0.95, 0.055, 0.045]} />
          <meshBasicMaterial color={signal} />
        </mesh>
        {edge.status === "blocked" && <mesh rotation={[0, 0, -Math.PI / 5]}>
          <boxGeometry args={[width * 0.95, 0.055, 0.045]} />
          <meshBasicMaterial color={signal} />
        </mesh>}
      </group>}
    </group>
  );
});

export function Tunnels({ nodes, edges, opacity, performanceMode, affectedEdges, occupiedEdges, routes, selectedEdge, onSelectEdge }: TunnelsProps) {
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, backendToThreePosition(n.position)])), [nodes]);
  return <group>{edges.map(edge => {
    const a = positions.get(edge.source), b = positions.get(edge.target);
    return a && b ? <Gallery key={edge.edge_id} edge={edge} a={a} b={b} opacity={opacity}
      simple={performanceMode} affected={affectedEdges.has(edge.edge_id)} occupied={routes && occupiedEdges.has(edge.edge_id)}
      selected={selectedEdge === edge.edge_id} onSelect={onSelectEdge} /> : null;
  })}</group>;
}
