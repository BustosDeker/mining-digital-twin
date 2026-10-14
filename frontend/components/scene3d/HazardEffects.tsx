"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { HazardEvent, MineNode } from "@/lib/types";
import { backendToThreePosition } from "./geometry";

function Hazard({ event, position, simple }: { event: HazardEvent; position: [number, number, number]; simple: boolean }) {
  const animated = useRef<THREE.Group>(null);
  const severity = Math.min(1, Math.max(0, event.intensity));
  const fire = event.hazard_type === "fire", gas = event.hazard_type === "gas_leak";
  const color = fire ? "#ef9851" : gas ? "#bac789" : "#d3b292";
  useFrame(({ clock }) => {
    if (simple || !animated.current) return;
    // Decorative flicker only; never advances or propagates the hazard.
    const pulse = 1 + Math.sin(clock.elapsedTime * (fire ? 6 : 1.4)) * (fire ? 0.1 : 0.04);
    animated.current.scale.set(1, pulse, 1);
  });
  return <group position={position}>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.015, 0]}>
      <ringGeometry args={[0.42, 0.45, simple ? 12 : 24]} /><meshBasicMaterial color={color} side={2} />
    </mesh>
    <group ref={animated} scale={[1, 1, 1]}>
      {fire ? Array.from({ length: simple ? 2 : 5 }, (_, i) => <mesh key={i}
        position={[Math.sin(i * 2.4) * 0.15, (0.2 + severity * 0.22) / 2, Math.cos(i * 2.4) * 0.15]}
        scale={[1, 0.8 + (i % 3) * 0.25, 1]}>
        <coneGeometry args={[0.06 + severity * 0.07, 0.2 + severity * 0.45, simple ? 4 : 7]} />
        <meshBasicMaterial color={i % 2 ? "#ffcb77" : "#eb713e"} transparent opacity={0.85} depthWrite={false} />
      </mesh>) : gas ? Array.from({ length: simple ? 1 : 5 }, (_, i) => <mesh key={i}
        position={[Math.sin(i * 2.4) * 0.23, 0.18 + (i % 3) * 0.12, Math.cos(i * 2.4) * 0.23]} scale={[1.3, 0.8, 1]}>
        <icosahedronGeometry args={[0.2 + severity * 0.22, simple ? 0 : 1]} />
        <meshBasicMaterial color={color} transparent opacity={0.1 + severity * 0.12} depthWrite={false} />
      </mesh>) : Array.from({ length: simple ? 3 : 9 }, (_, i) => <mesh key={i}
        position={[Math.sin(i * 2.4) * (0.1 + i * 0.025), 0.09 + (i % 3) * 0.04, Math.cos(i * 2.4) * 0.2]}
        rotation={[i * 0.7, i, i * 0.4]} scale={[1, 0.65, 0.8]}>
        <dodecahedronGeometry args={[0.08 + severity * 0.1, 0]} /><meshStandardMaterial color={i % 2 ? "#7c7367" : "#a79a84"} roughness={1} />
      </mesh>)}
      {!simple && !gas && [0, 1, 2].map(i => <mesh key={`haze-${i}`} position={[i * 0.05, 0.35 + i * 0.18, 0]}>
        <icosahedronGeometry args={[0.13 + i * 0.05, 1]} /><meshBasicMaterial color={fire ? "#8c9196" : "#b8a993"} transparent opacity={severity * 0.12} depthWrite={false} />
      </mesh>)}
    </group>
    {!gas && !fire && <group position={[0, 0.47, 0]}>{[-1, 1].map(s => <mesh key={s} rotation={[0, 0, s * Math.PI / 4]}><boxGeometry args={[0.3, 0.035, 0.035]} /><meshBasicMaterial color="#ef7068" /></mesh>)}</group>}
  </group>;
}

export function HazardEffects({ nodes, hazards, performanceMode }: { nodes: MineNode[]; hazards: HazardEvent[]; performanceMode: boolean }) {
  const positions = useMemo(() => new Map(nodes.map(n => [n.node_id, backendToThreePosition(n.position)])), [nodes]);
  const fire = hazards.find(h => h.hazard_type === "fire" && positions.has(h.origin_node_id));
  const firePosition = fire ? positions.get(fire.origin_node_id)! : null;
  return <>
    {!performanceMode && fire && firePosition && <pointLight position={[firePosition[0], firePosition[1] + 0.5, firePosition[2]]}
      color="#f0a265" intensity={1 + Math.max(0, Math.min(1, fire.intensity)) * 2} distance={2.5} decay={2} />}
    {hazards.map(event => {
    const position = positions.get(event.origin_node_id);
    return position ? <Hazard key={event.event_id} event={event} position={position} simple={performanceMode} /> : null;
  })}</>;
}
