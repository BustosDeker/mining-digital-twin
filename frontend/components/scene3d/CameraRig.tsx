"use client";

import { useRef, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { AgentSnapshot, MineNode } from "@/lib/types";
import { backendToThreePosition, lerpPosition } from "./geometry";

interface CameraRigProps {
  nodes: MineNode[];
  agents: Record<string, AgentSnapshot>;
  followedAgentId: string | null;
}

export function CameraRig({ nodes, agents, followedAgentId }: CameraRigProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();
  const nodePositions = useRef(new Map<string, [number, number, number]>());

  useEffect(() => {
    const map = new Map<string, [number, number, number]>();
    for (const node of nodes) {
      map.set(node.node_id, backendToThreePosition(node.position));
    }
    nodePositions.current = map;
  }, [nodes]);

  useFrame(() => {
    if (!followedAgentId) return;
    const agent = agents[followedAgentId];
    if (!agent) return;

    const from = nodePositions.current.get(agent.node_id);
    const to = nodePositions.current.get(agent.next_node_id) ?? from;
    if (!from || !to) return;

    const targetPos = lerpPosition(from, to, agent.progress);
    const target = new THREE.Vector3(...targetPos);

    if (controlsRef.current) {
      controlsRef.current.target.lerp(target, 0.08);
      controlsRef.current.update();
    }
    const desiredCameraPos = target
      .clone()
      .add(new THREE.Vector3(2.5, 2.2, 2.5));
    camera.position.lerp(desiredCameraPos, 0.04);
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={2}
      maxDistance={60}
    />
  );
}
