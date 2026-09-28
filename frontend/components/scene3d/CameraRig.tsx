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
  panMode: boolean;
}

export function CameraRig({ nodes, agents, followedAgentId, panMode }: CameraRigProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();
  const nodePositions = useRef(new Map<string, [number, number, number]>());
  const lastLayoutSignature = useRef("");

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    controls.listenToKeyEvents(document.body);
    controls.enablePan = true;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 1.2;
    controls.rotateSpeed = 1;
    controls.target.set(0, 0, 0);

    return () => controls.stopListenToKeyEvents();
  }, []);

  useEffect(() => {
    const map = new Map<string, [number, number, number]>();
    for (const node of nodes) {
      map.set(node.node_id, backendToThreePosition(node.position));
    }
    nodePositions.current = map;

    const layoutSignature = nodes
      .map((node) => `${node.node_id}:${node.position.join(",")}`)
      .join("|");
    if (!nodes.length || layoutSignature === lastLayoutSignature.current) return;
    lastLayoutSignature.current = layoutSignature;

    const bounds = new THREE.Box3();
    for (const position of map.values()) {
      bounds.expandByPoint(new THREE.Vector3(...position));
    }
    const center = bounds.getCenter(new THREE.Vector3());
    const sphere = bounds.getBoundingSphere(new THREE.Sphere());
    const fov = THREE.MathUtils.degToRad(
      (camera as THREE.PerspectiveCamera).fov ?? 45
    );
    const distance = Math.max(18, (sphere.radius / Math.sin(fov / 2)) * 1.4);
    const direction = new THREE.Vector3(1, 0.8, 1).normalize();

    controlsRef.current?.target.copy(center);
    camera.position.copy(center).addScaledVector(direction, distance);
    controlsRef.current?.update();
  }, [camera, nodes]);

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
      minDistance={4}
      maxDistance={220}
      minPolarAngle={0.15}
      maxPolarAngle={Math.PI / 2 - 0.1}
      enablePan={true}
      panSpeed={2.8}
      keyPanSpeed={7}
      mouseButtons={
        panMode
          ? {
              LEFT: THREE.MOUSE.PAN,
              MIDDLE: THREE.MOUSE.DOLLY,
              RIGHT: THREE.MOUSE.ROTATE,
            }
          : {
              LEFT: THREE.MOUSE.ROTATE,
              MIDDLE: THREE.MOUSE.DOLLY,
              RIGHT: THREE.MOUSE.PAN,
            }
      }
      rotateSpeed={1}
      zoomSpeed={1.2}
      minAzimuthAngle={-Infinity}
      maxAzimuthAngle={Infinity}
      screenSpacePanning={true}
      zoomToCursor
    />
  );
}
