"use client";

import { useRef, useMemo } from "react";
import { Html } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

export interface SceneLabel {
  id: string; position: [number, number, number]; text: string; color: string; priority: number;
}

/** Screen-space collision culling. DOM visibility changes do not trigger React renders. */
export function SceneLabels({ labels }: { labels: SceneLabel[] }) {
  const elements = useRef(new Map<string, HTMLDivElement>());
  const { camera, size, gl } = useThree();
  const scratch = useMemo(() => new THREE.Vector3(), []);
  const ordered = useMemo(() => [...labels].sort((a, b) => b.priority - a.priority), [labels]);
  const elapsed = useRef(0);
  useFrame((_, delta) => {
    elapsed.current += delta;
    if (elapsed.current < 0.12) return;
    elapsed.current = 0;
    const viewport = gl.domElement.getBoundingClientRect();
    const overlays = gl.domElement.closest(".twin-scene")?.querySelectorAll<HTMLElement>(
      ".twin-map-caption, .twin-levels, .twin-inspector, .twin-follow, .twin-scene-footer"
    );
    const placed = Array.from(overlays ?? []).map(element => {
      const r = element.getBoundingClientRect();
      return { x: r.left - viewport.left + r.width / 2, y: r.top - viewport.top + r.height / 2,
        width: r.width + 8, height: r.height + 8 };
    });
    for (const label of ordered) {
      const element = elements.current.get(label.id);
      if (!element) continue;
      scratch.set(...label.position).project(camera);
      const x = (scratch.x + 1) * size.width / 2;
      const y = (1 - scratch.y) * size.height / 2;
      const width = element.offsetWidth + 12, height = element.offsetHeight + 10;
      const visible = scratch.z > -1 && scratch.z < 1 && x > width / 2 && x < size.width - width / 2
        && y > 32 && y < size.height - 24
        && !placed.some(p => Math.abs(p.x - x) < (p.width + width) / 2 && Math.abs(p.y - y) < (p.height + height) / 2);
      element.style.visibility = visible ? "visible" : "hidden";
      if (visible) placed.push({ x, y, width, height });
    }
  });
  return <>{ordered.map(label => <Html key={label.id} position={label.position} center zIndexRange={[8, 0]} style={{ pointerEvents: "none" }}>
    <div ref={element => { if (element) elements.current.set(label.id, element); else elements.current.delete(label.id); }}
      className="twin-scene-label" style={{ borderLeftColor: label.color, visibility: "hidden" }}>{label.text}</div>
  </Html>)}</>;
}
