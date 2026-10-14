import * as THREE from "three";

export const WORLD_SCALE = 0.12;

/**
 * Convierte una posición del backend (x, y, z_nivel_profundidad en metros)
 * a coordenadas Three.js (Y arriba). El backend usa z negativo = más
 * profundo; en Three.js mapeamos ese mismo eje a Y para que "más profundo"
 * quede visualmente más abajo, y usamos el y del backend como profundidad
 * de pantalla (eje Z de Three).
 */
export function backendToThreePosition(
  position: [number, number, number]
): [number, number, number] {
  const [x, y, z] = position;
  const scale = WORLD_SCALE;
  return [x * scale, z * scale, y * scale];
}

export interface CylinderTransform {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  length: number;
}

const _up = new THREE.Vector3(0, 1, 0);

/** Local Z follows the edge, local Y stays upright, including vertical shafts. */
export function galleryBetween(a: [number, number, number], b: [number, number, number]) {
  const start = new THREE.Vector3(...a);
  const end = new THREE.Vector3(...b);
  const forward = end.clone().sub(start);
  const length = forward.length();
  forward.normalize();
  if (length < 1e-6) forward.set(0, 0, 1);
  const reference = Math.abs(forward.y) > 0.99
    ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
  const right = reference.clone().cross(forward).normalize();
  const up = forward.clone().cross(right).normalize();
  return {
    position: start.add(end).multiplyScalar(0.5).toArray() as [number, number, number],
    quaternion: new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, forward)),
    length,
  };
}

/** Orienta un CylinderGeometry (alineado por defecto al eje Y) para que
 * conecte el punto `a` con el punto `b`. */
export function cylinderBetween(
  a: [number, number, number],
  b: [number, number, number]
): CylinderTransform {
  const start = new THREE.Vector3(...a);
  const end = new THREE.Vector3(...b);
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

  const quaternion = new THREE.Quaternion();
  if (length > 1e-6) {
    quaternion.setFromUnitVectors(_up, direction.clone().normalize());
  }

  return {
    position: [mid.x, mid.y, mid.z],
    quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
    length,
  };
}

export function lerpPosition(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}
