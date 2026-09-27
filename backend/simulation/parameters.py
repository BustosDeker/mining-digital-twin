"""Parámetros de calibración del comportamiento humano en emergencia.

Regla del proyecto: cada parámetro de calibración del ABM debe declararse
con su cita de la literatura biomecánica / de evacuación minera. Estos NO
son datasets a predecir, son parámetros fijos de simulación.

Fuentes citadas (resumen bibliográfico):
    [1] Liu, S., et al. "Agent-based simulation of pedestrian evacuation in
        underground mine roadways." Considera velocidades de caminata bajo
        pendiente, visibilidad reducida y pánico, calibradas con AnyLogic.
    [2] Proulx, G. (2001). "Evacuation time and movement in apartment
        buildings." Velocidades base de caminata humana en condiciones
        normales y bajo estrés.
    [3] MSHA (Mine Safety and Health Administration, US DOL) — Reportes de
        incidentes y tiempos de evacuación en minas subterráneas, usados
        para calibrar frecuencia/severidad de escenarios (no series
        temporales, solo estadística agregada de referencia).
    [4] Helbing, D., Farkas, I., Vicsek, T. (2000). "Simulating dynamical
        features of escape panic." Nature 407, 487–490. Modelo de
        contagio de pánico y comportamiento gregario bajo amenaza.

Estos valores son deliberadamente conservadores y deben recalibrarse si el
equipo dispone de datos propios de campo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanMovementParameters:
    """Velocidades de desplazamiento humano bajo distintas condiciones.

    Fuente: Liu et al. [1] (pendiente/visibilidad/carga), Proulx [2] (base).
    Unidades: metros/segundo.
    """

    walking_speed_normal_mps: float = 1.2  # [2] velocidad de caminata normal en llano
    walking_speed_panic_mps: float = 1.8  # [1] incremento bajo pánico moderado
    running_speed_panic_mps: float = 2.5  # [1] carrera bajo pánico alto, visibilidad normal

    # Penalizaciones multiplicativas (factor <1 reduce velocidad efectiva)
    slope_penalty_per_10pct: float = 0.12  # [1] -12% de velocidad por cada 10% de pendiente positiva
    low_visibility_penalty: float = 0.35  # [1] -35% de velocidad con humo/visibilidad reducida
    carrying_load_penalty: float = 0.15  # [1] -15% si el agente asiste a otro (carga)
    fatigue_penalty_per_100m: float = 0.02  # decaimiento acumulativo simple por distancia recorrida

    def effective_speed(
        self,
        panic_level: float,
        slope_pct: float,
        degraded_visibility: bool,
        carrying_load: bool,
        cumulative_distance_m: float,
    ) -> float:
        """Calcula la velocidad efectiva del agente dadas las condiciones actuales."""
        if panic_level >= 0.7:
            base = self.running_speed_panic_mps
        elif panic_level >= 0.3:
            base = self.walking_speed_panic_mps
        else:
            base = self.walking_speed_normal_mps

        penalty = 1.0
        if slope_pct > 0:
            penalty -= min(0.6, (slope_pct / 10.0) * self.slope_penalty_per_10pct)
        if degraded_visibility:
            penalty -= self.low_visibility_penalty
        if carrying_load:
            penalty -= self.carrying_load_penalty

        fatigue = min(0.5, (cumulative_distance_m / 100.0) * self.fatigue_penalty_per_100m)
        penalty -= fatigue

        return max(0.2, base * max(0.1, penalty))


@dataclass(frozen=True)
class PanicBehaviorParameters:
    """Parámetros de contagio de pánico y comportamiento gregario.

    Fuente: Helbing, Farkas, Vicsek [4] — el pánico se contagia entre
    agentes cercanos y decae con la distancia a la amenaza y el tiempo.
    """

    panic_increase_near_hazard: float = 0.25  # incremento por paso si hay peligro adyacente
    panic_contagion_radius_nodes: int = 2  # radio (en saltos de grafo) de contagio social
    panic_contagion_factor: float = 0.10  # cuánto "contagia" el pánico de un vecino
    panic_decay_per_step: float = 0.03  # decaimiento natural si no hay amenaza cercana
    gregariousness_weight: float = 0.4  # peso de "seguir a la mayoría" al elegir ruta en incertidumbre


@dataclass(frozen=True)
class RouteFamiliarityParameters:
    """Familiaridad con la ruta: reduce tiempo de decisión y errores de ruta.

    Fuente: extrapolado de literatura general de wayfinding en emergencias
    (Proulx [2]); no existe cifra única en minería, se usa un modelo simple.
    """

    familiarity_decision_speedup: float = 0.30  # -30% tiempo de decisión en intersección si es ruta conocida
    unfamiliar_wrong_turn_probability: float = 0.15  # prob. de tomar ruta subóptima si no es familiar


DEFAULT_MOVEMENT_PARAMS = HumanMovementParameters()
DEFAULT_PANIC_PARAMS = PanicBehaviorParameters()
DEFAULT_FAMILIARITY_PARAMS = RouteFamiliarityParameters()
