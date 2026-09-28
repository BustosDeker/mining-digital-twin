export type Locale = "es" | "en";

export interface TranslationShape {
  twin: {
    operations: string;
    liveGeometry: string;
    emptyTitle: string;
    emptyDescription: string;
    fit: string;
    cancelFollow: string;
    follow: string;
    selectWorker: string;
    worker: string;
    panic: string;
    waiting: string;
    distance: string;
    capacity: string;
    width: string;
    length: string;
    slope: string;
    risk: string;
    affected: string;
    navigation: string;
    opacityHint: string;
    routesHint: string;
    webglError: string;
    webglHelp: string;
    retry: string;
    scenarioName: string;
    simulation: string;
    visualization: string;
    selected: string;
    depth: string;
    infrastructure: string;
    noSelection: string;
    status: string;
    inspect: string;
    cutaway: string;
  };
  appTitle: string;
  nav: {
    digitalTwin: string;
    mlResults: string;
    sessions: string;
    reports: string;
  };
  controls: {
    start: string;
    pause: string;
    stop: string;
    reset: string;
    step: string;
    newScenario: string;
    pan: string;
    panHint: string;
    opacity: string;
    labels: string;
    performance: string;
  };
  mapView: {
    toolbar: string;
    general: string;
    top: string;
    lateral: string;
    incidents: string;
    routes: string;
    level: string;
    allLevels: string;
  };
  scenario: {
    title: string;
    nAgents: string;
    router: string;
    routerAstar: string;
    routerQLearning: string;
    hazardType: string;
    hazardFire: string;
    hazardCollapse: string;
    hazardGasLeak: string;
    hazardNone: string;
    hazardIntensity: string;
    levels: string;
    galleries: string;
    refuges: string;
    exits: string;
    riskZones: string;
    seed: string;
    launch: string;
  };
  status: {
    ready: string;
    running: string;
    paused: string;
    stopped: string;
    finished: string;
    connected: string;
    disconnected: string;
  };
  legend: {
    title: string;
    clear: string;
    degraded: string;
    blocked: string;
    exit: string;
    refuge: string;
    riskZone: string;
    agent: string;
  };
  stressPanel: {
    title: string;
    avgPanic: string;
    evacuated: string;
    sheltered: string;
    lost: string;
    moving: string;
    step: string;
  };
  eventLog: {
    title: string;
    empty: string;
    startedAtStep: string;
  };
  ml: {
    datasets: string;
    available: string;
    unavailable: string;
    synthetic: string;
    runEDA: string;
    subjectLimit: string;
    models: string;
    activeModel: string;
    activate: string;
    architecture: string;
    version: string;
    metrics: string;
    routingComparison: string;
    runComparison: string;
    statisticallyBest: string;
  };
  sessions: {
    title: string;
    empty: string;
    scenario: string;
    router: string;
    status: string;
    step: string;
    date: string;
  };
  reports: {
    title: string;
    empty: string;
    download: string;
  };
  common: {
    loading: string;
    error: string;
    close: string;
  };
}

export const translations: Record<Locale, TranslationShape> = {
  es: {
    twin: {
      operations: "Centro de control minero",
      liveGeometry: "Topología recibida del simulador",
      emptyTitle: "Preparar la operación",
      emptyDescription: "Crea un escenario para visualizar la mina, sus trabajadores y las condiciones de evacuación.",
      fit: "Encuadrar mina",
      cancelFollow: "Cancelar seguimiento",
      follow: "Seguir trabajador",
      selectWorker: "Seleccionar trabajador",
      worker: "Trabajador",
      panic: "Pánico",
      waiting: "En espera",
      distance: "Distancia recorrida",
      capacity: "Capacidad",
      width: "Ancho",
      length: "Longitud",
      slope: "Pendiente",
      risk: "Riesgo actual",
      affected: "Galerías afectadas",
      navigation: "Arrastrar: rotar · Botón derecho: mover · Rueda: zoom",
      opacityHint: "Opacidad de la envolvente de roca; las señales permanecen visibles.",
      routesHint: "Tramos ocupados recibidos del simulador; no representa rutas futuras.",
      webglError: "No se pudo inicializar WebGL",
      webglHelp: "Abre esta misma dirección en un navegador con WebGL habilitado.",
      retry: "Reintentar",
      scenarioName: "Nombre del escenario",
      simulation: "Simulación",
      visualization: "Visualización",
      selected: "Selección",
      depth: "Cota",
      infrastructure: "Infraestructura",
      noSelection: "Selecciona una galería o señal para inspeccionar sus datos.",
      status: "Estado",
      inspect: "Inspección",
      cutaway: "Sección abierta · Dimensiones en metros",
    },
    appTitle: "Gemelo Digital de Evacuación Minera",
    nav: {
      digitalTwin: "Gemelo Digital",
      mlResults: "Motor IA",
      sessions: "Historial",
      reports: "Reportes",
    },
    controls: {
      start: "Iniciar",
      pause: "Pausar",
      stop: "Detener",
      reset: "Reiniciar",
      step: "Paso +1s",
      newScenario: "Nuevo escenario",
      pan: "Mover vista",
      panHint: "Activa y arrastra para desplazar el circuito",
      opacity: "Opacidad",
      labels: "Etiquetas",
      performance: "Rendimiento",
    },
    mapView: {
      toolbar: "Vistas del gemelo digital",
      general: "General",
      top: "Superior",
      lateral: "Lateral",
      incidents: "Incidentes",
      routes: "Rutas",
      level: "Nivel",
      allLevels: "Todos",
    },
    scenario: {
      title: "Configurar escenario",
      nAgents: "N.º de mineros",
      router: "Estrategia de enrutamiento",
      routerAstar: "Ruta adaptativa (A*)",
      routerQLearning: "Ruta aprendida (Q-learning)",
      hazardType: "Tipo de emergencia",
      hazardFire: "Incendio",
      hazardCollapse: "Colapso",
      hazardGasLeak: "Fuga de gas",
      hazardNone: "Sin emergencia",
      hazardIntensity: "Intensidad",
      levels: "Niveles",
      galleries: "Galerías por nivel",
      refuges: "Cámaras de refugio",
      exits: "Salidas",
      riskZones: "Zonas de riesgo",
      seed: "Semilla aleatoria",
      launch: "Lanzar escenario",
    },
    status: {
      ready: "Preparado",
      running: "En curso",
      paused: "Pausado",
      stopped: "Detenido",
      finished: "Finalizado",
      connected: "Gemelo digital conectado",
      disconnected: "Reconectando…",
    },
    legend: {
      title: "Leyenda",
      clear: "Transitable",
      degraded: "Degradado",
      blocked: "Bloqueado",
      exit: "Salida",
      refuge: "Refugio",
      riskZone: "Zona de riesgo",
      agent: "Minero",
    },
    stressPanel: {
      title: "Panel de estrés",
      avgPanic: "Pánico promedio",
      evacuated: "Evacuados",
      sheltered: "Refugiados",
      lost: "Perdidos",
      moving: "En tránsito",
      step: "Paso de simulación",
    },
    eventLog: {
      title: "Eventos activos",
      empty: "Sin eventos de emergencia activos",
      startedAtStep: "iniciado en el paso",
    },
    ml: {
      datasets: "Datasets",
      available: "Disponible",
      unavailable: "No disponible",
      synthetic: "SINTÉTICO — no usar en el artículo",
      runEDA: "Ejecutar EDA",
      subjectLimit: "Límite de sujetos",
      models: "Modelos registrados",
      activeModel: "Modelo activo en producción",
      activate: "Activar",
      architecture: "Arquitectura",
      version: "Versión",
      metrics: "Métricas (CV)",
      routingComparison: "Comparación de enrutamiento",
      runComparison: "Ejecutar comparación",
      statisticallyBest: "Mejor (justificado estadísticamente)",
    },
    sessions: {
      title: "Historial de sesiones",
      empty: "Aún no hay sesiones registradas.",
      scenario: "Escenario",
      router: "Router",
      status: "Estado",
      step: "Paso",
      date: "Fecha",
    },
    reports: {
      title: "Reportes PDF",
      empty: "Aún no se ha generado ningún reporte.",
      download: "Descargar",
    },
    common: {
      loading: "Cargando…",
      error: "Ocurrió un error",
      close: "Cerrar",
    },
  },
  en: {
    twin: {
      operations: "Mining operations center",
      liveGeometry: "Topology received from the simulator",
      emptyTitle: "Prepare the operation",
      emptyDescription: "Create a scenario to view the mine, its workers and evacuation conditions.",
      fit: "Fit mine",
      cancelFollow: "Stop following",
      follow: "Follow worker",
      selectWorker: "Select worker",
      worker: "Worker",
      panic: "Panic",
      waiting: "Waiting",
      distance: "Distance traveled",
      capacity: "Capacity",
      width: "Width",
      length: "Length",
      slope: "Slope",
      risk: "Current risk",
      affected: "Affected galleries",
      navigation: "Drag: orbit · Right button: pan · Wheel: zoom",
      opacityHint: "Rock envelope opacity; safety signals remain visible.",
      routesHint: "Occupied segments received from the simulator; does not show future routes.",
      webglError: "Unable to initialize WebGL",
      webglHelp: "Open this same address in a browser with WebGL enabled.",
      retry: "Retry",
      scenarioName: "Scenario name",
      simulation: "Simulation",
      visualization: "Visualization",
      selected: "Selection",
      depth: "Elevation",
      infrastructure: "Infrastructure",
      noSelection: "Select a gallery or sign to inspect its data.",
      status: "Status",
      inspect: "Inspection",
      cutaway: "Cutaway view · Dimensions in meters",
    },
    appTitle: "Mine Evacuation Digital Twin",
    nav: {
      digitalTwin: "Digital Twin",
      mlResults: "AI Engine",
      sessions: "History",
      reports: "Reports",
    },
    controls: {
      start: "Start",
      pause: "Pause",
      stop: "Stop",
      reset: "Reset",
      step: "Step +1s",
      newScenario: "New scenario",
      pan: "Pan view",
      panHint: "Activate, then drag to move around the circuit",
      opacity: "Opacity",
      labels: "Labels",
      performance: "Performance",
    },
    mapView: {
      toolbar: "Digital twin views",
      general: "General",
      top: "Top",
      lateral: "Side",
      incidents: "Incidents",
      routes: "Routes",
      level: "Level",
      allLevels: "All",
    },
    scenario: {
      title: "Configure scenario",
      nAgents: "Number of miners",
      router: "Routing strategy",
      routerAstar: "Adaptive route (A*)",
      routerQLearning: "Learned route (Q-learning)",
      hazardType: "Emergency type",
      hazardFire: "Fire",
      hazardCollapse: "Collapse",
      hazardGasLeak: "Gas leak",
      hazardNone: "No emergency",
      hazardIntensity: "Intensity",
      levels: "Levels",
      galleries: "Galleries per level",
      refuges: "Refuge chambers",
      exits: "Exits",
      riskZones: "Risk zones",
      seed: "Random seed",
      launch: "Launch scenario",
    },
    status: {
      ready: "Ready",
      running: "Running",
      paused: "Paused",
      stopped: "Stopped",
      finished: "Finished",
      connected: "Digital twin connected",
      disconnected: "Reconnecting…",
    },
    legend: {
      title: "Legend",
      clear: "Clear",
      degraded: "Degraded",
      blocked: "Blocked",
      exit: "Exit",
      refuge: "Refuge",
      riskZone: "Risk zone",
      agent: "Miner",
    },
    stressPanel: {
      title: "Stress panel",
      avgPanic: "Average panic",
      evacuated: "Evacuated",
      sheltered: "Sheltered",
      lost: "Lost",
      moving: "In transit",
      step: "Simulation step",
    },
    eventLog: {
      title: "Active events",
      empty: "No active emergency events",
      startedAtStep: "started at step",
    },
    ml: {
      datasets: "Datasets",
      available: "Available",
      unavailable: "Unavailable",
      synthetic: "SYNTHETIC — do not use in the paper",
      runEDA: "Run EDA",
      subjectLimit: "Subject limit",
      models: "Registered models",
      activeModel: "Active production model",
      activate: "Activate",
      architecture: "Architecture",
      version: "Version",
      metrics: "Metrics (CV)",
      routingComparison: "Routing comparison",
      runComparison: "Run comparison",
      statisticallyBest: "Best (statistically justified)",
    },
    sessions: {
      title: "Session history",
      empty: "No sessions recorded yet.",
      scenario: "Scenario",
      router: "Router",
      status: "Status",
      step: "Step",
      date: "Date",
    },
    reports: {
      title: "PDF Reports",
      empty: "No report has been generated yet.",
      download: "Download",
    },
    common: {
      loading: "Loading…",
      error: "An error occurred",
      close: "Close",
    },
  },
};
