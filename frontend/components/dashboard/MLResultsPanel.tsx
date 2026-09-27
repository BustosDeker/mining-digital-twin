"use client";

import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import { api, ApiError } from "@/lib/api";
import { useI18n } from "@/contexts/I18nContext";
import type {
  ActiveModelResponse,
  DatasetInfo,
  EDASummary,
  ModelMetadata,
  RoutingComparisonResponse,
} from "@/lib/types";

export function MLResultsPanel() {
  const { t } = useI18n();

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [activeModel, setActiveModel] = useState<ActiveModelResponse | null>(null);
  const [edaSummary, setEdaSummary] = useState<EDASummary | null>(null);
  const [edaArtifacts, setEdaArtifacts] = useState<string[]>([]);
  const [edaDataset, setEdaDataset] = useState<string>("wesad_synthetic_demo");
  const [loadingEDA, setLoadingEDA] = useState(false);
  const [routingResult, setRoutingResult] = useState<RoutingComparisonResponse | null>(null);
  const [loadingRouting, setLoadingRouting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [ds, ms] = await Promise.all([api.listDatasets(), api.listModels()]);
      setDatasets(ds);
      setModels(ms);
      try {
        setActiveModel(await api.getActiveModel());
      } catch {
        setActiveModel(null);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleRunEDA = async () => {
    setLoadingEDA(true);
    setError(null);
    try {
      const summary = await api.runEDA(edaDataset, 5);
      setEdaSummary(summary);
      setEdaArtifacts(await api.listEDAArtifacts(edaDataset));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoadingEDA(false);
    }
  };

  const handleActivate = async (architecture: string, versionId: string) => {
    await api.activateModel(architecture, versionId);
    await refresh();
  };

  const handleRoutingComparison = async () => {
    setLoadingRouting(true);
    setError(null);
    try {
      const result = await api.compareRoutingStrategies({
        scenario_name: "ml_panel_comparison",
        router_names: ["adaptive_astar", "q_learning"],
        n_runs: 15,
        n_agents: 15,
        hazard_type: "fire",
        q_learning_training_episodes: 60,
        layout_config: {
          n_levels: 3,
          galleries_per_level: 8,
          n_refuge_chambers: 3,
          n_exits: 2,
          n_risk_zones: 4,
          random_seed: 42,
        },
      });
      setRoutingResult(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoadingRouting(false);
    }
  };

  return (
    <div className="panel-scroll flex h-full flex-col gap-6 overflow-y-auto p-6">
      {error && (
        <div className="rounded-sm border border-red/40 bg-red/10 px-3 py-2 text-[12px] text-red">
          {error}
        </div>
      )}

      {/* Datasets + EDA */}
      <section>
        <h2 className="mb-3 text-[12px] font-semibold uppercase tracking-wide text-steel2">
          {t.ml.datasets}
        </h2>
        <div className="mb-3 flex flex-wrap gap-2">
          {datasets.map((d) => (
            <button
              key={d.dataset_name}
              onClick={() => setEdaDataset(d.dataset_name)}
              className={clsx(
                "rounded-sm border px-3 py-2 text-left text-[11px]",
                edaDataset === d.dataset_name
                  ? "border-signal/50 bg-signal/5"
                  : "border-hairline"
              )}
            >
              <div className="font-mono font-medium text-steel2">{d.dataset_name}</div>
              <div className="mt-0.5 text-steel">
                {d.is_available_locally ? t.ml.available : t.ml.unavailable}
                {d.is_synthetic && (
                  <span className="ml-1 text-amber">· {t.ml.synthetic}</span>
                )}
              </div>
            </button>
          ))}
        </div>
        <button
          onClick={handleRunEDA}
          disabled={loadingEDA}
          className="rounded-sm border border-signal/40 px-3 py-1.5 text-[12px] font-medium text-signal hover:bg-signal/10 disabled:opacity-50"
        >
          {loadingEDA ? t.common.loading : t.ml.runEDA}
        </button>

        {edaSummary && (
          <div className="mt-4 rounded-sm border border-hairline p-3">
            <div className="mb-2 font-mono text-[11px] text-steel">
              n_subjects={edaSummary.quality_report.n_subjects} · flatline%=
              {edaSummary.quality_report.mean_flatline_pct_across_channels.toFixed(2)}
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {edaArtifacts
                .filter((f) => f.endsWith(".png"))
                .map((file) => (
                  <img
                    key={file}
                    src={api.edaArtifactUrl(edaDataset, file)}
                    alt={file}
                    className="rounded-sm border border-hairline"
                  />
                ))}
            </div>
          </div>
        )}
      </section>

      {/* Modelos */}
      <section>
        <h2 className="mb-3 text-[12px] font-semibold uppercase tracking-wide text-steel2">
          {t.ml.models}
        </h2>
        {activeModel && (
          <div className="mb-3 rounded-sm border border-signal/30 bg-signal/5 p-3 text-[11px]">
            <div className="font-medium text-signal">{t.ml.activeModel}</div>
            <div className="mt-1 font-mono text-steel2">
              {activeModel.pointer.architecture_name} · {activeModel.pointer.version_id}
            </div>
          </div>
        )}
        <div className="flex flex-col gap-2">
          {models.map((m) => (
            <div
              key={`${m.architecture_name}-${m.version_id}`}
              className="flex items-center justify-between rounded-sm border border-hairline p-2 text-[11px]"
            >
              <div>
                <div className="font-mono text-steel2">
                  {m.architecture_name} · {m.version_id}
                </div>
                <div className="text-steel">
                  f1_macro={m.cv_metrics.mean_f1_macro?.toFixed?.(4) ?? "—"}
                </div>
              </div>
              <button
                onClick={() => handleActivate(m.architecture_name, m.version_id)}
                className="rounded-sm border border-hairline px-2 py-1 text-steel hover:text-steel2"
              >
                {t.ml.activate}
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Comparación de enrutamiento */}
      <section>
        <h2 className="mb-3 text-[12px] font-semibold uppercase tracking-wide text-steel2">
          {t.ml.routingComparison}
        </h2>
        <button
          onClick={handleRoutingComparison}
          disabled={loadingRouting}
          className="rounded-sm border border-signal/40 px-3 py-1.5 text-[12px] font-medium text-signal hover:bg-signal/10 disabled:opacity-50"
        >
          {loadingRouting ? t.common.loading : t.ml.runComparison}
        </button>

        {routingResult && (
          <div className="mt-3 rounded-sm border border-hairline p-3 text-[11px]">
            {Object.entries(routingResult.per_router_summary).map(([name, summary]) => (
              <div key={name} className="mb-1.5 flex justify-between font-mono">
                <span className="text-steel2">{name}</span>
                <span className="text-steel">
                  evac={summary.mean_evacuation_rate.toFixed(3)} · t=
                  {summary.mean_evacuation_time_steps.toFixed(1)}
                </span>
              </div>
            ))}
            <div className="mt-2 border-t border-hairline pt-2 text-signal">
              {t.ml.statisticallyBest}:{" "}
              {routingResult.statistical_decision.statistically_justified_best}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
