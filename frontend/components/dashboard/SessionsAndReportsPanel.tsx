"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { useI18n } from "@/contexts/I18nContext";
import type { SessionHistoryRecord } from "@/lib/types";

export function SessionHistoryPanel() {
  const { t } = useI18n();
  const [records, setRecords] = useState<SessionHistoryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listSessionHistory()
      .then(setRecords)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, []);

  return (
    <div className="panel-scroll h-full overflow-y-auto p-6">
      <h2 className="mb-4 text-[12px] font-semibold uppercase tracking-wide text-steel2">
        {t.sessions.title}
      </h2>
      {error && <p className="text-[12px] text-red">{error}</p>}
      {records.length === 0 ? (
        <p className="text-[12px] text-steel">{t.sessions.empty}</p>
      ) : (
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-hairline text-left text-steel">
              <th className="py-2 font-medium">{t.sessions.scenario}</th>
              <th className="py-2 font-medium">{t.sessions.router}</th>
              <th className="py-2 font-medium">{t.sessions.status}</th>
              <th className="py-2 font-medium">{t.sessions.step}</th>
              <th className="py-2 font-medium">{t.sessions.date}</th>
            </tr>
          </thead>
          <tbody className="font-mono text-steel2">
            {records.map((r) => (
              <tr key={r.session_id} className="border-b border-hairline/50">
                <td className="py-2">{r.scenario_name}</td>
                <td className="py-2">{r.router_name}</td>
                <td className="py-2">{r.status}</td>
                <td className="py-2">{r.step}</td>
                <td className="py-2 text-steel">
                  {new Date(r.persisted_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function ReportsPanel() {
  const { t } = useI18n();
  const [reports, setReports] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listReports()
      .then(setReports)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, []);

  return (
    <div className="panel-scroll h-full overflow-y-auto p-6">
      <h2 className="mb-4 text-[12px] font-semibold uppercase tracking-wide text-steel2">
        {t.reports.title}
      </h2>
      {error && <p className="text-[12px] text-red">{error}</p>}
      {reports.length === 0 ? (
        <p className="text-[12px] text-steel">{t.reports.empty}</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {reports.map((filename) => (
            <li
              key={filename}
              className="flex items-center justify-between rounded-sm border border-hairline p-2 text-[12px]"
            >
              <span className="font-mono text-steel2">{filename}</span>
              <a
                href={api.reportDownloadUrl(filename)}
                target="_blank"
                rel="noreferrer"
                className="rounded-sm border border-signal/40 px-2 py-1 text-[11px] text-signal hover:bg-signal/10"
              >
                {t.reports.download}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
