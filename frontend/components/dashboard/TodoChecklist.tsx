"use client";

import { useState } from "react";
import clsx from "clsx";

const tasks = [
  "Crear controles de vista útiles",
  "Conectar filtros con el grafo",
  "Reorganizar panel operativo",
  "Validar build y experiencia",
];

export function TodoChecklist({ open = true, onClose }: { open?: boolean; onClose?: () => void }) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  if (!open) return null;

  return (
    <div className="absolute left-4 top-4 z-30 w-[330px] rounded-md border border-[#1a73d6]/80 bg-[#050d12]/95 shadow-[0_0_0_1px_rgba(90,150,255,0.08),0_14px_30px_rgba(0,0,0,0.45)]">
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-2.5">
        <button
          type="button"
          onClick={() => setSelectedIndex((prev) => (prev + 1) % tasks.length)}
          className="flex items-center gap-2 text-left text-[13px] font-medium text-slate-100"
          aria-label="Alternar tareas"
        >
          <span className="text-[12px] text-sky-300">⌄</span>
          <span>Todos ({selectedIndex + 1}/{tasks.length})</span>
        </button>

        <button
          type="button"
          onClick={onClose}
          className="inline-flex h-6 w-6 items-center justify-center rounded-sm border border-white/10 text-[15px] text-slate-300 transition hover:border-sky-500/60 hover:text-sky-200"
          aria-label="Cerrar checklist"
        >
          ×
        </button>
      </div>

      <div className="px-2 py-2">
        {tasks.map((task, index) => {
          const active = selectedIndex === index;

          return (
            <button
              key={task}
              type="button"
              onClick={() => setSelectedIndex(index)}
              className={clsx(
                "flex w-full items-center gap-3 rounded-sm px-3 py-2 text-left transition-colors",
                active ? "bg-white/[0.02]" : "hover:bg-white/[0.015]"
              )}
            >
              <span
                className={clsx(
                  "mt-0.5 h-4 w-4 rounded-full border transition-all",
                  active
                    ? "border-[#4ea8ff] bg-[#2f9eff] shadow-[0_0_0_3px_rgba(61,157,255,0.14)]"
                    : "border-slate-500 bg-transparent"
                )}
                aria-hidden="true"
              />
              <span
                className={clsx(
                  "text-[13px] leading-5",
                  active ? "font-medium text-slate-50" : "text-slate-300"
                )}
              >
                {task}
              </span>
            </button>
          );
        })}
      </div>

      <div className="flex items-center justify-end gap-2 border-t border-white/10 px-2 py-2">
        <button
          type="button"
          onClick={onClose}
          className="flex h-8 w-8 items-center justify-center rounded-sm border border-white/10 bg-transparent text-[15px] text-slate-300 transition hover:border-sky-500/60 hover:text-sky-200"
          aria-label="Cerrar checklist"
        >
          ×
        </button>
        <button
          type="button"
          onClick={() => {
            if (selectedIndex < tasks.length - 1) {
              setSelectedIndex((prev) => prev + 1);
            }
          }}
          className="flex h-8 w-8 items-center justify-center rounded-sm border border-sky-500/60 bg-sky-500/10 text-[15px] text-sky-200 transition hover:bg-sky-500/20"
          aria-label="Siguiente tarea"
        >
          ✓
        </button>
      </div>
    </div>
  );
}
