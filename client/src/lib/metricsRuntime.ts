export interface DatasetSummary {
  dataset: string;
  metrics: Record<string, number>;
}

export interface ModelMetrics {
  model: string;
  datasets: DatasetSummary[];
}

export interface LoadedRunMetrics {
  source: string;
  models: ModelMetrics[];
  /** Flat view for backward compat — first model or merged */
  datasets: DatasetSummary[];
}

const DEFAULT_CANDIDATES = ["/data/metrics.json", "/metrics.json"] as const;

let cachedMetricsPromise: Promise<LoadedRunMetrics | null> | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseSummary(summary: unknown): DatasetSummary[] | null {
  if (!isRecord(summary)) {
    return null;
  }

  const datasets: DatasetSummary[] = [];

  for (const [dataset, rawMetrics] of Object.entries(summary)) {
    if (!isRecord(rawMetrics)) {
      continue;
    }

    const metrics: Record<string, number> = {};
    for (const [metricKey, metricValue] of Object.entries(rawMetrics)) {
      if (typeof metricValue === "number" && Number.isFinite(metricValue)) {
        metrics[metricKey] = metricValue;
      }
    }

    if (Object.keys(metrics).length > 0) {
      datasets.push({ dataset, metrics });
    }
  }

  return datasets.length > 0 ? datasets : null;
}

function normalizePayload(payload: unknown): {
  models: ModelMetrics[];
  datasets: DatasetSummary[];
} | null {
  if (!isRecord(payload)) {
    return null;
  }

  // Multi-model format: { models: { ModelName: { summary: {...} } } }
  if (isRecord(payload.models)) {
    const models: ModelMetrics[] = [];

    for (const [modelName, modelData] of Object.entries(payload.models)) {
      if (!isRecord(modelData) || !isRecord(modelData.summary)) {
        continue;
      }

      const datasets = parseSummary(modelData.summary);
      if (datasets) {
        models.push({ model: modelName, datasets });
      }
    }

    if (models.length > 0) {
      return { models, datasets: models[0].datasets };
    }
  }

  // Legacy single-model format: { summary: {...} }
  if (isRecord(payload.summary)) {
    const datasets = parseSummary(payload.summary);
    if (datasets) {
      return { models: [{ model: "default", datasets }], datasets };
    }
  }

  return null;
}

function candidateSources(): string[] {
  if (typeof window === "undefined") {
    return [...DEFAULT_CANDIDATES];
  }

  const base = import.meta.env.BASE_URL ?? "/";
  const withBase = DEFAULT_CANDIDATES.map(
    (c) => `${base.replace(/\/$/, "")}${c}`,
  );

  const querySource = new URL(window.location.href).searchParams.get("metrics");
  if (!querySource) {
    return [...withBase, ...DEFAULT_CANDIDATES];
  }

  return [querySource, ...withBase, ...DEFAULT_CANDIDATES];
}

async function fetchRunMetrics(): Promise<LoadedRunMetrics | null> {
  for (const source of candidateSources()) {
    try {
      const response = await fetch(source, { cache: "no-store" });
      if (!response.ok) {
        continue;
      }

      const payload: unknown = await response.json();
      const result = normalizePayload(payload);
      if (!result) {
        continue;
      }

      return { source, ...result };
    } catch {
      continue;
    }
  }

  return null;
}

export async function loadRunMetrics(): Promise<LoadedRunMetrics | null> {
  if (!cachedMetricsPromise) {
    cachedMetricsPromise = fetchRunMetrics();
  }
  return cachedMetricsPromise;
}

export function metricValue(
  metrics: Record<string, number>,
  key: string,
): number | null {
  if (typeof metrics[key] === "number") {
    return metrics[key];
  }

  const meanKey = `${key}_mean`;
  if (typeof metrics[meanKey] === "number") {
    return metrics[meanKey];
  }

  return null;
}

export function metricSem(
  metrics: Record<string, number>,
  key: string,
): number | null {
  const semKey = `${key}_sem`;
  if (typeof metrics[semKey] === "number") {
    return metrics[semKey];
  }
  return null;
}
