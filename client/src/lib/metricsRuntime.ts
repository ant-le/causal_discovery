export interface DatasetSummary {
  dataset: string;
  metrics: Record<string, number>;
}

export interface LoadedRunMetrics {
  source: string;
  datasets: DatasetSummary[];
}

const DEFAULT_CANDIDATES = ["/data/metrics.json", "/metrics.json"] as const;

let cachedMetricsPromise: Promise<LoadedRunMetrics | null> | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeSummary(payload: unknown): DatasetSummary[] | null {
  if (!isRecord(payload) || !isRecord(payload.summary)) {
    return null;
  }

  const summary = payload.summary;
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

function candidateSources(): string[] {
  if (typeof window === "undefined") {
    return [...DEFAULT_CANDIDATES];
  }

  const querySource = new URL(window.location.href).searchParams.get("metrics");
  if (!querySource) {
    return [...DEFAULT_CANDIDATES];
  }

  return [querySource, ...DEFAULT_CANDIDATES];
}

async function fetchRunMetrics(): Promise<LoadedRunMetrics | null> {
  for (const source of candidateSources()) {
    try {
      const response = await fetch(source, { cache: "no-store" });
      if (!response.ok) {
        continue;
      }

      const payload: unknown = await response.json();
      const datasets = normalizeSummary(payload);
      if (!datasets) {
        continue;
      }

      return { source, datasets };
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
