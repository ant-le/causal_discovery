export interface Citation {
  id: string;
  author: string;
  title: string;
  year: string;
  url: string;
}

export interface ExplainContent {
  title: string;
  explainText: string;
}

export interface TextData {
  category?: string;
  icon?: string;
  title: string;
  content: string;
  topics?: string[];
  explanation?: ExplainContent[];
}

export interface SideBarData {
  id: string;
  contents: string[];
}

export interface SourceReferences {
  code: string[];
  notes: string[];
  docs: string[];
}

export type SourceMapData = Record<
  "Motivation" | "Methodology" | "Benchmark" | "Results" | "Appendix",
  Record<string, Record<string, SourceReferences>>
>;
