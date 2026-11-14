"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
import { Beeswarm, ClassFeatureHeatmap, ImportanceChart, LocalExplanation } from "@/components/explain";
import { FlowTable } from "@/components/FlowTable";
import { Card, ErrorState, Loading, PageHeader, Td, Th, useApi } from "@/components/ui";
import { api, CLASSES } from "@/lib/api";
import { CLASS_META, featureValue } from "@/lib/format";

function LocalSection() {
  const params = useSearchParams();
  const a = params.get("a");
  const i = params.get("i");
  const flow = useApi(() => (a && i && /^\d+$/.test(i) ? api.flow(a, Number(i)) : Promise.resolve(null)), [a, i]);
  const recent = useApi(() => (a ? Promise.resolve(null) : api.threats({ order: "severity", limit: 8 })), [a]);

  if (!a || !i) {
    return (
      <Card title="Explain an individual detection" subtitle="Pick a flow — or use “Explain” on any table row" bodyClass="p-0">
        {recent.data?.items.length ? <FlowTable flows={recent.data.items} showSource />
          : <p className="p-4 text-sm text-ink-3">Analyse some traffic first; detections you can explain will appear here.</p>}
      </Card>
    );
  }
  if (flow.error) return <ErrorState message={flow.error} />;
  if (!flow.data) return <Loading label="Computing SHAP explanation…" />;
  const f = flow.data;
  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-[15px] font-medium text-ink">
          Flow #{f.idx} <span className="text-ink-3">in</span>{" "}
          <Link href={`/analyze/${f.analysis_id}`} className="text-accent hover:underline">{f.filename}</Link>
        </h2>
        <span className="text-xs text-ink-3">
          {f.true_label && <>Supplied label: <span className="text-ink">{f.true_label}</span> · </>}
          {f.imputed > 0 && <>{f.imputed} imputed feature value(s) · </>}
          <Link href="/explain" className="hover:text-ink">Choose another flow</Link>
        </span>
      </div>
      <LocalExplanation e={f.explanation} />
    </section>
  );
}

function GlobalSection() {
  const g = useApi(api.explainability);
  const model = useApi(api.model);
  const labels = useMemo(() => Object.fromEntries((model.data?.features.selected ?? []).map((f) => [f.name, f.label])), [model.data]);
  const units = useMemo(() => Object.fromEntries((model.data?.features.selected ?? []).map((f) => [f.name, f.unit])), [model.data]);
  if (g.error) return <ErrorState message={g.error} />;
  if (!g.data) return <Loading />;
  const d = g.data;
  return (
    <>
      <div className="mt-8 mb-4">
        <h2 className="text-[15px] font-medium text-ink">Global behaviour of the model</h2>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-ink-3">
          Computed with SHAP TreeExplainer on a class-balanced sample of {d.sample.size.toLocaleString()} held-out test flows (up to {d.sample.per_class_cap} per class), in units of {d.units}.
        </p>
      </div>
      <div className="grid gap-4 xl:grid-cols-5">
        <Card title="Feature importance" subtitle="Mean |SHAP| on the predicted class" className="xl:col-span-2"><ImportanceChart g={d} /></Card>
        <Card title="Attack evidence by feature" subtitle="Each dot is a flow; position = how much the feature pushed it away from BENIGN" className="xl:col-span-3">
          <Beeswarm g={d} labels={labels} />
        </Card>
      </div>
      <Card title="Which features drive each attack class" subtitle="How attack patterns differ: mean |SHAP| of the class’s own logit over flows of that class" className="mt-4">
        <ClassFeatureHeatmap g={d} labels={labels} />
      </Card>
      <Card title="Attack profiles vs. benign traffic" subtitle="Median feature value per class on the test split (top global features)" className="mt-4" bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px]">
            <thead className="border-b border-line">
              <tr><Th>Feature</Th>{CLASSES.map((c) => <Th key={c} className="text-right">{CLASS_META[c].short}</Th>)}</tr>
            </thead>
            <tbody className="divide-y divide-line">
              {d.class_profiles.features.map((f) => {
                const base = d.class_profiles.medians.BENIGN[f];
                return (
                  <tr key={f}>
                    <Td className="text-ink">{labels[f] ?? f}</Td>
                    {CLASSES.map((c) => {
                      const v = d.class_profiles.medians[c][f];
                      const ratio = base > 0 && v > 0 ? v / base : null;
                      const off = c !== "BENIGN" && ratio != null && (ratio > 3 || ratio < 1 / 3);
                      return (
                        <Td key={c} className={`tnum text-right font-mono text-xs ${off ? "text-ink" : ""}`}>
                          {featureValue(v, units[f])}
                          {off && <span className="ml-1 text-[10px] text-ink-3">{ratio! > 1 ? `×${ratio!.toFixed(0)}` : `÷${(1 / ratio!).toFixed(0)}`}</span>}
                        </Td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="border-t border-line px-4 py-2.5 text-xs text-ink-3">Multipliers mark medians more than 3× above or below the benign median.</p>
      </Card>
    </>
  );
}

export default function ExplainPage() {
  return (
    <>
      <PageHeader title="Explainability"
        description="Why the model classified traffic the way it did. Local explanations decompose a single flow’s prediction into exact per-feature SHAP contributions; the global view shows what the model relies on across classes." />
      <Suspense fallback={<Loading />}><LocalSection /></Suspense>
      <GlobalSection />
    </>
  );
}
