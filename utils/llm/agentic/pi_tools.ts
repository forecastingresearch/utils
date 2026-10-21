// FRI Forecaster: the pi extension. Generic over what is being forecast.
//
// Factored out of code/pi_harness/redlines_tools.ts (2026-10-05). The
// driver (harness.py) writes $FORECASTER_OUT/spec.json:
//
//   tool_cli        Python argument prefix for the research-tool module
//   research_tools  which of web_search / read_page to offer ([] = no research)
//   min_research    research calls that must return before submitting
//   tools           {name: {description, parameters?}} for every tool offered;
//                   `parameters`, a JSON schema, replaces the built-in one
//   cell_schema     JSON schema of ONE forecast cell
//   key_fields      fields of a cell that identify it, e.g. ["question_id","horizon"]
//   allowed_keys    every wanted key, fields joined by "|"
//   final_extra     {name: JSON schema} extra properties of submit_forecast
//
// Optional, each off by default (what the ForecastBench driver runs). AIRO's
// loop (redlines.llm.call_tools) sets them all:
//
//   floor_counts       "successful" (default): only research calls without an
//                      error count toward min_research; "returned": every call
//   coverage           question-group keys: submission waits for a web_search
//                      with `cause` = each key and `recent_days` set
//   withhold_final     the submit tools are not declared until the floor and
//                      coverage are met (default: declared, and refused)
//   warn_missing_keys  false: accept the final answer as sent (default true:
//                      one warning for unknown or missing keys)
//   max_turns          the turn limit: from the last turn, only
//                      submit_forecast is offered and the model is asked for it
//                      in words, for up to two more turns; then no tools. Also
//                      turns a reply in prose into a nudge to research or submit
//   debrief            {tool: {name, description, parameters}, prompt, retries,
//                      validator?: {file, function, kwargs}}: after the final
//                      answer, one more user turn offering only the debrief
//                      tool; deliveries merge, the validator's problems go back
//                      for up to `retries` more tries; written to debrief.json
//
// Imports match the pi the harness is pinned to (harness.py
// PINNED_PI_VERSION; package.json: @earendil-works/pi-coding-agent 0.99.2).
//
// Tools: the research tools shell out to the Python implementations
// (tools.py, via the tool_cli module) so every model shares one search
// stack; submit_cells appends pieces to pieces.jsonl; submit_forecast writes
// final.json. The driver merges them.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFileSync } from "node:child_process";
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";

const OUT = process.env.FORECASTER_OUT;
if (!OUT) throw new Error("FORECASTER_OUT is not set (the driver sets it)");
const PY = process.env.FORECASTER_PY || "python3";
const spec = JSON.parse(readFileSync(`${OUT}/spec.json`, "utf8"));
const keyFields: string[] = spec.key_fields;
const allowed = new Set<string>(spec.allowed_keys);
const minResearch: number = spec.min_research;
const countsAll = spec.floor_counts === "returned";
const coverage: string[] = spec.coverage || [];
const withholdFinal = !!spec.withhold_final;
const warnMissing = spec.warn_missing_keys !== false;
const maxTurns: number = spec.max_turns || 0;
const debrief = spec.debrief || null;
let research = 0;        // research calls that returned without an error
let researchFailed = 0;  // ...and those that did: they count toward the floor only under floor_counts "returned"
const covered = new Set<string>();
const received = new Map<string, any>();
const cap = (s: string, n = 8000) => (s.length > n ? s.slice(0, n) : s);
const keyOf = (f: any) => keyFields.map((k) => String(f?.[k])).join("|");
const counted = () => (countsAll ? research + researchFailed : research);
const uncovered = () => coverage.filter((k) => !covered.has(k));
let forcing = false;     // past the turn limit: submit_forecast is offered regardless
const unlocked = () => forcing || (counted() >= minResearch && uncovered().length === 0);

function runPy(args: string[], input?: string) {
  return JSON.parse(execFileSync(PY, [...spec.tool_cli, ...args],
    { input, encoding: "utf8", maxBuffer: 64 * 1024 * 1024, env: process.env }));
}
const py = (tool: string, args: any) => runPy([tool, JSON.stringify(args)]);
// A debrief can outgrow one argv string (128 KB on Linux): it goes on stdin.
const validated = (value: any): string[] =>
  (runPy(["validate"], JSON.stringify({ ...debrief.validator, value })).problems || []).map(String);

// A refused call: the model reads `body` as the tool's result.
function refuse(body: any) {
  return { content: [{ type: "text" as const, text: JSON.stringify(body) }], details: {}, isError: true };
}

function notYet(name: string) {
  const parts: string[] = [];
  if (counted() < minResearch) {
    parts.push(countsAll
      ? `${minResearch - counted()} more research call(s) must return first (${counted()} of ${minResearch} so far)`
      : `${minResearch - research} more successful research call(s) must return first (${research} of ${minResearch} so far; ${researchFailed} failed)`);
  }
  const missingCover = uncovered();
  if (missingCover.length) {
    parts.push(`a web_search with recent_days set and cause=<group> is still needed for: ${missingCover.join(", ")}`);
  }
  return refuse({ error: `${name} is not offered yet: ${parts.join("; ")}` });
}

function missing() {
  return [...allowed].filter((k) => !received.has(k));
}

function say(text: string) {
  return { type: "custom_message" as const, customType: "forecaster", content: text, display: true };
}

let finalDone = false;   // one final answer per call
let finalWarned = false; // one chance to fix unknown/missing keys, then accepted as is
const cells = Type.Array(Type.Unsafe<any>(spec.cell_schema));
const paramsOf = (name: string, builtin: any) =>
  spec.tools[name]?.parameters ? Type.Unsafe<any>(spec.tools[name].parameters) : builtin;

export default function (pi: ExtensionAPI) {
  const research_tools: string[] = spec.research_tools;
  const submitTools = [...(spec.tools.submit_cells ? ["submit_cells"] : []), "submit_forecast"];
  const offer = () => pi.setActiveTools(unlocked() || !withholdFinal ? [...research_tools, ...submitTools] : research_tools);

  function researched(params: any, r: any, isSearch: boolean) {
    const ok = r && !r.error;
    if (ok) research += 1; else researchFailed += 1;
    if (isSearch && params?.cause != null && params?.recent_days && (ok || countsAll)) covered.add(String(params.cause));
    if (withholdFinal) offer();
  }

  if (research_tools.includes("web_search")) pi.registerTool({
    name: "web_search", label: "web_search",
    description: spec.tools.web_search.description,
    parameters: paramsOf("web_search", Type.Object({
      query: Type.String(),
      max_results: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
      recent_days: Type.Optional(Type.Integer({ minimum: 1 })),
    })),
    async execute(_id, params) {
      const r = py("web_search", params);
      researched(params, r, true);
      return { content: [{ type: "text", text: cap(JSON.stringify(r)) }], details: r };
    },
  });
  if (research_tools.includes("read_page")) pi.registerTool({
    name: "read_page", label: "read_page",
    description: spec.tools.read_page.description,
    parameters: paramsOf("read_page", Type.Object({ url: Type.String(), offset: Type.Optional(Type.Integer({ minimum: 0 })) })),
    async execute(_id, params) {
      const r = py("read_page", params);
      researched(params, r, false);
      return { content: [{ type: "text", text: cap(JSON.stringify(r)) }], details: r };
    },
  });
  if (spec.tools.submit_cells) pi.registerTool({
    name: "submit_cells", label: "submit_cells",
    description: spec.tools.submit_cells.description,
    parameters: paramsOf("submit_cells", Type.Object({ forecasts: cells })),
    async execute(_id, params: any) {
      if (!unlocked()) return notYet("submit_cells");
      let n = 0;
      for (const f of params.forecasts || []) {
        if (f && typeof f === "object" && allowed.has(keyOf(f))) {
          received.set(keyOf(f), f);
          n += 1;
        }
      }
      appendFileSync(`${OUT}/pieces.jsonl`, JSON.stringify({ t: Date.now(), accepted: n, forecasts: params.forecasts }) + "\n");
      const miss = missing();
      const r = { accepted: n, received: received.size, of: allowed.size, missing: miss.length, missing_first: miss.slice(0, 20) };
      return { content: [{ type: "text", text: JSON.stringify(r) }], details: r };
    },
  });
  const finalProps: Record<string, any> = {
    forecasts: Type.Optional(cells),
    rationale: Type.String(),
    key_sources: Type.Optional(Type.Array(Type.String())),
  };
  for (const [k, s] of Object.entries(spec.final_extra || {})) finalProps[k] = Type.Unsafe<any>(s);
  pi.registerTool({
    name: "submit_forecast", label: "submit_forecast",
    description: spec.tools.submit_forecast.description,
    parameters: paramsOf("submit_forecast", Type.Object(finalProps)),
    async execute(_id, params: any) {
      if (!unlocked()) return notYet("submit_forecast");
      if (finalDone) return refuse({ error: "submit_forecast was already called; stop here." });
      const unknown = (params.forecasts || []).filter((f: any) => !allowed.has(keyOf(f))).map(keyOf);
      const have = new Set([...received.keys(), ...(params.forecasts || []).map(keyOf).filter((k: string) => allowed.has(k))]);
      const stillMissing = [...allowed].filter((k) => !have.has(k));
      if (warnMissing && (unknown.length || stillMissing.length) && !finalWarned) {
        finalWarned = true;
        return refuse({ error: "not submitted: fix and call submit_forecast again", unknown_keys: unknown, missing_keys: stillMissing });
      }
      finalDone = true;
      writeFileSync(`${OUT}/final.json`, JSON.stringify({ t: Date.now(), research, research_failed: researchFailed, forced: forcing, answer: params }, null, 1));
      const r = { ok: true, received: received.size, of: allowed.size, in_final: (params.forecasts || []).length };
      return { content: [{ type: "text", text: JSON.stringify(r) + (debrief ? "" : " Done: stop here.") }], details: r };
    },
  });

  // The debrief: deliveries merge; the merged whole is validated.
  const d = { result: null as any, complete: false, attempts: 0, problems: [] as string[], error: null as string | null };
  let phase: "forecast" | "debrief" | "done" = "forecast";
  let debriefCalled = false;
  const saveDebrief = () => writeFileSync(`${OUT}/debrief.json`, JSON.stringify(d, null, 1));
  const parsed = (args: any) => {
    if (!args || typeof args !== "object" || Array.isArray(args)) return args;
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(args)) {
      let x: any = v;
      if (typeof v === "string" && v.trim().startsWith("{")) { try { x = JSON.parse(v); } catch { /* not JSON: kept */ } }
      out[k] = x;
    }
    return out;
  };
  if (debrief) pi.registerTool({
    name: debrief.tool.name, label: debrief.tool.name,
    description: debrief.tool.description,
    parameters: Type.Unsafe<any>(debrief.tool.parameters),
    async execute(_id, params: any) {
      if (phase !== "debrief") return refuse({ error: `${debrief.tool.name} is offered after submit_forecast` });
      debriefCalled = true;
      const args = parsed(params);
      d.result = d.result && typeof d.result === "object" && args && typeof args === "object" ? { ...d.result, ...args } : args;
      d.error = null;
      try {
        d.problems = debrief.validator ? validated(d.result) : [];
      } catch (e: any) {
        d.error = `validator failed: ${String(e?.message || e).slice(0, 300)}`;
        d.problems = [d.error];
      }
      d.complete = d.error === null && d.problems.length === 0;
      saveDebrief();
      const r = { accepted: d.complete, problems: d.problems };
      return { content: [{ type: "text", text: cap(JSON.stringify(r)) + (d.complete ? " Done: stop here." : "") }], details: r };
    },
  });

  let turns = 0;
  const limit = maxTurns ? maxTurns + 2 : 0;
  const askFinal = "Now call submit_forecast with your final answer. Include any remaining forecast cells and required summary fields; previously accepted cells are retained.";

  pi.on("session_start", async () => {
    if (debrief) pi.setActiveTools(pi.getActiveTools().filter((t) => t !== debrief.tool.name));
    if (withholdFinal) offer();
  });

  pi.on("turn_end", async () => {
    turns += 1;
    if (phase === "forecast" && finalDone) {
      if (!debrief) { phase = "done"; return; }
      phase = "debrief";
      pi.setActiveTools([debrief.tool.name]);
      saveDebrief();
      return { entries: [say(debrief.prompt)], continue: true };
    }
    if (phase === "debrief") {
      d.attempts += 1;
      const called = debriefCalled;
      debriefCalled = false;
      if (!called) d.problems = [`${debrief.tool.name} was not called`];
      saveDebrief();
      if (d.complete || d.attempts >= 1 + (debrief.retries ?? 1)) {
        phase = "done";
        pi.setActiveTools([]);
        return;
      }
      // Not called (prose, or a refusal, which cannot continue from here): the
      // run is settling, and agent_before_settle asks again.
      if (!called) return;
      return { entries: [say(`Call ${debrief.tool.name} again with what is still missing or wrong (entries you already sent are kept), fixing: ${d.problems.join("; ").slice(0, 4000)}`)], continue: true };
    }
    if (phase === "forecast" && maxTurns) {
      if (turns >= limit) { pi.setActiveTools([]); return; }
      if (turns >= maxTurns - 1) {
        forcing = true;
        pi.setActiveTools(["submit_forecast"]);
        return { entries: [say(askFinal)], continue: true };
      }
    }
  });

  // A reply in prose (no tool call, or a provider refusal) ends pi's run; in the
  // debrief, and under max_turns, nudge instead.
  pi.on("agent_before_settle", async () => {
    if (phase === "debrief") return { entries: [say(`Now call ${debrief.tool.name} with the debrief.`)], continue: true };
    if (phase !== "forecast" || finalDone || !maxTurns || turns >= limit || forcing) return;
    const missingCover = uncovered();
    return { entries: [say(unlocked()
      ? "Now call submit_forecast with your final answer."
      : `Keep researching with the tools: submit_forecast is offered after ${minResearch} tool calls have returned (${counted()} so far)` +
        (missingCover.length ? ` and after a web_search with recent_days on each of: ${missingCover.join(", ")}` : "") + ".")], continue: true };
  });
}
