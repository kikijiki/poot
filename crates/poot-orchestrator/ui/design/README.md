# Orchestrator dashboard overhaul (POOT-1359): audit and design

Mockup: `mockup.html` (rendered: `mockup.png`). Current-UI screenshots: `audit/` (taken from the shipped
binary against a copy of the live state DB, port 8799; the live 8787 instance was not touched).

## 1. UX audit of the current UI

| View                       | Screenshot                                                 | Findings                                                                                                                                                                                                                                                                                                                                 |
| -------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runs list + log            | `audit/run-exec-log.png`                                   | Cards repeat id, scenario, pod, GPU list, ssh, image digest: the info panel is mostly duplicated or noise. The run id (`run-<epoch>-<pid>`) is the title; model, GPU actually used, repeat progress, cost and ETA are absent. GPU row prints the whole candidate list comma-joined. No status colour for outcome beyond the phase badge. |
| Exec run, Results tab      | `audit/run-exec-running.png`, `run-sweep-done-results.png` | Results tab shown for runs that cannot have sets; page shows the raw JS error `E.filter is not a function` because `/api/results` returns `{error}` where the client expects an array. No explanation either way (POOT-1335).                                                                                                            |
| Sweep waiting for capacity | `audit/run-sweep-waiting.png`                              | Only the badge text "provisioning (waiting for capacity, pass 4)"; no minutes waited, price cap, next try.                                                                                                                                                                                                                               |
| Pods                       | `audit/pods.png`                                           | `zvvftp2c3m33hz` and `ed2s23b4z3zlj4` show PROVISIONING for 25 min to hours: the DB state is shown as truth. Terminated pods keep full cards. No uptime, no accumulated cost, no owner link, no verification against RunPod.                                                                                                             |
| Image builds / Serve       | `audit/images.png`, `serve.png`                            | Empty states are bare; serve target is a raw text box.                                                                                                                                                                                                                                                                                   |
| Narrow                     | `audit/narrow-runs.png`                                    | Two-pane layout collapses poorly; tabs scroll off.                                                                                                                                                                                                                                                                                       |
| Cross-cutting              |                                                            | No overview; no spend or burn rate; failures are not aggregated; no log search/copy; no run comparison; dark only; hand-rolled tab buttons without roles or keyboard handling; loading/error states are per-view ad hoc.                                                                                                                 |

Owner complaints (1)-(11) map to the changes below.

## 2. Information architecture

`?view=<overview|runs|pods|compare|images|serve>&run=<id>&tab=<summary|log|results|pod>&sub=<table|charts|failures>`
(the URL stays the single source of view state; extends `urlstate.ts`, unit-tested). Default view is **Overview**.

Pages and what each needs:

1. **Overview** (new landing): KPI strip (running, waiting, spend 24 h, burn rate/h, failures 24 h); failure banner;
   Running now; Waiting for capacity (pass, minutes waited, price cap, next try); Pods table (state, $/h, uptime,
   owner run, _verified_ column); Recent outcomes (ok / incomplete / errors / host-fault-replaced / cancelled).
2. **Runs**: master list (human title `scenario - model - GPU`, outcome pill, repeat progress, cost, started/ETA;
   filter by status/kind; id behind a copy button) + detail with tabs **Summary** (facts grid, no duplication),
   **Log** (live tail, search, copy, follow), **Results** (Table / Charts / Failures; live partial mirror while
   running; hidden for exec runs; empty state says why), **Pod**.
3. **Pods**: sortable table, same columns as Overview but all pods; terminated collapsed by default.
4. **Compare**: pick 2-6 finished runs/sets, overlay decode curves by context length per GPU; same chart components,
   direct labels, dash + colour encoding.
5. **Images**, **Serve**: kept, restyled with the shared empty/loading/error components.
6. **Scheduler slot (POOT-1358)**: a `views.ts` registry (`{id, title, icon, component, badge}`) drives nav and the
   Overview panel grid. 1358 adds a record and its components; nothing here implements queue/attempt/lease views.

Components (all under `ui/src/`): `Shell` (nav, theme toggle, live indicator, skip link), `Kpi`, `Pill` (outcome
and state with text + colour, never colour alone), `DataTable` (sort, sticky head, stacks into cards at <=640 px),
`AsyncView` (one loading/empty/error contract used by every view), `LogPane` (tail, search, copy), existing charts.
Stack unchanged: React + Vite + CSS modules, theme tokens as CSS variables with light/dark + `prefers-color-scheme`.
Accessibility: landmark roles, `role=tablist` with arrow-key navigation, focus rings, `aria-live` for the live
indicator, chart `role=img` with a text summary, contrast >= 4.5:1 in both themes.

## 3. API additions (as implemented; documented in the header of `src/web.rs`)

`/api/state` stays backward compatible (`scripts/status-board` in Backstage reads `runs[].id/status/models[].model`
and `pods[].id/gpu_type/cloud/status`): fields are only added, never renamed.

- `runs[]` gains `repeats`, `has_sets` (false for exec runs), `sets_done`/`sets_total` (repeats x models),
  `eta_secs` (extrapolated from the sets gathered, while the run has not ended), `wait`
  (`{pass, waited_min, retry_secs, price_cap}` parsed from the run log while it waits for capacity: the DB does not
  hold the price cap) and `cost_usd` (its pod's estimate).
- `pods[]` gains `uptime_secs`, `accrued_usd` (price x lifetime, an estimate) and `verified`
  (`confirmed`/`gone`/`unverified`/`terminated`): RunPod's own pod list, read-only, listed every 30 s on a
  background thread so a request never waits on the network; owner-approved 2026-10-24. A pod RunPod does not list
  stops accruing; a failed call is `unverified` with its reason, never `gone`.
- New `GET /api/overview`: `spend_24h_usd`, `burn_usd_per_hr`, `billing_pods`, `failures` (failed/aborted runs of
  the last 24 h with the last line of their log as the reason and the pod replacements counted from it) and
  `replacements_24h`.
- `GET /api/results` answers `{sets, reason}` (it used to answer a bare array, or `{error}` where the page crashed
  with `E.filter is not a function`): `reason` is `exec_run`, `no_results_dir`, `no_sets_yet`,
  `ended_without_sets` or `unknown_run` when `sets` is empty. Partial mirrors are served while the run is active.
- Host CPU comes from the sets' `env.json` (`/api/results`), the log tail from the existing `/api/log?tail=`.

Not built, from the first design: a `since=` offset for log tails (the tail is polled every 3 s, 400 to 10000
lines); a price cap column from the DB (there is none to read).

## Before and after

Before: `audit/*.png` (the shipped UI against a copy of the live state DB). After: `screens/<view>.<dark|light|narrow>.png`
(every view of the new UI against `src/fixtures.ts`, which includes states the live DB does not hold: waiting for
capacity, errors, incomplete, failed, a pod RunPod no longer lists) and `live/*.png` (the new UI against the copy
of the live DB and the real RunPod pod list).

| View                                                  | Before                                                                  | After                                                                                                                                                           |
| ----------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Landing / overview                                    | `audit/run-exec-log.png` (a run list with a log)                        | `screens/overview.dark.png`, `.light.png`, `.narrow.png`; `live/overview.png`                                                                                   |
| Runs list                                             | `audit/run-exec-log.png`                                                | `screens/runs-list.dark.png`                                                                                                                                    |
| Run summary (running, waiting, incomplete, exec)      | `audit/run-sweep-waiting.png`                                           | `screens/run-sweeping-summary.dark.png`, `run-waiting-summary`, `run-incomplete-summary`, `run-exec-summary`                                                    |
| Run log                                               | `audit/run-exec-log.png`, `run-sweep-done-log.png`                      | `screens/run-waiting-log.dark.png`, `run-exec-log.dark.png`                                                                                                     |
| Results: live partial, charts, table, failures, empty | `audit/run-sweep-done-results.png` (a JS error), `run-exec-running.png` | `screens/run-sweeping-results-live`, `run-done-results-charts`, `run-done-results-table`, `run-errors-failures`, `run-failed-results-empty`; `live/results.png` |
| Pod of a run                                          | none                                                                    | `screens/run-exec-pod.dark.png`                                                                                                                                 |
| Pods                                                  | `audit/pods.png`                                                        | `screens/pods.dark.png`, `.light.png`; `live/pods.png`                                                                                                          |
| Compare                                               | none                                                                    | `screens/compare.dark.png`                                                                                                                                      |
| Image builds                                          | `audit/images.png`                                                      | `screens/images.dark.png`                                                                                                                                       |
| Serve                                                 | `audit/serve.png`                                                       | `screens/serve.dark.png` (error state)                                                                                                                          |
| Narrow                                                | `audit/narrow-runs.png`                                                 | the `.narrow.png` files                                                                                                                                         |

## 4. Tests

Component tests with `react-dom/server` bundled by esbuild and run by `node --test` (no new dependency: loading/empty/
error per view, outcome colouring, exec run hides Results, URL round-trip, page registry = URL views), Rust tests for the
new JSON fields (`web.rs`, `web/derive.rs`), and a screenshot set of every view (dark/light/narrow) committed in
`design/screens/`, regenerated by `npm run screens` against the fixture API.

## 5. Open questions for the owner (answered 2026-10-24)

1. RunPod verification: yes, read-only; `unverified` only when the call fails.
2. Cost: pod price x uptime, labelled estimated.
3. Keep Image builds and Serve as top-level views, or move under a "Tools" menu?
