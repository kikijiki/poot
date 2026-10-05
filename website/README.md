# poot docs website

The poot documentation site, built with [Docusaurus](https://docusaurus.io/): getting started,
architecture, usage, and examples (`docs/`). Benchmark results live under `../benchmarks/results`.

## Structure

- `docs/` - user-facing documentation at `/docs`, organized as:
  - **Documentation** (`index.md`) - path chooser for Develop / Serve / Architecture
  - **Develop** - library build, run, kernels, embedding, models
  - **Serve** - `poot-serve` operator guide and configuration
  - **Architecture** - design, IR, backends, serving design
  - **Reference** - feature matrix, performance, FAQ
- `src/components/architecture/` - interactive diagrams used by the architecture pages.

## Commands

Run these from the repo root inside `nix develop` (which provides `bun`):

```bash
just site-install   # install node deps (first run)
just site-dev       # live dev server at http://localhost:3000
just site-build     # production build into website/build (also the broken-link check)
just site-serve     # serve the production build
just site-check     # typecheck + build + ascii check
```

Or run `bun` directly from this directory: `bun start`, `bun run build`, `bun run serve`.

## Plugins

- Offline local search (`@easyops-cn/docusaurus-search-local`) - no network, no Algolia.
- `llms.txt` generation (`docusaurus-plugin-llms`) - emits `llms.txt` and `llms-full.txt`.
- Rust / bash / toml / nix syntax highlighting via Prism (`additionalLanguages`).
