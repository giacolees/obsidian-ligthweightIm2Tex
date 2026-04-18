# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev        # watch mode — rebuilds main.js on every change
npm run build      # type-check then produce production main.js
```

There are no tests. TypeScript type-check only runs as part of `build`.

To test in Obsidian: symlink or copy the repo folder into your vault's `.obsidian/plugins/lightweight-im2tex/`, enable the plugin, then reload.

## Architecture

Single-file plugin (`main.ts` → bundled to `main.js` via esbuild). Everything lives in one file:

- **`Im2TexPlugin`** — Obsidian `Plugin` entry point. Registers the sidebar view, ribbon icon, command, and settings tab.
- **`Im2TexView`** — Obsidian `ItemView` that renders the full sidebar UI: drop zone → canvas pair → result block. Manages image loading (file picker / drag-and-drop / paste), rubber-band region selection drawn on an overlay `<canvas>`, and the inference flow.
- **`Im2TexSettingTab`** — Settings tab with a single **Model ID** field (default: `alephpi/FormulaNet`). Changing the model ID resets the loaded singleton so the new model is fetched on next inference.
- **`ModelDownloadModal`** — Modal shown during the first-time model download. Displays a message and animated progress bar; closed automatically once loading finishes.
- **`ensureModel(modelId, onProgress)`** — Atomically loads `_model` and `_tokenizer` using a `_loadingPromise` guard so concurrent calls all await the same fetch. Resets the promise on failure so the user can retry.
- **`preprocessDataUrl(dataUrl)`** — Canvas-based image preprocessing pipeline matching FormulaNet's training: greyscale → auto-invert (dark-on-light heuristic) → margin crop → 384×384 center-pad with white → UniMERNet normalisation (mean 0.7931, std 0.1738). Returns a `Float32Array` shaped `[1, 3, 384, 384]`.
- **`runInference(dataUrl)`** — Calls `_model.generate()` with the preprocessed tensor, decodes via `_tokenizer.batch_decode()`, and strips `\!` spacing artifacts that FormulaNet sometimes emits.

### Canvas layout

Two stacked `<canvas>` elements share the same dimensions inside `.im2tex-canvas-container`:

| Element | Purpose |
|---|---|
| `canvas` (bottom) | Displays the loaded image |
| `overlayCanvas` (top) | Captures mouse/touch events; draws the selection rectangle |

`canvasPos()` converts mouse/touch events from CSS display pixels to canvas-internal pixels (the two can differ when CSS `width: 100%` scales a canvas that is narrower than its container). `getCropDataUrl()` maps the selection rectangle to natural-image coordinates and copies the region to an offscreen canvas before passing the PNG data-URL to `runInference`.

### Build notes

- esbuild externalises `obsidian`, `electron`, and all CodeMirror/Lezer packages.
- `@huggingface/transformers` and `onnxruntime-web` are bundled (not in the external list).
- `dev` mode produces an inline source map; `production` mode strips it.
- **`esbuild.config.mjs` contains two Electron-specific workarounds** that must be preserved:
  1. **Banner** — `delete globalThis[Symbol.for("onnxruntime")]` runs before any module initialises. Obsidian/Electron pre-registers an ORT singleton at that symbol; if present, transformers.js skips the onnxruntime-web branch and leaves `supportedDevices` empty, causing "Unsupported device: wasm".
  2. **`patchOnnxPlugin`** — After each build, replaces the `Oa()` guard in `main.js` from `if (!e && !t && Aa && Ue && Ln(Ue))` to `if (Aa)`. In a CJS bundle `import.meta.url` is undefined so `Ue` is never set, which causes `Oa()` to fall back to a CDN dynamic `import()` of `ort-wasm-simd-threaded.jsep.mjs`. That file spawns em-pthread workers which call `import('worker_threads')` — unsupported in Chromium's ESM worker scope. The patch forces the already-bundled inline WASM factory; the `.wasm` binary is still fetched from the CDN via `locateFile`.
- **`define: { "process.release.name": '"browser"' }`** — In Electron's renderer `process.release.name === "node"`, which makes transformers.js select `onnxruntime-node` (bundled as an empty stub). Patching the constant forces the onnxruntime-web path.
