import {
  App,
  ItemView,
  Modal,
  Plugin,
  PluginSettingTab,
  Setting,
  WorkspaceLeaf,
  Notice,
} from "obsidian";
import {
  VisionEncoderDecoderModel,
  PreTrainedTokenizer,
  Tensor,
  cat,
  env,
  type ProgressInfo,
} from "@huggingface/transformers";

const VIEW_TYPE = "im2tex-sidebar";
const MODEL_ID = "alephpi/FormulaNet";
const TARGET_SIZE = 384;

// UniMERNet normalisation constants (must match training)
const NORM_MEAN = 0.7931;
const NORM_STD = 0.1738;

env.allowLocalModels = false;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

interface Im2TexSettings {
  modelId: string;
}

const DEFAULT_SETTINGS: Im2TexSettings = {
  modelId: MODEL_ID,
};

// ---------------------------------------------------------------------------
// Model singleton — loaded atomically so _model/_tokenizer are never mismatched
// ---------------------------------------------------------------------------

let _model: VisionEncoderDecoderModel | null = null;
let _tokenizer: PreTrainedTokenizer | null = null;
let _loadingPromise: Promise<void> | null = null;

async function ensureModel(
  modelId: string,
  onProgress: (msg: string, pct?: number) => void
): Promise<void> {
  if (_model && _tokenizer) return;

  if (!_loadingPromise) {
    _loadingPromise = (async () => {
      const model = await VisionEncoderDecoderModel.from_pretrained(modelId, {
        dtype: "fp32",
        progress_callback: (info: ProgressInfo) => {
          const { msg, pct } = parseProgress(info);
          onProgress(msg, pct);
        },
      });
      const tokenizer = await PreTrainedTokenizer.from_pretrained(modelId);
      // assign both only after both succeed
      _model = model;
      _tokenizer = tokenizer;
    })();

    // on failure reset so the user can retry
    _loadingPromise.catch(() => { _loadingPromise = null; });
  }

  await _loadingPromise;
}

function parseProgress(info: ProgressInfo): { msg: string; pct?: number } {
  if (info.status === "progress") {
    const i = info as unknown as { loaded: number; total: number; file?: string };
    const pct = i.total ? Math.round((i.loaded / i.total) * 100) : undefined;
    return { msg: `Downloading${i.file ? ` ${i.file}` : ""}…`, pct };
  }
  if (info.status === "initiate" || info.status === "download") return { msg: "Downloading model…" };
  if (info.status === "done") return { msg: "Loading weights…" };
  return { msg: "Initialising…" };
}

// ---------------------------------------------------------------------------
// Download modal
// ---------------------------------------------------------------------------

class ModelDownloadModal extends Modal {
  private msgEl!: HTMLParagraphElement;
  private barEl!: HTMLDivElement;

  constructor(app: App) {
    super(app);
    this.modalEl.addClass("im2tex-download-modal");
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Downloading Im2Tex model" });
    contentEl.createEl("p", {
      text: "This only happens once. The model (~100 MB) will be cached locally.",
      cls: "im2tex-download-desc",
    });
    this.msgEl = contentEl.createEl("p", { text: "Starting…", cls: "im2tex-download-msg" });
    const wrap = contentEl.createDiv({ cls: "im2tex-bar-wrap" });
    this.barEl = wrap.createDiv({ cls: "im2tex-bar" });
  }

  onClose() { this.contentEl.empty(); }

  update(msg: string, pct?: number) {
    this.msgEl.setText(msg);
    if (pct !== undefined) this.barEl.style.width = `${pct}%`;
  }
}

// ---------------------------------------------------------------------------
// Image preprocessing  (mirrors Texo-web imageProcessor.ts, canvas-only)
// ---------------------------------------------------------------------------

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

async function preprocessDataUrl(dataUrl: string): Promise<Float32Array> {
  const img = await new Promise<HTMLImageElement>((res, rej) => {
    const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = dataUrl;
  });

  const w = img.width, h = img.height;
  const oc = makeCanvas(w, h);
  const ctx = oc.getContext("2d")!;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;

  // Greyscale (luminance)
  const grey = new Uint8Array(w * h);
  for (let i = 0; i < grey.length; i++) {
    grey[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
  }

  // Auto-invert: white-on-black → black-on-white
  const histogram = new Uint32Array(256);
  for (const v of grey) histogram[v]++;
  const darkPx = histogram.slice(0, 200).reduce((s, v) => s + v, 0);
  const lightPx = histogram.slice(200).reduce((s, v) => s + v, 0);
  if (darkPx >= lightPx) for (let i = 0; i < grey.length; i++) grey[i] = 255 - grey[i];

  // Crop margins
  const threshold = 200;
  let minX = w, minY = h, maxX = 0, maxY = 0, hasContent = false;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (grey[y * w + x] < threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        hasContent = true;
      }
    }
  }
  if (!hasContent) { minX = 0; minY = 0; maxX = w - 1; maxY = h - 1; }

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;

  // Build crop ImageData directly (avoid slow pixel-by-pixel fillRect)
  const cropData = new ImageData(cropW, cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const v = grey[(y + minY) * w + (x + minX)];
      const idx = (y * cropW + x) * 4;
      cropData.data[idx] = v;
      cropData.data[idx + 1] = v;
      cropData.data[idx + 2] = v;
      cropData.data[idx + 3] = 255;
    }
  }
  const cc = makeCanvas(cropW, cropH);
  cc.getContext("2d")!.putImageData(cropData, 0, 0);

  // Resize to 384×384 preserving aspect ratio, centre-pad with white
  const scale = Math.min(TARGET_SIZE / cropW, TARGET_SIZE / cropH);
  const newW = Math.round(cropW * scale);
  const newH = Math.round(cropH * scale);
  const padX = Math.floor((TARGET_SIZE - newW) / 2);
  const padY = Math.floor((TARGET_SIZE - newH) / 2);

  const rc = makeCanvas(TARGET_SIZE, TARGET_SIZE);
  const rctx = rc.getContext("2d")!;
  rctx.fillStyle = "white";
  rctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  rctx.drawImage(cc, padX, padY, newW, newH);

  const outRgba = rctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
  const result = new Float32Array(TARGET_SIZE * TARGET_SIZE);
  for (let i = 0; i < result.length; i++) {
    result[i] = (outRgba[i * 4] / 255 - NORM_MEAN) / NORM_STD;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Inference (assumes model already loaded via ensureModel)
// ---------------------------------------------------------------------------

async function runInference(dataUrl: string): Promise<string> {
  const array = await preprocessDataUrl(dataUrl);
  const t = new Tensor("float32", array, [1, 1, TARGET_SIZE, TARGET_SIZE]);
  const pixel_values = cat([t, t, t], 1);
  const outputs = await _model!.generate({ inputs: pixel_values });
  const tok = _tokenizer!;
  return (tok.batch_decode(outputs as Parameters<typeof tok.batch_decode>[0], {
    skip_special_tokens: true,
  }) as string[])[0];
}

// ---------------------------------------------------------------------------
// Sidebar View
// ---------------------------------------------------------------------------

class Im2TexView extends ItemView {
  private settings: Im2TexSettings;

  private dropZone: HTMLDivElement;
  private canvasContainer: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private overlayCanvas: HTMLCanvasElement;
  private resultContainer: HTMLDivElement;
  private latexDisplay: HTMLDivElement;
  private inferBtn: HTMLButtonElement;
  private statusEl: HTMLSpanElement;

  private loadedImage: HTMLImageElement | null = null;
  private isDragging = false;
  private startX = 0;
  private startY = 0;
  private currentRect: { x: number; y: number; w: number; h: number } | null = null;
  private busy = false;

  constructor(leaf: WorkspaceLeaf, settings: Im2TexSettings) {
    super(leaf);
    this.settings = settings;
  }

  getViewType() { return VIEW_TYPE; }
  getDisplayText() { return "Im2Tex"; }
  getIcon() { return "sigma"; }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("im2tex-root");
    this.buildUI(root);
  }

  async onClose() {}

  // -------------------------------------------------------------------------
  // UI
  // -------------------------------------------------------------------------

  private buildUI(root: HTMLElement) {
    const header = root.createDiv({ cls: "im2tex-header" });
    header.createEl("h4", { text: "Im2Tex" });
    this.statusEl = header.createEl("span", { cls: "im2tex-status" });

    this.dropZone = root.createDiv({ cls: "im2tex-dropzone" });
    this.dropZone.createEl("p", { text: "Drop or paste an image here", cls: "im2tex-dropzone-hint" });

    const browseBtn = this.dropZone.createEl("button", {
      text: "Browse file…",
      cls: "im2tex-btn im2tex-btn--primary im2tex-browse-btn",
    });
    const fileInput = this.dropZone.createEl("input") as HTMLInputElement;
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (file) this.loadFile(file);
      fileInput.value = "";
    });
    browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

    this.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropZone.addClass("im2tex-dropzone--active");
    });
    this.dropZone.addEventListener("dragleave", () => {
      this.dropZone.removeClass("im2tex-dropzone--active");
    });
    this.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropZone.removeClass("im2tex-dropzone--active");
      const file = e.dataTransfer?.files[0];
      if (file && file.type.startsWith("image/")) this.loadFile(file);
    });
    root.addEventListener("paste", (e) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
      if (item) { const f = item.getAsFile(); if (f) this.loadFile(f); }
    });
    root.setAttribute("tabindex", "0");

    this.canvasContainer = root.createDiv({ cls: "im2tex-canvas-container" });
    this.canvasContainer.style.display = "none";
    this.canvas = this.canvasContainer.createEl("canvas", { cls: "im2tex-canvas" });
    this.overlayCanvas = this.canvasContainer.createEl("canvas", { cls: "im2tex-overlay" });
    this.attachSelectionListeners();

    const btnRow = root.createDiv({ cls: "im2tex-btn-row" });
    this.inferBtn = btnRow.createEl("button", {
      text: "Detect formula",
      cls: "im2tex-btn im2tex-btn--primary",
    });
    this.inferBtn.addEventListener("click", () => this.handleInfer());

    const clearBtn = btnRow.createEl("button", { text: "Clear", cls: "im2tex-btn" });
    clearBtn.addEventListener("click", () => this.clearAll());

    this.resultContainer = root.createDiv({ cls: "im2tex-result" });
    this.resultContainer.style.display = "none";

    const resultHeader = this.resultContainer.createDiv({ cls: "im2tex-result-header" });
    resultHeader.createEl("span", { text: "LaTeX formula" });
    const copyBtn = resultHeader.createEl("button", {
      text: "Copy",
      cls: "im2tex-btn im2tex-btn--sm im2tex-btn--primary",
    });
    copyBtn.addEventListener("click", () => this.copyLatex());
    this.latexDisplay = this.resultContainer.createDiv({ cls: "im2tex-latex-display" });
  }

  // -------------------------------------------------------------------------
  // Image loading
  // -------------------------------------------------------------------------

  private loadFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => this.loadImageSrc(e.target?.result as string);
    reader.readAsDataURL(file);
  }

  private loadImageSrc(src: string) {
    const img = new Image();
    img.onload = () => {
      this.loadedImage = img;
      this.currentRect = null;
      this.renderImage();
      this.dropZone.style.display = "none";
      this.canvasContainer.style.display = "block";
      this.resultContainer.style.display = "none";
      this.setStatus("");
    };
    img.src = src;
  }

  private renderImage() {
    if (!this.loadedImage) return;
    const maxW = this.canvasContainer.clientWidth || 300;
    const scale = Math.min(1, maxW / this.loadedImage.naturalWidth);
    const w = Math.round(this.loadedImage.naturalWidth * scale);
    const h = Math.round(this.loadedImage.naturalHeight * scale);
    for (const c of [this.canvas, this.overlayCanvas]) { c.width = w; c.height = h; }
    this.canvas.getContext("2d")!.drawImage(this.loadedImage, 0, 0, w, h);
    this.clearOverlay();
  }

  // -------------------------------------------------------------------------
  // Rectangle selection
  // -------------------------------------------------------------------------

  private attachSelectionListeners() {
    this.overlayCanvas.addEventListener("mousedown", (e) => {
      const { x, y } = this.canvasPos(e);
      this.isDragging = true; this.startX = x; this.startY = y; this.currentRect = null;
    });
    this.overlayCanvas.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      const { x, y } = this.canvasPos(e);
      this.currentRect = this.normalizeRect(this.startX, this.startY, x, y);
      this.redrawOverlay();
    });
    this.overlayCanvas.addEventListener("mouseup", (e) => {
      if (!this.isDragging) return;
      this.isDragging = false;
      const { x, y } = this.canvasPos(e);
      this.currentRect = this.normalizeRect(this.startX, this.startY, x, y);
      this.redrawOverlay();
    });
    this.overlayCanvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const { x, y } = this.canvasPos(e.touches[0]);
      this.isDragging = true; this.startX = x; this.startY = y;
    });
    this.overlayCanvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      if (!this.isDragging) return;
      const { x, y } = this.canvasPos(e.touches[0]);
      this.currentRect = this.normalizeRect(this.startX, this.startY, x, y);
      this.redrawOverlay();
    });
    this.overlayCanvas.addEventListener("touchend", () => { this.isDragging = false; });
  }

  private canvasPos(e: MouseEvent | Touch) {
    const r = this.overlayCanvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  private normalizeRect(x1: number, y1: number, x2: number, y2: number) {
    return { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
  }

  private clearOverlay() {
    this.overlayCanvas.getContext("2d")!.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  private redrawOverlay() {
    this.clearOverlay();
    if (!this.currentRect || this.currentRect.w < 2 || this.currentRect.h < 2) return;
    const ctx = this.overlayCanvas.getContext("2d")!;
    const { x, y, w, h } = this.currentRect;

    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    ctx.clearRect(x, y, w, h);

    ctx.strokeStyle = "#7c6af7";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(x, y, w, h);

    ctx.setLineDash([]);
    ctx.fillStyle = "#7c6af7";
    const hs = 6;
    for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
      ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
    }
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  private async handleInfer() {
    if (!this.loadedImage) { new Notice("Please load an image first."); return; }
    if (this.busy) return;
    this.setBusy(true);

    const isFirstLoad = !_model || !_tokenizer;
    const modal = isFirstLoad ? new ModelDownloadModal(this.app) : null;
    if (modal) modal.open();

    try {
      await ensureModel(this.settings.modelId, (msg, pct) => {
        modal?.update(msg, pct);
        this.setStatus(msg);
      });
      modal?.close();

      this.setStatus("Preprocessing…");
      const latex = await runInference(this.getCropDataUrl());
      this.showResult(latex);
      this.setStatus("Done");
    } catch (err: unknown) {
      modal?.close();
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[Im2Tex]", err);
      new Notice(`Im2Tex error: ${msg}`);
      this.setStatus("Error");
    } finally {
      this.setBusy(false);
    }
  }

  private getCropDataUrl(): string {
    const img = this.loadedImage!;
    const scaleX = img.naturalWidth / this.canvas.width;
    const scaleY = img.naturalHeight / this.canvas.height;
    const hasRect = this.currentRect && this.currentRect.w >= 4 && this.currentRect.h >= 4;

    const srcX = hasRect ? Math.round(this.currentRect!.x * scaleX) : 0;
    const srcY = hasRect ? Math.round(this.currentRect!.y * scaleY) : 0;
    const srcW = hasRect ? Math.round(this.currentRect!.w * scaleX) : img.naturalWidth;
    const srcH = hasRect ? Math.round(this.currentRect!.h * scaleY) : img.naturalHeight;

    const off = document.createElement("canvas");
    off.width = srcW; off.height = srcH;
    off.getContext("2d")!.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, srcW, srcH);
    return off.toDataURL("image/png");
  }

  private showResult(latex: string) {
    this.latexDisplay.setText(latex);
    this.resultContainer.style.display = "block";
    this.resultContainer.scrollIntoView({ behavior: "smooth" });
  }

  private copyLatex() {
    navigator.clipboard.writeText(this.latexDisplay.getText()).then(() => new Notice("Copied!"));
  }

  private clearAll() {
    this.loadedImage = null;
    this.currentRect = null;
    this.canvasContainer.style.display = "none";
    this.dropZone.style.display = "";
    this.resultContainer.style.display = "none";
    this.setStatus("");
  }

  private setBusy(busy: boolean) {
    this.busy = busy;
    this.inferBtn.disabled = busy;
    this.inferBtn.setText(busy ? "Running…" : "Detect formula");
  }

  private setStatus(msg: string) { this.statusEl.setText(msg); }
}

// ---------------------------------------------------------------------------
// Settings tab
// ---------------------------------------------------------------------------

class Im2TexSettingTab extends PluginSettingTab {
  plugin: Im2TexPlugin;
  constructor(app: App, plugin: Im2TexPlugin) { super(app, plugin); this.plugin = plugin; }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Im2Tex settings" });

    new Setting(containerEl)
      .setName("Model ID")
      .setDesc("HuggingFace model ID used for inference.")
      .addText((t) =>
        t.setPlaceholder(MODEL_ID)
          .setValue(this.plugin.settings.modelId)
          .onChange(async (v) => {
            this.plugin.settings.modelId = v || MODEL_ID;
            _model = null;
            _tokenizer = null;
            _loadingPromise = null;
            await this.plugin.saveSettings();
          })
      );
  }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export default class Im2TexPlugin extends Plugin {
  settings: Im2TexSettings;

  async onload() {
    await this.loadSettings();
    this.registerView(VIEW_TYPE, (leaf) => new Im2TexView(leaf, this.settings));
    this.addRibbonIcon("sigma", "Open Im2Tex", () => this.activateView());
    this.addCommand({ id: "open-im2tex", name: "Open Im2Tex sidebar", callback: () => this.activateView() });
    this.addSettingTab(new Im2TexSettingTab(this.app, this));
  }

  onunload() { this.app.workspace.detachLeavesOfType(VIEW_TYPE); }

  async activateView() {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      leaf = workspace.getRightLeaf(false)!;
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  async loadSettings() { this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData()); }
  async saveSettings() { await this.saveData(this.settings); }
}
