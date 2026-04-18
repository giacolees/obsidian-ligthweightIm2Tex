import { ItemView, Notice, WorkspaceLeaf } from "obsidian";
import { Im2TexSettings } from "./settings";
import { ModelDownloadModal } from "./modal";
import { ensureModel, isModelLoaded, runInference } from "./inference";

export const VIEW_TYPE = "im2tex-sidebar";

type Rect = { x: number; y: number; w: number; h: number };

export class Im2TexView extends ItemView {
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
  private currentRect: Rect | null = null;
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

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Image loading
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Rectangle selection
  // ---------------------------------------------------------------------------

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
    return {
      x: (e.clientX - r.left) * (this.overlayCanvas.width  / r.width),
      y: (e.clientY - r.top)  * (this.overlayCanvas.height / r.height),
    };
  }

  private normalizeRect(x1: number, y1: number, x2: number, y2: number): Rect {
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

  // ---------------------------------------------------------------------------
  // Inference
  // ---------------------------------------------------------------------------

  private async handleInfer() {
    if (!this.loadedImage) { new Notice("Please load an image first."); return; }
    if (this.busy) return;
    this.setBusy(true);

    const modal = !isModelLoaded() ? new ModelDownloadModal(this.app) : null;
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
