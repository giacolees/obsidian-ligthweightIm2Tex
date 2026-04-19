import { ItemView, MarkdownView, Notice, type Editor, type WorkspaceLeaf } from "obsidian";
import { ensureModel, isModelLoaded, runInference } from "./inference";
import { ModelDownloadModal } from "./modal";
import type { Im2TexSettings } from "./settings";

export const VIEW_TYPE = "math-convert-sidebar";

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
	private resizeObserver: ResizeObserver | null = null;

	private loadedImage: HTMLImageElement | null = null;
	private isDragging = false;
	private startX = 0;
	private startY = 0;
	private currentRect: Rect | null = null;
	private busy = false;
	private lastEditor: Editor | null = null;

	constructor(leaf: WorkspaceLeaf, settings: Im2TexSettings) {
		super(leaf);
		this.settings = settings;
	}

	getViewType() {
		return VIEW_TYPE;
	}
	getDisplayText() {
		return "Math-Convert";
	}
	getIcon() {
		return "sigma";
	}

	async onOpen() {
		await Promise.resolve();
		const root = this.containerEl.children[1] as HTMLElement;
		root.empty();
		root.addClass("im2tex-root");
		this.buildUi(root);
		this.registerEvent(
			this.app.workspace.on("active-leaf-change", () => {
				const view = this.app.workspace.getActiveViewOfType(MarkdownView);
				if (view) this.lastEditor = view.editor;
			}),
		);
	}

	async onClose() {
		await Promise.resolve();
		this.resizeObserver?.disconnect();
		this.resizeObserver = null;
	}

	// ---------------------------------------------------------------------------
	// UI
	// ---------------------------------------------------------------------------

	private buildUi(root: HTMLElement) {
		const header = root.createDiv({ cls: "im2tex-header" });
		header.createEl("h4", { text: "Math-Convert" });
		this.statusEl = header.createEl("span", { cls: "im2tex-status" });

		this.dropZone = root.createDiv({ cls: "im2tex-dropzone" });
		this.dropZone.createEl("p", {
			text: "Drop or paste an image here",
			cls: "im2tex-dropzone-hint",
		});

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
		browseBtn.addEventListener("click", (e) => {
			e.stopPropagation();
			fileInput.click();
		});

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
			if (file?.type.startsWith("image/")) this.loadFile(file);
		});
		root.addEventListener("paste", (e) => {
			const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
				i.type.startsWith("image/"),
			);
			if (item) {
				const f = item.getAsFile();
				if (f) this.loadFile(f);
			}
		});
		root.setAttribute("tabindex", "0");

		this.canvasContainer = root.createDiv({ cls: "im2tex-canvas-container" });
		this.canvasContainer.style.display = "none";
		this.canvas = this.canvasContainer.createEl("canvas", { cls: "im2tex-canvas" });
		this.overlayCanvas = this.canvasContainer.createEl("canvas", { cls: "im2tex-overlay" });
		this.attachSelectionListeners();
		this.resizeObserver?.disconnect();
		this.resizeObserver = new ResizeObserver(() => {
			if (this.loadedImage) this.renderImage();
		});
		this.resizeObserver.observe(this.canvasContainer);

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
		const actions = resultHeader.createDiv({ cls: "im2tex-result-actions" });
		const copyBtn = actions.createEl("button", {
			text: "Copy",
			cls: "im2tex-btn im2tex-btn--sm im2tex-btn--primary",
		});
		copyBtn.addEventListener("click", () => this.copyLatex());
		const insertBtn = actions.createEl("button", {
			text: "Insert",
			cls: "im2tex-btn im2tex-btn--sm im2tex-btn--primary",
		});
		insertBtn.addEventListener("click", () => this.insertLatex());
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
		img.decoding = "async";
		img.onload = () => {
			this.loadedImage = img;
			this.currentRect = null;
			this.renderImage();
			this.dropZone.style.display = "none";
			this.canvasContainer.style.display = "block";
			this.resultContainer.style.display = "none";
			this.setStatus("");
		};
		img.onerror = () => {
			console.error("[Im2Tex] Failed to load image");
			new Notice("Could not load that image.");
			this.setStatus("Image load failed");
		};
		img.src = src;
	}

	private renderImage() {
		if (!this.loadedImage) return;
		const maxW = this.canvasContainer.clientWidth || 300;
		const scale = Math.min(1, maxW / this.loadedImage.naturalWidth);
		const w = Math.round(this.loadedImage.naturalWidth * scale);
		const h = Math.round(this.loadedImage.naturalHeight * scale);
		for (const c of [this.canvas, this.overlayCanvas]) {
			c.width = w;
			c.height = h;
		}
		this.get2dContext(this.canvas).drawImage(this.loadedImage, 0, 0, w, h);
		this.clearOverlay();
	}

	// ---------------------------------------------------------------------------
	// Rectangle selection
	// ---------------------------------------------------------------------------

	private attachSelectionListeners() {
		this.overlayCanvas.addEventListener("pointerdown", (e) => {
			if (e.pointerType === "mouse" && e.button !== 0) return;
			e.preventDefault();
			this.overlayCanvas.setPointerCapture(e.pointerId);
			const { x, y } = this.canvasPos(e);
			this.isDragging = true;
			this.startX = x;
			this.startY = y;
			this.currentRect = null;
			this.redrawOverlay();
		});
		this.overlayCanvas.addEventListener("pointermove", (e) => {
			if (!this.isDragging) return;
			const { x, y } = this.canvasPos(e);
			this.currentRect = this.normalizeRect(this.startX, this.startY, x, y);
			this.redrawOverlay();
		});

		const finishDrag = (e: PointerEvent) => {
			if (!this.isDragging) return;
			this.isDragging = false;
			if (this.overlayCanvas.hasPointerCapture(e.pointerId)) {
				this.overlayCanvas.releasePointerCapture(e.pointerId);
			}
			const { x, y } = this.canvasPos(e);
			this.currentRect = this.normalizeRect(this.startX, this.startY, x, y);
			this.redrawOverlay();
		};

		this.overlayCanvas.addEventListener("pointerup", finishDrag);
		this.overlayCanvas.addEventListener("pointercancel", finishDrag);
	}

	private canvasPos(e: MouseEvent | PointerEvent | Touch) {
		const r = this.overlayCanvas.getBoundingClientRect();
		return {
			x: this.clamp(
				(e.clientX - r.left) * (this.overlayCanvas.width / r.width),
				0,
				this.overlayCanvas.width,
			),
			y: this.clamp(
				(e.clientY - r.top) * (this.overlayCanvas.height / r.height),
				0,
				this.overlayCanvas.height,
			),
		};
	}

	private clamp(value: number, min: number, max: number): number {
		return Math.min(max, Math.max(min, value));
	}

	private normalizeRect(x1: number, y1: number, x2: number, y2: number): Rect {
		return { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
	}

	private clearOverlay() {
		this.get2dContext(this.overlayCanvas).clearRect(
			0,
			0,
			this.overlayCanvas.width,
			this.overlayCanvas.height,
		);
	}

	private redrawOverlay() {
		this.clearOverlay();
		if (!this.currentRect || this.currentRect.w < 2 || this.currentRect.h < 2) return;
		const ctx = this.get2dContext(this.overlayCanvas);
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
		for (const [cx, cy] of [
			[x, y],
			[x + w, y],
			[x, y + h],
			[x + w, y + h],
		]) {
			ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
		}
	}

	private get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
		const context = canvas.getContext("2d");
		if (!context) {
			throw new Error("Could not get a 2D canvas context.");
		}
		return context;
	}

	// ---------------------------------------------------------------------------
	// Inference
	// ---------------------------------------------------------------------------

	private async handleInfer() {
		if (!this.loadedImage) {
			new Notice("Please load an image first.");
			return;
		}
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
			console.error("[Math-Convert]", err);
			new Notice(`Math-Convert error: ${msg}`);
			this.setStatus("Error");
		} finally {
			this.setBusy(false);
		}
	}

	private getCropDataUrl(): string {
		const img = this.loadedImage;
		if (!img) {
			throw new Error("No image loaded.");
		}
		const scaleX = img.naturalWidth / this.canvas.width;
		const scaleY = img.naturalHeight / this.canvas.height;
		const rect =
			this.currentRect && this.currentRect.w >= 4 && this.currentRect.h >= 4
				? this.currentRect
				: null;

		const srcX = rect ? Math.round(rect.x * scaleX) : 0;
		const srcY = rect ? Math.round(rect.y * scaleY) : 0;
		const srcW = rect ? Math.round(rect.w * scaleX) : img.naturalWidth;
		const srcH = rect ? Math.round(rect.h * scaleY) : img.naturalHeight;

		const off = document.createElement("canvas");
		off.width = srcW;
		off.height = srcH;
		this.get2dContext(off).drawImage(img, srcX, srcY, srcW, srcH, 0, 0, srcW, srcH);
		return off.toDataURL("image/png");
	}

	private showResult(latex: string) {
		this.latexDisplay.setText(latex);
		this.resultContainer.style.display = "block";
		this.resultContainer.scrollIntoView({ behavior: "smooth" });
	}

	private copyLatex() {
		const latex = this.latexDisplay.getText();
		if (!latex) {
			new Notice("No LaTeX formula to copy yet.");
			return;
		}

		navigator.clipboard
			.writeText(latex)
			.then(() => new Notice("Copied!"))
			.catch((err: unknown) => {
				console.error("[Im2Tex] Clipboard write failed", err);
				new Notice("Could not copy to the clipboard.");
			});
	}

	private insertLatex() {
		const latex = this.latexDisplay.getText();
		if (!latex) {
			new Notice("No LaTeX formula to insert yet.");
			return;
		}
		const editor = this.app.workspace.activeEditor?.editor ?? this.lastEditor;
		if (!editor) {
			new Notice("No active editor — click into a note first.");
			return;
		}
		editor.replaceSelection(`$${latex}$`);
	}

	private clearAll() {
		this.loadedImage = null;
		this.currentRect = null;
		this.latexDisplay.empty();
		this.clearOverlay();
		this.canvas.width = 0;
		this.canvas.height = 0;
		this.overlayCanvas.width = 0;
		this.overlayCanvas.height = 0;
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

	private setStatus(msg: string) {
		this.statusEl.setText(msg);
	}
}
