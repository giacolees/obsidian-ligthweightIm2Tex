import { type App, Modal } from "obsidian";

export class ModelDownloadModal extends Modal {
	private msgEl!: HTMLParagraphElement;
	private barEl!: HTMLDivElement;

	constructor(app: App) {
		super(app);
		this.modalEl.addClass("im2tex-download-modal");
	}

	onOpen() {
		const { contentEl } = this;
		contentEl.createEl("h3", { text: "Downloading Math-Convert model" });
		contentEl.createEl("p", {
			text: "This only happens once. The model (~100 MB) will be cached locally.",
			cls: "im2tex-download-desc",
		});
		this.msgEl = contentEl.createEl("p", { text: "Starting…", cls: "im2tex-download-msg" });
		const wrap = contentEl.createDiv({ cls: "im2tex-bar-wrap" });
		this.barEl = wrap.createDiv({ cls: "im2tex-bar" });
	}

	onClose() {
		this.contentEl.empty();
	}

	update(msg: string, pct?: number) {
		this.msgEl.setText(msg);
		if (pct !== undefined) this.barEl.style.width = `${pct}%`;
	}
}
