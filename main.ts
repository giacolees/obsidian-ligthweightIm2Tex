import { Plugin } from "obsidian";
import { resetModel } from "./src/inference";
import { DEFAULT_SETTINGS, type Im2TexSettings } from "./src/settings";
import { Im2TexSettingTab } from "./src/settingsTab";
import { Im2TexView, VIEW_TYPE } from "./src/view";

export default class Im2TexPlugin extends Plugin {
	settings: Im2TexSettings;

	async onload() {
		await this.loadSettings();
		this.registerView(VIEW_TYPE, (leaf) => new Im2TexView(leaf, this.settings));
		this.addRibbonIcon("sigma", "Open Math-Convert", () => this.activateView());
		this.addCommand({
			id: "open-math-convert",
			name: "Open Math-Convert sidebar",
			callback: () => this.activateView(),
		});
		this.addSettingTab(new Im2TexSettingTab(this.app, this));
	}

	onunload() {
		resetModel();
		this.app.workspace.detachLeavesOfType(VIEW_TYPE);
	}

	async activateView() {
		const { workspace } = this.app;
		let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
		if (!leaf) {
			const rightLeaf = workspace.getRightLeaf(false);
			if (!rightLeaf) {
				throw new Error("Could not open the Math-Convert sidebar.");
			}
			leaf = rightLeaf;
			await leaf.setViewState({ type: VIEW_TYPE, active: true });
		}
		workspace.revealLeaf(leaf);
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}
	async saveSettings() {
		await this.saveData(this.settings);
	}
}
