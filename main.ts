import { Plugin } from "obsidian";
import { resetModel } from "./src/inference";
import { DEFAULT_RULES, DEFAULT_SETTINGS, type MathConvertSettings } from "./src/settings";
import { MathConvertSettingTab } from "./src/settingsTab";
import { MathConvertView, VIEW_TYPE } from "./src/view";

export default class MathConvertPlugin extends Plugin {
	settings!: MathConvertSettings;

	async onload() {
		await this.loadSettings();
		this.registerView(VIEW_TYPE, (leaf) => new MathConvertView(leaf, this.settings));
		this.addRibbonIcon("sigma", "Open math-convert", () => this.activateView());
		this.addCommand({
			id: "open",
			name: "Open math-convert sidebar",
			callback: () => this.activateView(),
		});
		this.addSettingTab(new MathConvertSettingTab(this.app, this));
	}

	onunload() {
		resetModel();
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
		void workspace.revealLeaf(leaf);
	}

	async loadSettings() {
		const saved = (await this.loadData()) as Partial<MathConvertSettings> | null;
		this.settings = Object.assign({} as MathConvertSettings, DEFAULT_SETTINGS, saved);
		if (!saved?.replacementRules) {
			this.settings.replacementRules = DEFAULT_RULES.map((r) => ({ ...r }));
		}
	}
	async saveSettings() {
		await this.saveData(this.settings);
	}
}
