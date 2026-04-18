import { Plugin } from "obsidian";
import { Im2TexSettings, DEFAULT_SETTINGS } from "./src/settings";
import { VIEW_TYPE, Im2TexView } from "./src/view";
import { Im2TexSettingTab } from "./src/settingsTab";

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
