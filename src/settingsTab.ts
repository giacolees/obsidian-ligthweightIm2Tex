import { App, Plugin, PluginSettingTab, Setting } from "obsidian";
import { Im2TexSettings, MODEL_ID } from "./settings";
import { resetModel } from "./inference";

export class Im2TexSettingTab extends PluginSettingTab {
  private plugin: Plugin & { settings: Im2TexSettings; saveSettings(): Promise<void> };

  constructor(app: App, plugin: Plugin & { settings: Im2TexSettings; saveSettings(): Promise<void> }) {
    super(app, plugin);
    this.plugin = plugin;
  }

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
            resetModel();
            await this.plugin.saveSettings();
          })
      );
  }
}
