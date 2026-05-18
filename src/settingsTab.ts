import { type App, type Plugin, PluginSettingTab, Setting } from "obsidian";
import { resetModel } from "./inference";
import { type MathConvertSettings, type PostProcessingRule, MODEL_ID } from "./settings";

export class MathConvertSettingTab extends PluginSettingTab {
	private plugin: Plugin & { settings: MathConvertSettings; saveSettings(): Promise<void> };

	constructor(
		app: App,
		plugin: Plugin & { settings: MathConvertSettings; saveSettings(): Promise<void> },
	) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();
		new Setting(containerEl).setName("Math-convert").setHeading();

		new Setting(containerEl)
			.setName("Model ID")
			.setDesc("Huggingface model ID used for inference.")
			.addText((t) =>
				t
					.setPlaceholder(MODEL_ID)
					.setValue(this.plugin.settings.modelId)
					.onChange(async (v) => {
						this.plugin.settings.modelId = v || MODEL_ID;
						resetModel();
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl).setName("Post-Processing Rules").setHeading();

		containerEl.createEl("p", {
			text: "Define rules to automatically modify the output LaTeX before it is inserted or copied. Rules are applied in order from top to bottom.",
			cls: "setting-item-description",
		});

		const rulesContainer = containerEl.createDiv({ cls: "math-convert-rules-container" });
		this.renderRules(rulesContainer);

		new Setting(containerEl).addButton((btn) =>
			btn.setButtonText("Add Rule").onClick(async () => {
				this.plugin.settings.replacementRules.push({
					find: "",
					replace: "",
					isRegex: false,
					enabled: true,
				});
				await this.plugin.saveSettings();
				this.display();
			}),
		);
	}

	private renderRules(container: HTMLElement): void {
		container.empty();
		const { replacementRules } = this.plugin.settings;

		if (replacementRules.length === 0) {
			container.createEl("p", {
				text: "No rules defined. Click 'Add Rule' to create one.",
				cls: "math-convert-rules-empty",
			});
			return;
		}

		const table = container.createEl("table", { cls: "math-convert-rules-table" });
		const thead = table.createEl("thead");
		const headerRow = thead.createEl("tr");
		for (const label of ["Find", "Replace With", "Regex", "On", ""]) {
			headerRow.createEl("th", { text: label });
		}

		const tbody = table.createEl("tbody");
		for (let i = 0; i < replacementRules.length; i++) {
			this.renderRuleRow(tbody, replacementRules, i);
		}
	}

	private renderRuleRow(
		tbody: HTMLTableSectionElement,
		rules: PostProcessingRule[],
		index: number,
	): void {
		const rule = rules[index];
		const tr = tbody.createEl("tr");

		const findTd = tr.createEl("td");
		const findInput = findTd.createEl("input");
		findInput.type = "text";
		findInput.addClass("math-convert-rules-input");
		findInput.value = rule.find;
		findInput.placeholder = "Find…";
		findInput.addEventListener("change", async () => {
			rule.find = findInput.value;
			await this.plugin.saveSettings();
		});

		const replaceTd = tr.createEl("td");
		const replaceInput = replaceTd.createEl("input");
		replaceInput.type = "text";
		replaceInput.addClass("math-convert-rules-input");
		replaceInput.value = rule.replace;
		replaceInput.placeholder = "Replace with…";
		replaceInput.addEventListener("change", async () => {
			rule.replace = replaceInput.value;
			await this.plugin.saveSettings();
		});

		const regexTd = tr.createEl("td");
		regexTd.addClass("math-convert-rules-cell--center");
		const regexCb = regexTd.createEl("input");
		regexCb.type = "checkbox";
		regexCb.checked = rule.isRegex;
		regexCb.title = "Treat 'Find' as a regular expression";
		regexCb.addEventListener("change", async () => {
			rule.isRegex = regexCb.checked;
			await this.plugin.saveSettings();
		});

		const enabledTd = tr.createEl("td");
		enabledTd.addClass("math-convert-rules-cell--center");
		const enabledCb = enabledTd.createEl("input");
		enabledCb.type = "checkbox";
		enabledCb.checked = rule.enabled;
		enabledCb.title = "Enable this rule";
		enabledCb.addEventListener("change", async () => {
			rule.enabled = enabledCb.checked;
			await this.plugin.saveSettings();
		});

		const actionsTd = tr.createEl("td");
		const deleteBtn = actionsTd.createEl("button", {
			text: "Delete",
			cls: "math-convert-btn math-convert-btn--sm",
		});
		deleteBtn.addEventListener("click", async () => {
			rules.splice(index, 1);
			await this.plugin.saveSettings();
			this.display();
		});
	}
}
