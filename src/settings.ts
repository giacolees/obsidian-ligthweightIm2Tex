export const MODEL_ID = "alephpi/FormulaNet";

export interface PostProcessingRule {
	find: string;
	replace: string;
	isRegex: boolean;
	enabled: boolean;
}

export interface MathConvertSettings {
	modelId: string;
	replacementRules: PostProcessingRule[];
}

export const DEFAULT_RULES: PostProcessingRule[] = [
	{ find: "\\rarr", replace: "\\rightarrow", isRegex: false, enabled: true },
	{ find: "\\larr", replace: "\\leftarrow", isRegex: false, enabled: true },
	{ find: "\\rArr", replace: "\\Rightarrow", isRegex: false, enabled: true },
	{ find: "\\lArr", replace: "\\Leftarrow", isRegex: false, enabled: true },
	{ find: "\\harr", replace: "\\leftrightarrow", isRegex: false, enabled: true },
	{ find: "\\hArr", replace: "\\Leftrightarrow", isRegex: false, enabled: true },
];

export const DEFAULT_SETTINGS: MathConvertSettings = {
	modelId: MODEL_ID,
	replacementRules: [],
};

export function applyPostProcessing(latex: string, rules: PostProcessingRule[]): string {
	let result = latex;
	for (const rule of rules) {
		if (!rule.enabled || !rule.find) continue;
		if (rule.isRegex) {
			try {
				result = result.replace(new RegExp(rule.find, "g"), rule.replace);
			} catch {
				// Skip rules with invalid regex patterns
			}
		} else {
			result = result.split(rule.find).join(rule.replace);
		}
	}
	return result;
}
