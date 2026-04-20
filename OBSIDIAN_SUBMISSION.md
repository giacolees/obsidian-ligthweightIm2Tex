# Obsidian Community Plugin Submission

This repository is prepared for submission to the Obsidian community plugin directory.

## Official submission flow

According to the current Obsidian docs, the initial submission requires:

1. A public GitHub repository with source code.
2. A GitHub release whose tag exactly matches `manifest.json`'s `version`.
3. Release assets attached to that release:
   - `main.js`
   - `manifest.json`
   - `styles.css` (optional, but included by this plugin)
4. A pull request to `obsidianmd/obsidian-releases` adding the plugin to `community-plugins.json`.

Reference:

- https://docs.obsidian.md/Plugins/Releasing/Submit%20your%20plugin

## Community plugins entry

Add this object at the end of `community-plugins.json`:

```json
{
  "id": "math-convert",
  "name": "Math-Convert: Local Image-to-LaTeX",
  "author": "giacomolisita",
  "description": "Convert image regions to LaTeX formulas with local, offline AI inference.",
  "repo": "giacolees/obsidian-ligthweightIm2Tex"
}
```

## Suggested pull request title

```text
Add plugin: Math-Convert: Local Image-to-LaTeX
```

## Suggested pull request body

```md
## Checklist

- [x] I have performed a self-review of my own code.
- [x] I have tested the plugin myself.
- [x] I have checked that all provided links work and are correct.
- [x] I have included all mandatory files in the root of the repository.
- [x] I have created a release with a matching version tag and attached `main.js`, `manifest.json`, and `styles.css`.

## Description

Math-Convert adds a sidebar for converting pasted, dropped, or selected image regions into LaTeX formulas directly inside Obsidian.

Inference runs locally in the desktop app using `@huggingface/transformers` and WebAssembly. The plugin downloads the model from Hugging Face on first use, caches it locally, and then works offline afterward.

The plugin is desktop-only because the inference runtime depends on the Electron desktop environment.
```

## Before opening the PR

1. Confirm `manifest.json` and the Git tag use the same version.
2. Confirm the GitHub release includes `main.js`, `manifest.json`, and `styles.css`.
3. Confirm the README explains first-run model download behavior.
4. Confirm the repository root contains `README.md`, `LICENSE`, and `manifest.json`.

## Notes

- I checked the current official submission instructions on April 19, 2026.
- I did not find an existing `math-convert` plugin entry in the current `community-plugins.json`, but you should still do one last quick search in the live file right before submitting.
