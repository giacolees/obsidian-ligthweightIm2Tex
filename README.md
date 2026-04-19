# Math-Convert: Local Image-to-LaTeX

**Snap a photo of any equation. Get clean LaTeX. Never leave Obsidian.**

Math-Convert runs a full AI model directly on your machine — no cloud, no subscription, no data leaving your device. Photograph a textbook, screenshot a paper, or paste from your clipboard, and within seconds you have publication-ready LaTeX to drop straight into your notes.

---

## Why Math-Convert?

Retyping equations is tedious and error-prone. Online converters are slow, require accounts, and send your work to someone else's server. Math-Convert is different:

- **Fully offline after the first download** — the model runs locally via WebAssembly
- **Privacy-first** — your images never leave your machine
- **Zero friction** — lives right inside Obsidian, no tab-switching required
- **Instant on repeat use** — the model is cached after the first run (~100 MB, one-time)

---

## How it works

Math-Convert embeds [FormulaNet](https://huggingface.co/alephpi/FormulaNet), a 20 M-parameter vision-to-sequence model, and runs it locally using `@huggingface/transformers` over WebAssembly. The model is fetched from Hugging Face on first use and cached automatically — every subsequent conversion is instant and requires no internet connection.

---

## Usage

1. **Open the sidebar** — click the ribbon icon or run *Open Math-Convert sidebar* from the command palette.
2. **Load an image** — drag and drop, click **Browse**, or paste directly from your clipboard.
3. **Select your formula** — draw a rectangle around the region you care about. Skip this to convert the whole image.
4. **Click Detect formula** — the first run downloads the model with a progress bar; after that it's instant.
5. **Use your LaTeX** — hit **Copy** and paste it anywhere in your vault.

---

## Settings

| Setting | Default | Description |
|---|---|---|
| Model ID | `alephpi/FormulaNet` | Hugging Face model ID. Swap in any compatible VisionEncoderDecoder checkpoint to experiment with alternative models. |

---

## Installation

### From source

```bash
git clone https://github.com/giacolees/obsidian-ligthweightIm2Tex
cd obsidian-ligthweightIm2Tex
npm install
npm run build
```

Copy (or symlink) the repo folder into your vault's `.obsidian/plugins/math-convert/`, then enable it under **Settings → Community plugins**.

---

## Development

```bash
npm run dev   # watch mode — rebuilds main.js on every change
npm run build # type-check + production build
```

See [CLAUDE.md](CLAUDE.md) for architecture notes.

---

## License

MIT
