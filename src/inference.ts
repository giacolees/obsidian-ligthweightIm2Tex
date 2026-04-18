import {
	cat,
	env,
	PreTrainedTokenizer,
	type ProgressInfo,
	Tensor,
	VisionEncoderDecoderModel,
} from "@huggingface/transformers";

const TARGET_SIZE = 384;
const NORM_MEAN = 0.7931;
const NORM_STD = 0.1738;

env.allowLocalModels = false;

let _model: VisionEncoderDecoderModel | null = null;
let _tokenizer: PreTrainedTokenizer | null = null;
let _loadingPromise: Promise<void> | null = null;
let _loadedModelId: string | null = null;
let _loadingModelId: string | null = null;

export function resetModel() {
	_model = null;
	_tokenizer = null;
	_loadingPromise = null;
	_loadedModelId = null;
	_loadingModelId = null;
}

export async function ensureModel(
	modelId: string,
	onProgress: (msg: string, pct?: number) => void,
): Promise<void> {
	if (_model && _tokenizer && _loadedModelId === modelId) return;

	if (_loadingPromise && _loadingModelId !== modelId) {
		try {
			await _loadingPromise;
		} catch {
			// The previous load failed; we can proceed with the requested model below.
		}
	}

	if (_model && _tokenizer && _loadedModelId === modelId) return;

	if (_model && _tokenizer && _loadedModelId !== modelId) {
		resetModel();
	}

	if (!_loadingPromise) {
		_loadingModelId = modelId;
		_loadingPromise = (async () => {
			const model = await VisionEncoderDecoderModel.from_pretrained(modelId, {
				dtype: "fp32",
				progress_callback: (info: ProgressInfo) => {
					const { msg, pct } = parseProgress(info);
					onProgress(msg, pct);
				},
			});
			const tokenizer = await PreTrainedTokenizer.from_pretrained(modelId);
			_model = model;
			_tokenizer = tokenizer;
			_loadedModelId = modelId;
		})();

		_loadingPromise
			.catch(() => {
				resetModel();
			})
			.finally(() => {
				_loadingPromise = null;
				_loadingModelId = null;
			});
	}

	await _loadingPromise;

	if (!_model || !_tokenizer || _loadedModelId !== modelId) {
		throw new Error(`Model "${modelId}" is not ready.`);
	}
}

export function isModelLoaded(): boolean {
	return !!(_model && _tokenizer);
}

function parseProgress(info: ProgressInfo): { msg: string; pct?: number } {
	if (info.status === "progress") {
		const i = info as unknown as { loaded: number; total: number; file?: string };
		const pct = i.total ? Math.round((i.loaded / i.total) * 100) : undefined;
		return { msg: `Downloading${i.file ? ` ${i.file}` : ""}…`, pct };
	}
	if (info.status === "initiate" || info.status === "download")
		return { msg: "Downloading model…" };
	if (info.status === "done") return { msg: "Loading weights…" };
	return { msg: "Initialising…" };
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
	const c = document.createElement("canvas");
	c.width = w;
	c.height = h;
	return c;
}

function get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
	const context = canvas.getContext("2d");
	if (!context) {
		throw new Error("Could not get a 2D canvas context.");
	}
	return context;
}

export async function preprocessDataUrl(dataUrl: string): Promise<Float32Array> {
	const img = await new Promise<HTMLImageElement>((res, rej) => {
		const i = new Image();
		i.onload = () => res(i);
		i.onerror = rej;
		i.src = dataUrl;
	});

	const w = img.width,
		h = img.height;
	const oc = makeCanvas(w, h);
	const ctx = get2dContext(oc);
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, w, h);
	ctx.drawImage(img, 0, 0);
	const rgba = ctx.getImageData(0, 0, w, h).data;

	const grey = new Uint8Array(w * h);
	for (let i = 0; i < grey.length; i++) {
		grey[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
	}

	const histogram = new Uint32Array(256);
	for (const v of grey) histogram[v]++;
	const darkPx = histogram.slice(0, 200).reduce((s, v) => s + v, 0);
	const lightPx = histogram.slice(200).reduce((s, v) => s + v, 0);
	if (darkPx >= lightPx) for (let i = 0; i < grey.length; i++) grey[i] = 255 - grey[i];

	const threshold = 200;
	let minX = w,
		minY = h,
		maxX = 0,
		maxY = 0,
		hasContent = false;
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			if (grey[y * w + x] < threshold) {
				if (x < minX) minX = x;
				if (x > maxX) maxX = x;
				if (y < minY) minY = y;
				if (y > maxY) maxY = y;
				hasContent = true;
			}
		}
	}
	if (!hasContent) {
		minX = 0;
		minY = 0;
		maxX = w - 1;
		maxY = h - 1;
	}

	const cropW = maxX - minX + 1;
	const cropH = maxY - minY + 1;

	const cropData = new ImageData(cropW, cropH);
	for (let y = 0; y < cropH; y++) {
		for (let x = 0; x < cropW; x++) {
			const v = grey[(y + minY) * w + (x + minX)];
			const idx = (y * cropW + x) * 4;
			cropData.data[idx] = v;
			cropData.data[idx + 1] = v;
			cropData.data[idx + 2] = v;
			cropData.data[idx + 3] = 255;
		}
	}
	const cc = makeCanvas(cropW, cropH);
	get2dContext(cc).putImageData(cropData, 0, 0);

	const scale = Math.min(TARGET_SIZE / cropW, TARGET_SIZE / cropH);
	const newW = Math.round(cropW * scale);
	const newH = Math.round(cropH * scale);
	const padX = Math.floor((TARGET_SIZE - newW) / 2);
	const padY = Math.floor((TARGET_SIZE - newH) / 2);

	const rc = makeCanvas(TARGET_SIZE, TARGET_SIZE);
	const rctx = get2dContext(rc);
	rctx.fillStyle = "white";
	rctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
	rctx.drawImage(cc, padX, padY, newW, newH);

	const outRgba = rctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
	const result = new Float32Array(TARGET_SIZE * TARGET_SIZE);
	for (let i = 0; i < result.length; i++) {
		result[i] = (outRgba[i * 4] / 255 - NORM_MEAN) / NORM_STD;
	}
	return result;
}

export async function runInference(dataUrl: string): Promise<string> {
	if (!_model || !_tokenizer) {
		throw new Error("Model is not loaded.");
	}

	const array = await preprocessDataUrl(dataUrl);
	const t = new Tensor("float32", array, [1, 1, TARGET_SIZE, TARGET_SIZE]);
	const pixelValues = cat([t, t, t], 1);
	const outputs = await _model.generate({ inputs: pixelValues });
	const tok = _tokenizer;
	const raw = (
		tok.batch_decode(outputs as Parameters<typeof tok.batch_decode>[0], {
			skip_special_tokens: true,
		}) as string[]
	)[0];
	return raw.replace(/\\!/g, "");
}
