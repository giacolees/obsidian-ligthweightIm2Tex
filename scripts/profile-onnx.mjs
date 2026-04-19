/**
 * Download a HuggingFace model and profile one of its ONNX files with onnxruntime-node.
 *
 * Usage:
 *   node scripts/profile-onnx.mjs [model_id] [onnx_file] [optimization_level]
 *
 * model_id          HuggingFace model ID        (default: alephpi/FormulaNet)
 * onnx_file         Filename to profile, e.g. encoder_model.onnx or model.onnx
 *                   Use "list" to print available ONNX files and exit.
 *                   Defaults to encoder_model.onnx, then the first .onnx found.
 * optimization_level  disabled | basic | extended | all  (default: all)
 */

import { env } from "@huggingface/transformers";
import https from "node:https";
import http from "node:http";
import * as ort from "onnxruntime-node";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
const MODEL_ID  = process.argv[2] ?? "alephpi/FormulaNet";
const ONNX_ARG  = process.argv[3] ?? "auto";
const optLevel  = process.argv[4] ?? "all";

const validLevels = ["disabled", "basic", "extended", "all"];
if (!validLevels.includes(optLevel)) {
	console.error(`optimization_level must be one of: ${validLevels.join(", ")}`);
	process.exit(1);
}

// ---------------------------------------------------------------------------
// Download model via transformers.js (Node.js mode — filesystem cache)
// ---------------------------------------------------------------------------
const CACHE_DIR = path.join(import.meta.dirname, ".model-cache");
env.cacheDir = CACHE_DIR;
env.allowLocalModels = false;

console.log(`\nModel:              ${MODEL_ID}`);
console.log(`ONNX file:          ${ONNX_ARG}`);
console.log(`Optimization level: ${optLevel}`);
console.log(`Cache dir:          ${CACHE_DIR}\n`);

// ---------------------------------------------------------------------------
// Download helpers
// ---------------------------------------------------------------------------
function hfUrl(modelId, filePath, revision = "main") {
	return `${env.remoteHost}${modelId}/resolve/${revision}/${filePath}`;
}

function downloadFile(url, destPath) {
	return new Promise((resolve, reject) => {
		// Follow redirects before opening the WriteStream (XetHub CDN uses 302)
		const follow = (u, hops = 0) => {
			if (hops > 10) return reject(new Error("Too many redirects"));
			const mod = u.startsWith("https") ? https : http;
			mod.get(u, { headers: { "User-Agent": "node/profile-onnx" } }, (res) => {
				if (res.statusCode === 301 || res.statusCode === 302) {
					res.resume(); // drain the tiny redirect body
					return follow(res.headers.location, hops + 1);
				}
				if (res.statusCode === 404) return reject(new Error("404"));
				if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));

				const dest = fs.createWriteStream(destPath);
				const total = Number(res.headers["content-length"] ?? 0);
				let loaded = 0;
				let lastPct = -1;
				let stallTimer = setTimeout(() => { res.destroy(); }, 60_000);
				res.on("data", (chunk) => {
					clearTimeout(stallTimer);
					stallTimer = setTimeout(() => { res.destroy(); }, 60_000);
					loaded += chunk.length;
					if (total) {
						const pct = Math.round((loaded / total) * 100);
						if (pct !== lastPct) {
							lastPct = pct;
							process.stdout.write(`\r  ${path.basename(destPath)} ${pct}%   `);
						}
					}
				});
				res.pipe(dest);
				dest.on("finish", () => { clearTimeout(stallTimer); resolve(destPath); });
				dest.on("error", (e) => { clearTimeout(stallTimer); fs.rmSync(destPath, { force: true }); reject(e); });
				res.on("error", (e) => { clearTimeout(stallTimer); fs.rmSync(destPath, { force: true }); reject(e); });
			}).on("error", (e) => { fs.rmSync(destPath, { force: true }); reject(e); });
		};
		follow(url);
	});
}

function getJson(url, hops = 0) {
	return new Promise((resolve, reject) => {
		if (hops > 10) return reject(new Error("Too many redirects"));
		https.get(url, { timeout: 15_000 }, (res) => {
			if ([301, 302, 307, 308].includes(res.statusCode)) {
				res.resume();
				const loc = res.headers.location;
				const next = loc.startsWith("/") ? new URL(loc, url).href : loc;
				return resolve(getJson(next, hops + 1));
			}
			if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
			let data = "";
			res.on("data", (d) => { data += d; });
			res.on("end", () => { try { resolve(JSON.parse(data)); } catch (e) { reject(e); } });
		}).on("error", reject).on("timeout", () => reject(new Error("timeout")));
	});
}

async function listRepoOnnx(modelId) {
	const meta = await getJson(`${env.remoteHost}api/models/${modelId}`);
	return (meta.siblings ?? []).map((f) => f.rfilename).filter((f) => f.endsWith(".onnx"));
}

async function ensureOnnxFile(modelId, repoPath, destDir) {
	const dest = path.join(destDir, modelId, repoPath);
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
		process.stdout.write(`\r  ${path.basename(dest)} (cached)                    `);
		return dest;
	}
	fs.rmSync(dest, { force: true });
	process.stdout.write(`\r  ${repoPath}…                    `);
	await downloadFile(hfUrl(modelId, repoPath), dest);
	return dest;
}

// ---------------------------------------------------------------------------
console.log("Fetching repo file list…");
const repoOnnxFiles = await listRepoOnnx(MODEL_ID);
if (repoOnnxFiles.length === 0) {
	console.error(`No .onnx files found in ${MODEL_ID}.`);
	process.exit(1);
}

if (ONNX_ARG === "list") {
	console.log("\nAvailable ONNX files:");
	repoOnnxFiles.forEach((f) => console.log(`  ${f}`));
	process.exit(0);
}

// Resolve which repo path to download
let repoFilePath;
if (ONNX_ARG !== "auto") {
	repoFilePath = repoOnnxFiles.find((f) => path.basename(f) === ONNX_ARG || f === ONNX_ARG);
	if (!repoFilePath) {
		console.error(`"${ONNX_ARG}" not found. Run with "list" to see available files.`);
		process.exit(1);
	}
} else {
	repoFilePath =
		repoOnnxFiles.find((f) => path.basename(f) === "encoder_model.onnx") ??
		repoOnnxFiles[0];
}

console.log(`Downloading ${repoFilePath} (skipped if cached)…`);
const targetPath = await ensureOnnxFile(MODEL_ID, repoFilePath, CACHE_DIR);
process.stdout.write("\r  Done.                              \n");

// ---------------------------------------------------------------------------
// Fetch preprocessor config for correct input shape hints
// ---------------------------------------------------------------------------
const inputHints = {};
try {
	const ppCfg = await getJson(hfUrl(MODEL_ID, "preprocessor_config.json"));
	const sz = ppCfg.size;
	if (typeof sz === "number") {
		inputHints.height = sz;
		inputHints.width = sz;
		inputHints.spatial = sz;
	} else if (sz && typeof sz === "object") {
		inputHints.height = sz.height ?? sz.shortest_edge ?? 384;
		inputHints.width = sz.width ?? sz.shortest_edge ?? 384;
		inputHints.spatial = inputHints.height;
	}
	if (ppCfg.num_channels != null) inputHints.num_channels = ppCfg.num_channels;
	if (Object.keys(inputHints).length)
		console.info(`  preprocessor hints: ${JSON.stringify(inputHints)}`);
} catch (e) {
	console.info(`  no preprocessor_config.json (${e.message}) — using defaults`);
}

const optimizedPath = targetPath.replace(/\.onnx$/, `_optimized_${optLevel}.onnx`);
const sizeMb = (fs.statSync(targetPath).size / 1024 / 1024).toFixed(1);
console.log(`Profiling: ${path.basename(targetPath)} (${sizeMb} MB)`);
console.log(`Full path: ${targetPath}`);

// ---------------------------------------------------------------------------
// ORT session
// ---------------------------------------------------------------------------
const session = await ort.InferenceSession.create(targetPath, {
	executionProviders: ["cpu"],
	graphOptimizationLevel: optLevel,
	enableProfiling: true,
	profileFilePrefix: path.join(os.tmpdir(), "ort_profile"),
	optimizedModelFilePath: optimizedPath,
	logSeverityLevel: 2,
});

// ---------------------------------------------------------------------------
// Build dummy inputs from model metadata
// ---------------------------------------------------------------------------
function makeDummy(type, dims) {
	const size = dims.reduce((a, b) => a * b, 1);
	const arrays = {
		float32: Float32Array,
		float16: Uint16Array,
		int64: BigInt64Array,
		int32: Int32Array,
		uint8: Uint8Array,
	};
	const Arr = arrays[type] ?? Float32Array;
	const data =
		type === "int64"
			? new BigInt64Array(size).fill(1n)
			: new Arr(size).fill(type === "float32" ? 0.5 : 1);
	return new ort.Tensor(type, data, dims);
}

function resolveDim(d, hints = {}) {
	if (typeof d === "number" && d > 0) return d;
	if (d === "batch_size" || d === null || d === "batch") return 1;
	if (d === "sequence_length") return 512;
	if (d === "num_channels") return hints.num_channels ?? 3;
	if (d === "height") return hints.height ?? 384;
	if (d === "width") return hints.width ?? 384;
	return hints.spatial ?? 384;
}

const feeds = {};
for (const name of session.inputNames) {
	const meta = session.inputMetadata?.find?.((m) => m.name === name)
		?? session.inputMetadata?.[name];
	const rawShape = meta?.shape ?? meta?.dims ?? [1, 3, 384, 384];
	const dims = rawShape.map((d) => resolveDim(d, inputHints));
	const type = meta?.type ?? "float32";
	console.log(`  input "${name}": ${type} [${dims.join(", ")}]`);
	feeds[name] = makeDummy(type, dims);
}

// ---------------------------------------------------------------------------
// Warm-up + measured run
// ---------------------------------------------------------------------------
console.log("\nWarm-up pass…");
try {
	await session.run(feeds);
} catch (e) {
	// Channel mismatch: model expects grayscale (1 ch) but we sent RGB (3 ch)
	const match = e.message?.match(/kernel channels:\s*(\d+)/);
	if (match) {
		const correctCh = Number(match[1]);
		console.log(`  channel mismatch — retrying with ${correctCh} channel(s)`);
		for (const name of Object.keys(feeds)) {
			const t = feeds[name];
			if (t.dims[1] !== correctCh) {
				const newDims = [...t.dims];
				newDims[1] = correctCh;
				feeds[name] = makeDummy(t.type, newDims);
				console.log(`  input "${name}": ${t.type} [${newDims.join(", ")}]`);
			}
		}
		await session.run(feeds);
	} else {
		throw e;
	}
}

console.log("Measured pass…");
const t0 = performance.now();
await session.run(feeds);
const wallMs = (performance.now() - t0).toFixed(1);

session.endProfiling();

// endProfiling() may return undefined in some ORT versions — find by prefix
const profileFile = fs
	.readdirSync(os.tmpdir())
	.filter((f) => f.startsWith("ort_profile") && f.endsWith(".json"))
	.map((f) => ({ f, mtime: fs.statSync(path.join(os.tmpdir(), f)).mtimeMs }))
	.sort((a, b) => b.mtime - a.mtime)[0]?.f;

if (!profileFile) {
	console.error("Profile JSON not found in", os.tmpdir());
	process.exit(1);
}

const profileFullPath = path.join(os.tmpdir(), profileFile);
console.log(`\nWall time: ${wallMs} ms`);
console.log(`Profile:   ${profileFullPath}`);
console.log(`Optimised: ${optimizedPath}`);

// ---------------------------------------------------------------------------
// Parse + summarise profiling JSON
// ---------------------------------------------------------------------------
const events = JSON.parse(fs.readFileSync(profileFullPath, "utf8"));
const kernelEvents = events.filter(
	(e) => e.cat === "Node" && e.ph === "X" && typeof e.dur === "number",
);

const byOp = new Map();
for (const e of kernelEvents) {
	const op = e.args?.op_name ?? e.name;
	const prev = byOp.get(op) ?? { count: 0, totalUs: 0 };
	byOp.set(op, { count: prev.count + 1, totalUs: prev.totalUs + e.dur });
}

const ranked = [...byOp.entries()]
	.map(([op, { count, totalUs }]) => ({ op, count, totalUs, avgUs: totalUs / count }))
	.sort((a, b) => b.totalUs - a.totalUs);

const totalUs = ranked.reduce((s, r) => s + r.totalUs, 0);

const FUSED = ["Fused", "fused"];

console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
console.log(` Kernel summary  (${kernelEvents.length} calls, total ${(totalUs / 1000).toFixed(1)} ms)`);
console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
console.log(
	`${"Op".padEnd(38)} ${"Calls".padStart(5)} ${"Total(ms)".padStart(10)} ${"Avg(µs)".padStart(9)} ${"Share".padStart(7)}`,
);
console.log("─".repeat(74));

for (const { op, count, totalUs: t, avgUs } of ranked.slice(0, 30)) {
	const isFused = FUSED.some((m) => op.includes(m));
	const share = ((t / totalUs) * 100).toFixed(1);
	const line = `${op.padEnd(38)} ${String(count).padStart(5)} ${(t / 1000).toFixed(2).padStart(10)} ${avgUs.toFixed(0).padStart(9)} ${(share + "%").padStart(7)}`;
	console.log(isFused ? `\x1b[32m${line}  ✦ fused\x1b[0m` : line);
}

console.log("─".repeat(74));

const fusedOps = ranked.filter(({ op }) => FUSED.some((m) => op.includes(m)));
const fusedUs = fusedOps.reduce((s, r) => s + r.totalUs, 0);

if (fusedOps.length === 0) {
	console.log("\n\x1b[33m⚠  No fused kernels detected.\x1b[0m");
	console.log("   Open the optimised ONNX in Netron to inspect the graph after ORT's optimizer ran.");
} else {
	console.log(
		`\n\x1b[32m✦  ${fusedOps.length} fused op type(s) — ${((fusedUs / totalUs) * 100).toFixed(1)}% of kernel time.\x1b[0m`,
	);
}

console.log(`\nOpen in Netron to compare graphs:`);
console.log(`  original:  ${targetPath}`);
console.log(`  optimised: ${optimizedPath}\n`);

process.exit(0);
