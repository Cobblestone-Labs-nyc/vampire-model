// Stage-2 in-browser inference (Deploy A) — ONNX Runtime Web / WebGPU.
//
// Drop-in replacement for the `fetch('/api/restyle', ...)` call in index.html. Keeps the
// EXACT contract: takes a webcam snapshot, returns a restyled still — so the attention
// trigger -> align -> hold -> fade -> cooldown flow is untouched.
//
// Pre/post-processing MUST match training (train_student.py): input RGB, [0,1], NCHW,
// square at config.data.resolution (default 512). Output same layout, sigmoid [0,1].
//
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js"></script>
//   <script type="module">
//     import { createVampireRestyler } from './vampire/infer.js';
//     const restyle = await createVampireRestyler('./vampire/model/vampire_student_int8.onnx', 512);
//     // ...on grab:
//     const styledImg = await restyle(snapshotCanvasOrImage, strength01);
//   </script>

export async function createVampireRestyler(modelUrl, res = 512) {
  const ort = window.ort;
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['webgpu', 'wasm'],   // WebGPU first, WASM fallback
    graphOptimizationLevel: 'all',
  });
  const hasStrength = session.inputNames.includes('strength');

  // square center-cropped pre-process into a CHW float32 tensor in [0,1]
  function preprocess(srcImageOrCanvas) {
    const c = document.createElement('canvas');
    c.width = res; c.height = res;
    const ctx = c.getContext('2d');
    ctx.drawImage(srcImageOrCanvas, 0, 0, res, res);
    const { data } = ctx.getImageData(0, 0, res, res); // RGBA, row-major
    const chw = new Float32Array(3 * res * res);
    const plane = res * res;
    for (let i = 0; i < plane; i++) {
      chw[i] = data[i * 4] / 255;                 // R
      chw[plane + i] = data[i * 4 + 1] / 255;     // G
      chw[2 * plane + i] = data[i * 4 + 2] / 255; // B
    }
    return new ort.Tensor('float32', chw, [1, 3, res, res]);
  }

  function postprocess(outTensor) {
    const d = outTensor.data; // CHW [0,1]
    const plane = res * res;
    const c = document.createElement('canvas');
    c.width = res; c.height = res;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(res, res);
    for (let i = 0; i < plane; i++) {
      img.data[i * 4] = Math.min(255, Math.max(0, d[i] * 255));
      img.data[i * 4 + 1] = Math.min(255, Math.max(0, d[plane + i] * 255));
      img.data[i * 4 + 2] = Math.min(255, Math.max(0, d[2 * plane + i] * 255));
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c; // a canvas the page can draw/align exactly like the old returned image
  }

  // strength01: map the page's 0.30..0.85 slider -> 0..1 before calling (same as server)
  return async function restyle(srcImageOrCanvas, strength01 = 0.6) {
    const feeds = { x: preprocess(srcImageOrCanvas) };
    if (hasStrength) feeds.strength = new ort.Tensor('float32', new Float32Array([strength01]), [1]);
    const t0 = performance.now();
    const results = await session.run(feeds);
    const ms = performance.now() - t0;
    const outName = session.outputNames[0];
    const canvas = postprocess(results[outName]);
    canvas.dataset && (canvas.dataset.inferMs = ms.toFixed(0));
    return canvas;
  };
}
