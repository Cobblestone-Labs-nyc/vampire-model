// Drop-in entry for the ENGINES array in /var/www/christopher/server.js (handoff §3, §8).
//
// Adds the Stage-1 vampire model as PRIMARY, keeping ip-adapter-face-id as FALLBACK so the
// existing error-path reset + re-arm behavior is preserved. The front-end is unchanged: it
// still POSTs { image, strength } to /api/restyle and reads back a single restyled still.
//
// Wiring: put STAGE1_URL in /var/www/christopher/.env (e.g. http://127.0.0.1:8008 or the
// GPU endpoint), then `pm2 restart christopher-checklist && pm2 save`.

const VAMPIRE_TURBO = {
  name: 'vampire-turbo-instantid',          // log this as `engine` in restyle.log
  url: `${process.env.STAGE1_URL}/api/restyle`,
  identity: true,                            // InstantID-locked
  timeoutMs: 12000,                          // few-step turbo; well under the 90s SDXL budget
  // body builder: the snapshot + slider map straight through (server does the slider math)
  buildBody: ({ imageUrlOrDataUrl, strength }) => ({
    image: imageUrlOrDataUrl,                // data URL or image URL, same as today
    strength,                                // front-end 0.30–0.85; server maps to turbo knobs
  }),
  // response reader: server returns { image: { b64 | url } }
  readImage: (json) => json?.image?.b64 || json?.image?.url || null,
  // log fields to append to restyle.log (JSON line), matching existing convention
  logFields: (json) => ({ engine: 'vampire-turbo-instantid', identity: true, ms: json?.ms }),
};

// Existing fallback, unchanged — used only if the primary errors/times out.
const IP_ADAPTER_FACE_ID = {
  name: 'ip-adapter-face-id',
  url: 'https://fal.run/fal-ai/ip-adapter-face-id',
  identity: true,
  timeoutMs: 90000,
  // ...existing body/readImage as in the live server...
};

// ENGINES = [PRIMARY, FALLBACK]
module.exports = { ENGINES: [VAMPIRE_TURBO, IP_ADAPTER_FACE_ID] };

// --- Deploy A (in-browser) note ---------------------------------------------------
// If/when the Stage-2 student ships, the restyle becomes a LOCAL inference in index.html
// (see ../web/infer.js) and this server entry is bypassed entirely for those clients —
// /api/restyle stays as the server-side fallback for browsers without WebGPU.
