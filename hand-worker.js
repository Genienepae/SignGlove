import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

let detector = null;
const backend = 'CPU';

async function init() {
  if (detector) {
    self.postMessage({ type: 'READY', backend });
    return;
  }

  self.postMessage({ type: 'STATUS', message: 'Carregando MediaPipe…' });
  const files = await FilesetResolver.forVisionTasks(WASM_URL);

  self.postMessage({ type: 'STATUS', message: 'Carregando modelo de mão…' });
  detector = await HandLandmarker.createFromOptions(files, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'CPU'
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  self.postMessage({ type: 'READY', backend });
}

self.onmessage = async (event) => {
  const data = event.data || {};

  try {
    if (data.type === 'INIT') {
      await init();
      return;
    }

    if (data.type === 'DETECT') {
      if (!detector) await init();

      const bitmap = data.bitmap;
      const started = performance.now();
      const detected = detector.detectForVideo(bitmap, data.timestamp);
      const inferenceTime = performance.now() - started;
      bitmap.close();

      self.postMessage({
        type: 'RESULT',
        landmarks: detected.landmarks || [],
        inferenceTime,
        backend
      });
    }
  } catch (error) {
    try { data.bitmap?.close?.(); } catch {}

    self.postMessage({
      type: 'ERROR',
      error: error?.message || String(error)
    });
  }
};
