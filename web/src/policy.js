// Main-thread side of the learned policy: talks to the inference worker and turns action
// chunks into per-step joint targets ("execute the first N steps of a chunk, then look again").

export class PolicyClient {
  constructor() {
    this.worker = new Worker(new URL('./policy_worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.info = null;
    this.worker.onmessage = ({ data }) => {
      const { resolve, reject } = this.pending.get(data.id);
      this.pending.delete(data.id);
      if (data.ok) resolve(data.result);
      else reject(new Error(data.error));
    };
  }

  call(message, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }

  async init(options) {
    this.info = await this.call({ type: 'init', ...options });
    return this.info;
  }

  infer(images, joints) {
    return this.call({ type: 'infer', images, joints }, images.map((a) => a.buffer));
  }
}

// Mirrors keigrasp/policy.py. `observe()` must return { images: [RGBA per camera], joints }.
export class ChunkRunner {
  constructor(client, { chunk, execute = 10 }) {
    this.client = client;
    this.chunk = chunk;
    this.execute = execute;
    this.reset();
  }

  reset() {
    this.plan = null; // Float32Array(chunk * 6)
    this.planStep = 0; // sim step at which the plan's observation was taken
    this.inFlight = null;
    this.generation = (this.generation ?? 0) + 1;
    this.lastMs = 0;
    this.inferences = 0;
    this.totalMs = 0;
  }

  _request(step, observe) {
    const { images, joints } = observe();
    const generation = this.generation;
    this.inFlight = this.client.infer(images, joints).then(({ actions, ms }) => {
      if (generation !== this.generation) return; // the episode was reset meanwhile
      this.plan = actions;
      this.planStep = step;
      this.lastMs = ms;
      this.inferences += 1;
      this.totalMs += ms;
    }).finally(() => {
      if (generation === this.generation) this.inFlight = null;
    });
    return this.inFlight;
  }

  _action(step) {
    const k = Math.min(step - this.planStep, this.chunk - 1);
    return Array.from(this.plan.subarray(6 * k, 6 * k + 6));
  }

  // Synchronous-in-simulation-time mode (evaluation): the simulation waits for each inference,
  // exactly like the Python harness.
  async actBlocking(step, observe) {
    if (!this.plan || step - this.planStep >= this.execute) await this._request(step, observe);
    return this._action(step);
  }

  // Real-time mode (the demo): never blocks. Returns null until the first plan exists; after
  // that it keeps following the current plan while a newer one is being computed. A new plan
  // is indexed by the step its observation was taken at, so inference latency is skipped over.
  actRealtime(step, observe) {
    if (!this.inFlight && (!this.plan || step - this.planStep >= this.execute)) this._request(step, observe);
    return this.plan ? this._action(step) : null;
  }
}
