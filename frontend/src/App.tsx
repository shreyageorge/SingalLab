import { useEffect, useState, useRef, type ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BrainCircuit,
  Check,
  ChevronRight,
  Clock3,
  Code2,
  Database,
  ExternalLink,
  ImagePlus,
  LoaderCircle,
  Menu,
  ScanLine,
  ShieldCheck,
  UploadCloud,
  X,
  TriangleAlert,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
} from "recharts";

type Prediction = {
  class_id: number;
  class_name: string;
  score: number;
  inference_ms: number;
  top_k: { class_id: number; class_name: string; score: number }[];
};
type Health = {
  model_available: boolean;
  model_status: string;
  message: string;
};
const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const apiUrl = (path: string) => `${API_BASE}/api/${path.replace(/^\//, "")}`;
const api = async (path: string, options: RequestInit = {}) => {
  const r = await fetch(apiUrl(path), {
    ...options,
    signal: AbortSignal.timeout(30000),
  });
  const d = await r
    .json()
    .catch(() => ({ error: "Backend returned an invalid response." }));
  if (!r.ok) throw Error(d.error || `Request failed (${r.status})`);
  return d;
};
function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Database size={24} />
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
const nav = [
  ["Recognition", ScanLine],
  ["Model & Training", BrainCircuit],
  ["Dataset", Database],
] as const;
function App() {
  const [page, setPage] = useState("Recognition"),
    [mobile, setMobile] = useState(false);
  const [health, setHealth] = useState<Health | null>(null),
    [backendError, setBackendError] = useState("");
  const [metrics, setMetrics] = useState<any>(null),
    [model, setModel] = useState<any>(null),
    [samples, setSamples] = useState<any[]>([]);
  const [image, setImage] = useState(""),
    [file, setFile] = useState<File | null>(null),
    [filename, setFilename] = useState("");
  const [result, setResult] = useState<Prediction | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const refresh = () => {
    api("health")
      .then((d) => {
        setHealth(d);
        setBackendError("");
      })
      .catch(() => {
        setHealth(null);
        setBackendError(
          "The Python API is unavailable. Check the backend connection.",
        );
      });
    api("metrics")
      .then(setMetrics)
      .catch(() => setMetrics(null));
    api("model-info")
      .then(setModel)
      .catch(() => setModel(null));
    api("samples")
      .then((d) => setSamples(d.samples))
      .catch(() => setSamples([]));
  };
  useEffect(() => {
    refresh();
  }, []);
  useEffect(() => {
    const handle = () => {
      let route = "";
      try {
        route = decodeURIComponent(location.hash.slice(1));
      } catch {}
      const legacy: Record<string, string> = {
        "Recognition Studio": "Recognition",
        Training: "Model & Training",
        "AI Model Lab": "Model & Training",
        Evaluation: "Model & Training",
        "Dataset Explorer": "Dataset",
        "Sign Classes": "Dataset",
      };
      const target = legacy[route] || route;
      if (nav.some(([name]) => name === target)) {
        setPage(target);
        if (target !== route)
          history.replaceState(null, "", `#${encodeURIComponent(target)}`);
      } else {
        setPage("Recognition");
        history.replaceState(null, "", "#Recognition");
      }
    };
    handle();
    window.addEventListener("hashchange", handle);
    return () => window.removeEventListener("hashchange", handle);
  }, []);
  function go(p: string) {
    setPage(p);
    location.hash = encodeURIComponent(p);
    setMobile(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  async function choose(f: File) {
    if (busy) return;
    setError("");
    setResult(null);
    if (f.size > 4 * 1024 * 1024) {
      setError("Choose an image under 4 MB.");
      return;
    }
    if (!/\.(jpe?g|png|webp|bmp|ppm)$/i.test(f.name)) {
      setError("Choose JPEG, PNG, WebP, BMP or PPM.");
      return;
    }
    if (/\.ppm$/i.test(f.name)) {
      try {
        const body = new FormData();
        body.append("image", f);
        const response = await fetch(apiUrl("preview"), {
          method: "POST",
          body,
          signal: AbortSignal.timeout(30000),
        });
        if (!response.ok) {
          const d = await response.json();
          throw Error(d.error || "Cannot preview image.");
        }
        const blob = await response.blob();
        f = new File([blob], f.name.replace(/\.ppm$/i, ".png"), {
          type: "image/png",
        });
      } catch (e) {
        setError((e as Error).message);
        return;
      }
    }
    const reader = new FileReader();
    reader.onload = () => {
      setFile(f);
      setFilename(f.name);
      setImage(reader.result as string);
    };
    reader.onerror = () => setError("Unable to read this file.");
    reader.readAsDataURL(f);
  }
  async function useSample(s: any) {
    try {
      const response = await fetch(`${API_BASE}${s.url}`);
      if (!response.ok) throw Error("Dataset sample unavailable.");
      const blob = await response.blob();
      await choose(
        new File([blob], `gtsrb-class-${s.class_id}.png`, {
          type: "image/png",
        }),
      );
      go("Recognition");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function recognize() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const d: Prediction = await api("predict", {
        method: "POST",
        body: form,
      });
      setResult(d);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobile ? "visible" : ""}`}>
        <a
          className="brand"
          href="#Recognition"
          onClick={() => go("Recognition")}
        >
          <span className="brand-icon">
            <ScanLine size={23} />
          </span>
          Signal<span>Lab</span>
          <small>AI</small>
        </a>
        <div className="workspace-label">COMPUTER VISION WORKSPACE</div>
        <nav>
          {nav.map(([name, Icon]) => (
            <div key={name}>
              <button
                className={page === name ? "active" : ""}
                onClick={() => go(name)}
              >
                <Icon size={18} />
                <span>{name}</span>
                {page === name && <i />}
              </button>
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <p>Built to learn. Designed to explore.</p>
          <div className="sidebar-profile">
            <span>AI</span>
            <div>
              Artificial Intelligence<small>Academic workspace</small>
            </div>
            <ShieldCheck size={16} />
          </div>
        </div>
      </aside>
      {mobile && <div className="scrim" onClick={() => setMobile(false)} />}
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="mobile-menu icon-button"
              aria-label="Toggle navigation"
              onClick={() => setMobile(!mobile)}
            >
              <Menu />
            </button>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <b>{page}</b>
          </div>
          <div className="top-actions">
            <button
              className="icon-button"
              onClick={refresh}
              title="Refresh backend and artifacts"
              aria-label="Refresh backend and artifacts"
            >
              <Activity size={18} />
            </button>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <h1>{page}</h1>
              <p>
                {page === "Recognition"
                  ? "Upload a cropped traffic sign and see the model’s prediction."
                  : page === "Model & Training"
                    ? "How the CNN works, how to train it, and what the measured results show."
                    : "The data behind the model: GTSRB, its classes and image preparation."}
              </p>
            </div>
            <span className="page-tag">
              <BrainCircuit size={15} /> CNN · GTSRB
            </span>
          </div>
          {page === "Recognition" && (
            <>
              <div className="studio-banner">
                <div className="banner-icon">
                  <ScanLine size={23} />
                </div>
                <div>
                  <b>A small sign. A smarter perspective.</b>
                  <p>
                    Upload a cropped traffic sign and let the CNN classify it
                    across 43 categories.
                  </p>
                </div>
                <span>
                  32 × 32 RGB <ArrowRight size={14} /> 43 CLASSES
                </span>
              </div>
              <div className="studio-grid">
                <section className="panel upload-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="step">01</span>
                      <h2>Input image</h2>
                    </div>
                    <span className="muted">Cropped traffic sign</span>
                  </div>
                  <div
                    className={`dropzone ${drag ? "dragging" : ""} ${image ? "has-image" : ""}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDrag(true);
                    }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDrag(false);
                      if (e.dataTransfer.files[0])
                        void choose(e.dataTransfer.files[0]);
                    }}
                  >
                    {image ? (
                      <>
                        <img src={image} alt="Selected traffic sign" />
                        <button
                          className="remove-image icon-button"
                          aria-label="Remove image"
                          disabled={busy}
                          onClick={() => {
                            setImage("");
                            setFile(null);
                            setResult(null);
                            setError("");
                          }}
                        >
                          <X size={17} />
                        </button>
                        <span className="filename">{filename}</span>
                      </>
                    ) : (
                      <>
                        <div className="upload-icon">
                          <UploadCloud size={30} />
                        </div>
                        <h3>Drop your traffic sign here</h3>
                        <p>or choose an image from your device</p>
                        <button
                          className="secondary"
                          onClick={() => input.current?.click()}
                        >
                          <ImagePlus size={16} />
                          Browse files
                        </button>
                        <small>JPG, PNG, WebP, BMP, PPM · up to 4 MB</small>
                      </>
                    )}
                    <input
                      ref={input}
                      type="file"
                      accept=".jpg,.jpeg,.png,.webp,.bmp,.ppm"
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files?.[0]) void choose(e.target.files[0]);
                        e.target.value = "";
                      }}
                    />
                  </div>
                  {image && (
                    <button
                      className="text-button replace"
                      disabled={busy}
                      onClick={() => input.current?.click()}
                    >
                      Replace image
                    </button>
                  )}
                  {samples.length > 0 ? (
                    <div className="real-samples">
                      <label htmlFor="dataset-sample">
                        Try an actual dataset image
                      </label>
                      <select
                        id="dataset-sample"
                        value=""
                        onChange={(e) => {
                          const s = samples.find(
                            (s) => s.class_id === Number(e.target.value),
                          );
                          if (s) void useSample(s);
                        }}
                      >
                        <option value="">Select a GTSRB example…</option>
                        {samples.map((s) => (
                          <option key={s.class_id} value={s.class_id}>
                            {s.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : (
                    <p className="sample-note">
                      Upload a cropped sign to classify it across 43 GTSRB
                      classes.
                    </p>
                  )}
                  <button
                    className="primary recognize"
                    disabled={!file || busy}
                    onClick={() => void recognize()}
                  >
                    {busy ? (
                      <LoaderCircle className="spin" size={18} />
                    ) : (
                      <ScanLine size={18} />
                    )}{" "}
                    {busy
                      ? "Running model inference…"
                      : "Recognize Traffic Sign"}
                    <ArrowRight size={18} />
                  </button>
                  <div className="privacy">
                    <ShieldCheck size={13} /> Processed by the Python API. No
                    permanent upload storage.
                  </div>
                </section>
                <section className="panel result-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="step">02</span>
                      <h2>Recognition result</h2>
                    </div>
                    <span className={`pill ${result ? "green" : ""}`}>
                      <i />
                      {result ? "Prediction complete" : "Awaiting image"}
                    </span>
                  </div>
                  {result ? (
                    <div className="prediction">
                      <div className="predicted-sign">
                        <img src={image} alt="Classified image" />
                        <div>
                          <span className="eyebrow">
                            PREDICTED CLASS · {result.class_id}
                          </span>
                          <h2>{result.class_name}</h2>
                          <p className="score">
                            {(result.score * 100).toFixed(2)}%{" "}
                            <span>softmax score</span>
                          </p>
                        </div>
                      </div>
                      <div className="timing">
                        <Clock3 size={14} />
                        {result.inference_ms} ms model inference
                      </div>
                      <h3>Top 5 predictions</h3>
                      <ResponsiveContainer width="100%" height={240}>
                        <BarChart
                          data={result.top_k.map((x) => ({
                            ...x,
                            percent: x.score * 100,
                          }))}
                          layout="vertical"
                          margin={{ left: 0, right: 25 }}
                        >
                          <XAxis
                            type="number"
                            domain={[0, 100]}
                            tickFormatter={(v) => `${v}%`}
                          />
                          <YAxis
                            type="category"
                            dataKey="class_name"
                            width={145}
                            tick={{ fontSize: 11 }}
                          />
                          <Tooltip
                            formatter={(v: number) => `${v.toFixed(2)}%`}
                          />
                          <Bar dataKey="percent" radius={[0, 4, 4, 0]}>
                            {result.top_k.map((x, i) => (
                              <Cell
                                key={x.class_id}
                                fill={i === 0 ? "#4ecdc4" : "#334a65"}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="caption">
                        Softmax scores describe model output; they do not
                        guarantee correctness.
                      </p>
                    </div>
                  ) : (
                    <div className="result-empty">
                      <div className="viewfinder">
                        <ScanLine size={48} />
                        <span />
                      </div>
                      <h3>Your prediction starts here</h3>
                      <p>
                        Upload a traffic-sign image, then select
                        <br />
                        <b>Recognize Traffic Sign</b> to see the results.
                      </p>
                      <div className="empty-chips">
                        <span>Class label</span>
                        <span>Softmax score</span>
                        <span>Top 5 predictions</span>
                      </div>
                    </div>
                  )}
                  <div className="model-notice">
                    <TriangleAlert size={18} />
                    <div>
                      <b>
                        {health?.model_available
                          ? "Model ready for inference"
                          : health
                            ? "Trained model not configured"
                            : "Connect your Python backend"}
                      </b>
                      <p>
                        {health?.model_available
                          ? "Actual predictions are generated by your loaded Keras model."
                          : health?.message ||
                            backendError ||
                            "Checking the API…"}
                      </p>
                      <button
                        className="text-button"
                        onClick={() => go("Model & Training")}
                      >
                        View setup instructions <ArrowUpRight size={13} />
                      </button>
                    </div>
                  </div>
                </section>
              </div>
              {error && (
                <div role="alert" className="error">
                  <TriangleAlert size={18} />
                  {error}
                </div>
              )}
            </>
          )}

          {page === "Model & Training" && (
            <>
              <section className="panel prose">
                <div className="eyebrow">CONVOLUTIONAL NEURAL NETWORK</div>
                <h2>How a picture becomes a prediction</h2>
                <p>
                  The model learns patterns in labelled traffic-sign images.
                  Convolutions detect features such as edges and shapes, pooling
                  reduces their spatial size, and dense layers assign a score to
                  each of the 43 classes.
                </p>
                <div className="workflow compact-pipeline">
                  {[
                    "32 × 32 RGB",
                    "Rescale to 0–1",
                    "3 convolution blocks",
                    "Pooling + flatten",
                    "Dense + dropout",
                    "43 softmax scores",
                  ].map((label, i) => (
                    <div key={label}>
                      <span>{i + 1}</span>
                      {label}
                      {i < 5 && <ArrowRight size={15} />}
                    </div>
                  ))}
                </div>
                <p className="caption">
                  Each block uses a 3 × 3 convolution with ReLU, then max
                  pooling. Filters: 32 → 64 → 128. Dense layer: 128 units.
                  Dropout: 0.5. Optimizer: Adam. Loss: sparse categorical
                  cross-entropy.
                </p>
                {model?.available ? (
                  <p>
                    Loaded model: {model.parameters.toLocaleString()} parameters
                    · Input {JSON.stringify(model.input_shape)} · Output{" "}
                    {JSON.stringify(model.output_shape)}
                  </p>
                ) : (
                  <p className="notice">
                    Configured baseline shown above. No trained model is
                    currently loaded.
                  </p>
                )}
              </section>
              <section className="panel prose">
                <h2>Training results</h2>
                <p>
                  Measured results from saved training artifacts. The validation
                  set helps select the model; it is not an independent test set.
                </p>
                {!metrics?.history?.available &&
                !metrics?.evaluation?.available ? (
                  <Empty title="No measured results yet">
                    Accuracy, loss and training curves will appear after real
                    training artifacts are available. No example scores are
                    substituted.
                  </Empty>
                ) : (
                  <>
                    <div className="stat-grid">
                      {[
                        [
                          "Training accuracy",
                          metrics?.history?.data?.accuracy?.slice(-1)[0],
                          true,
                        ],
                        [
                          "Validation accuracy",
                          metrics?.evaluation?.data?.accuracy,
                          true,
                        ],
                        [
                          "Validation loss",
                          metrics?.evaluation?.data?.loss,
                          false,
                        ],
                        [
                          "Best epoch",
                          metrics?.history?.data?.best_epoch,
                          false,
                        ],
                      ].map(([label, value, percent]) => (
                        <div className="panel stat" key={String(label)}>
                          <span>{label}</span>
                          <b>
                            {typeof value === "number"
                              ? percent
                                ? `${(value * 100).toFixed(2)}%`
                                : value.toFixed(label === "Best epoch" ? 0 : 4)
                              : "Unavailable"}
                          </b>
                        </div>
                      ))}
                    </div>
                    {metrics.history?.available && (
                      <div className="two-col">
                        {["accuracy", "loss"].map((key) => (
                          <div key={key}>
                            <h3>
                              {key === "accuracy" ? "Accuracy" : "Loss"} by
                              epoch
                            </h3>
                            <ResponsiveContainer width="100%" height={230}>
                              <LineChart
                                data={metrics.history.data[key].map(
                                  (value: number, i: number) => ({
                                    epoch: i + 1,
                                    training: value,
                                    validation:
                                      metrics.history.data[`val_${key}`][i],
                                  }),
                                )}
                              >
                                <CartesianGrid
                                  stroke="#273242"
                                  strokeDasharray="3 3"
                                />
                                <XAxis dataKey="epoch" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line
                                  dot={false}
                                  dataKey="training"
                                  stroke="#4cc8c8"
                                />
                                <Line
                                  dot={false}
                                  dataKey="validation"
                                  stroke="#8b93ff"
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        ))}
                      </div>
                    )}
                    <p className="caption">
                      Training accuracy is from the last epoch; validation
                      evaluation uses the best saved checkpoint.
                    </p>
                  </>
                )}
              </section>
            </>
          )}
          {page === "Dataset" && (
            <>
              <section className="panel prose dataset-intro">
                <span className="eyebrow">THE BENCHMARK</span>
                <h2>
                  German Traffic Sign
                  <br />
                  Recognition Benchmark
                </h2>
                <p>
                  A labelled image dataset for multi-class traffic-sign
                  recognition. This baseline uses numeric class IDs 0–42 and
                  learns from cropped RGB images.
                </p>
                <a
                  className="secondary"
                  href="https://benchmark.ini.rub.de/gtsrb_dataset.html"
                  target="_blank"
                  rel="noreferrer"
                >
                  Official GTSRB dataset <ExternalLink size={15} />
                </a>
              </section>
              <div className="workflow">
                {[
                  "Raw images",
                  "Label assignment",
                  "RGB + resize",
                  "Group split",
                  "CNN training",
                  "Evaluation",
                ].map((s, i) => (
                  <div key={s}>
                    <span>{i + 1}</span>
                    {s}
                    {i < 5 && <ArrowRight size={15} />}
                  </div>
                ))}
              </div>
              {metrics?.dataset?.available ? (
                <section className="panel prose">
                  <h2>Configured dataset</h2>
                  <p>
                    {metrics.dataset.data.training_samples.toLocaleString()}{" "}
                    training samples ·{" "}
                    {metrics.dataset.data.validation_samples.toLocaleString()}{" "}
                    validation samples
                  </p>
                  <p>{metrics.dataset.data.split_method}</p>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                      data={metrics.dataset.data.class_counts.map(
                        (count: number, id: number) => ({ id, count }),
                      )}
                    >
                      <XAxis dataKey="id" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#52c7cb" />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="caption">
                    Class distribution from the actual configured data. Minority
                    classes can have weaker recall; compare their sample counts.
                  </p>
                </section>
              ) : (
                <section className="panel">
                  <Empty title="Dataset not configured">
                    Download GTSRB, arrange the images in numeric class folders,
                    then run training. Image counts and class distributions
                    appear only from generated metadata.
                  </Empty>
                </section>
              )}
            </>
          )}

        </main>
      </div>
    </div>
  );
}
export default App;
