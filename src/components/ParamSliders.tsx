import type { Params } from "../App";
import "./ParamSliders.css";

const SEP_MODELS = [
  { value: "UVR-MDX-NET-Inst_HQ_3.onnx", label: "MDX-Net（快速）" },
  { value: "model_bs_roformer_ep_368_sdr_12.9628.ckpt", label: "BS-Roformer（高质量）" },
];

const POSTPROCESS_OPTIONS = [
  { value: "", label: "无后处理" },
  { value: "denoise", label: "去噪" },
  { value: "deecho", label: "去混响" },
  { value: "both", label: "去噪 + 去混响" },
];

export interface SepSettings {
  sourceEnabled: boolean;
  targetEnabled: boolean;
  model: string;
  postprocess: string;
}

interface Props {
  params: Params;
  onChange: (p: Params) => void;
  sep: SepSettings;
  onSepChange: (s: SepSettings) => void;
  disabled?: boolean;
}

function Slider({
  label, hint, min, max, step, value,
  onChange,
}: {
  label: string; hint?: string;
  min: number; max: number; step: number; value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider-row">
      <div className="slider-meta">
        <span className="slider-label">{label}</span>
        <span className="slider-value">{value}</span>
      </div>
      {hint && <div className="slider-hint">{hint}</div>}
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function Toggle({
  label, checked, onChange, disabled,
}: {
  label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
}) {
  return (
    <label className={`toggle-label${disabled ? " toggle-disabled" : ""}`}>
      <div
        className={`toggle-track${checked ? " toggle-on" : ""}`}
        onClick={() => !disabled && onChange(!checked)}
      >
        <div className="toggle-thumb" />
      </div>
      <span>{label}</span>
    </label>
  );
}

export default function ParamSliders({ params, onChange, sep, onSepChange, disabled }: Props) {
  function set<K extends keyof Params>(key: K, val: Params[K]) {
    onChange({ ...params, [key]: val });
  }

  function setSep<K extends keyof SepSettings>(key: K, val: SepSettings[K]) {
    onSepChange({ ...sep, [key]: val });
  }

  const anySepOn = sep.sourceEnabled || sep.targetEnabled;

  return (
    <div className="params-panel">
      <div className="params-title">调音参数</div>
      <div className="params-grid">
        <Slider label="音调（半调）" hint="正值升调，负值降调"
          min={-24} max={24} step={1} value={params.pitch_shift}
          onChange={(v) => set("pitch_shift", v)} />

        <Slider label="速度" hint="< 1.0 加速，> 1.0 减速"
          min={0.5} max={2.0} step={0.1} value={params.length_adjust}
          onChange={(v) => set("length_adjust", v)} />
      </div>

      <div className="params-section-divider" />
      <div className="params-title">人声分离</div>

      <div className="sep-toggles">
        <Toggle label="原始音轨" checked={sep.sourceEnabled}
          onChange={(v) => setSep("sourceEnabled", v)} disabled={disabled} />
        <Toggle label="目标音色" checked={sep.targetEnabled}
          onChange={(v) => setSep("targetEnabled", v)} disabled={disabled} />
      </div>

      {anySepOn && (
        <div className="sep-config">
          <div className="sep-config-row">
            <span className="sep-config-label">模型</span>
            <select className="sep-select" value={sep.model}
              onChange={(e) => setSep("model", e.target.value)} disabled={disabled}>
              {SEP_MODELS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <div className="sep-config-row">
            <span className="sep-config-label">后处理</span>
            <select className="sep-select" value={sep.postprocess}
              onChange={(e) => setSep("postprocess", e.target.value)} disabled={disabled}>
              {POSTPROCESS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
