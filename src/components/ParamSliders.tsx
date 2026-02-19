import type { Params } from "../App";
import "./ParamSliders.css";

interface Props {
  params: Params;
  onChange: (p: Params) => void;
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

export default function ParamSliders({ params, onChange }: Props) {
  function set<K extends keyof Params>(key: K, val: Params[K]) {
    onChange({ ...params, [key]: val });
  }

  return (
    <div className="params-panel">
      <div className="params-title">转换参数</div>
      <div className="params-grid">
        <Slider label="扩散步数" hint="步数越多质量越高，速度越慢（推荐 30~50）"
          min={1} max={100} step={1} value={params.diffusion_steps}
          onChange={(v) => set("diffusion_steps", v)} />

        <Slider label="音调偏移（半音）" hint="正值升调，负值降调"
          min={-24} max={24} step={1} value={params.pitch_shift}
          onChange={(v) => set("pitch_shift", v)} />

        <Slider label="时长调整" hint="< 1.0 加速，> 1.0 减速"
          min={0.5} max={2.0} step={0.1} value={params.length_adjust}
          onChange={(v) => set("length_adjust", v)} />

        <Slider label="CFG Rate"
          min={0.0} max={1.0} step={0.1} value={params.inference_cfg_rate}
          onChange={(v) => set("inference_cfg_rate", v)} />

        <div className="slider-row checkbox-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={params.use_fp16}
              onChange={(e) => set("use_fp16", e.target.checked)}
            />
            <span>启用 FP16</span>
          </label>
          <div className="slider-hint">速度更快但可能影响稳定性，MPS 设备建议关闭</div>
        </div>

        <div className="slider-row checkbox-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={params.auto_f0_adjust}
              onChange={(e) => set("auto_f0_adjust", e.target.checked)}
            />
            <span>自动 F0 调整</span>
          </label>
          <div className="slider-hint">自动将源音高对齐参考音高（歌声转换通常建议关闭）</div>
        </div>
      </div>
    </div>
  );
}
