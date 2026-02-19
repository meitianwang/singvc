import AudioUpload from "./AudioUpload";
import "./AudioUploadWithSep.css";

const SEP_MODELS = [
  { value: "UVR-MDX-NET-Inst_HQ_3.onnx", label: "MDX-Net（快速）" },
  { value: "model_bs_roformer_ep_368_sdr_12.9628.ckpt", label: "BS-Roformer（高质量）" },
];

const POSTPROCESS_OPTIONS = [
  { value: "", label: "无后处理" },
  { value: "denoise", label: "去噪" },
  { value: "deecho", label: "去混响" },
  { value: "both", label: "去噪 + 去混响（最干净）" },
];

interface Props {
  label: string;
  file: File | null;
  onFile: (f: File) => void;
  sepEnabled: boolean;
  onSepEnabled: (v: boolean) => void;
  sepModel: string;
  onSepModel: (v: string) => void;
  postprocess: string;
  onPostprocess: (v: string) => void;
  disabled?: boolean;
}

export default function AudioUploadWithSep({
  label, file, onFile,
  sepEnabled, onSepEnabled,
  sepModel, onSepModel,
  postprocess, onPostprocess,
  disabled = false,
}: Props) {
  return (
    <div className="upload-with-sep">
      <AudioUpload label={label} file={file} onFile={onFile} />
      <div className={`sep-row${disabled ? " sep-row-disabled" : ""}`}>
        <label className="sep-check">
          <input
            type="checkbox"
            checked={sepEnabled}
            onChange={(e) => onSepEnabled(e.target.checked)}
            disabled={disabled}
          />
          分离人声后使用
        </label>
      </div>
      {sepEnabled && (
        <div className={`sep-selects${disabled ? " sep-row-disabled" : ""}`}>
          <select
            className="sep-model-mini"
            value={sepModel}
            onChange={(e) => onSepModel(e.target.value)}
            disabled={disabled}
          >
            {SEP_MODELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
          <select
            className="sep-model-mini"
            value={postprocess}
            onChange={(e) => onPostprocess(e.target.value)}
            disabled={disabled}
          >
            {POSTPROCESS_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
