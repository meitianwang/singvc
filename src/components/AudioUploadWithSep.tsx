import AudioUpload from "./AudioUpload";

interface Props {
  label: string;
  file: File | null;
  onFile: (f: File) => void;
}

export default function AudioUploadWithSep({ label, file, onFile }: Props) {
  return <AudioUpload label={label} file={file} onFile={onFile} />;
}
