"""Inti transkripsi: potong audio dengan ffmpeg, kirim ke OpenAI, gabungkan hasil.

Bisa dipakai langsung dari terminal:
    python transcriber.py rekaman.m4a -m gpt-4o-transcribe -l id -f srt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

MODELS = {
    "gpt-transcribe": "GPT Transcribe — model terbaru, istilah khusus jadi keywords",
    "gpt-4o-transcribe": "GPT-4o Transcribe — paling akurat",
    "gpt-4o-mini-transcribe": "GPT-4o mini Transcribe — lebih murah",
    "gpt-4o-transcribe-diarize": "GPT-4o Transcribe Diarize — dengan label pembicara",
    "whisper-1": "Whisper — timestamp per kalimat (bagus untuk SRT)",
}
DEFAULT_MODEL = "gpt-4o-transcribe"

# Model gpt-4o-* membatasi durasi per permintaan (~23 menit) dan semua model
# membatasi ukuran file 25 MB. Potongan 10 menit mono 64 kbps ≈ 4.8 MB, aman.
CHUNK_SECONDS = 600
CONTEXT_CHARS = 300

ProgressFn = Callable[[float, str], None]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    approx: bool = False  # waktu diperkirakan dari panjang teks, bukan dari model


@dataclass
class Result:
    text: str
    model: str
    language: str | None
    duration: float
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        segs = [Segment(**s) for s in data.get("segments", [])]
        return cls(**{**data, "segments": segs})


class TranscribeError(RuntimeError):
    pass


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise TranscribeError(f"{name} tidak ditemukan. Pasang dengan: brew install ffmpeg")
    return path


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise TranscribeError(f"File tidak bisa dibaca sebagai audio/video: {path.name}")


def extract_audio(src: Path, dst: Path) -> None:
    """Ambil suaranya saja jadi mp3 mono 16 kHz 64 kbps (juga dipakai pemutar di editor)."""
    proc = subprocess.run(
        [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst)],
        capture_output=True, text=True,
    )
    if "does not contain any stream" in proc.stderr:
        raise TranscribeError(f"File ini tidak punya trek suara: {src.name}")
    if proc.returncode != 0 or not dst.exists():
        raise TranscribeError(f"ffmpeg gagal memproses file: {proc.stderr.strip()[-400:]}")


def split_audio(audio: Path, workdir: Path) -> list[Path]:
    """Potong mp3 hasil extract_audio per CHUNK_SECONDS tanpa encode ulang."""
    proc = subprocess.run(
        [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
         "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
         "-c", "copy", str(workdir / "chunk%03d.mp3")],
        capture_output=True, text=True,
    )
    chunks = sorted(workdir.glob("chunk*.mp3"))
    if proc.returncode != 0 or not chunks:
        raise TranscribeError(f"ffmpeg gagal memotong audio: {proc.stderr.strip()[-400:]}")
    return chunks


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def paragraphize(segments: list[Segment], max_chars: int = 260) -> list[Segment]:
    """Pecah segmen panjang tanpa timestamp (model gpt-4o-*) jadi potongan 1–3 kalimat.

    Waktunya dibagi sebanding panjang teks, jadi hanya perkiraan (approx=True).
    """
    out: list[Segment] = []
    for seg in segments:
        if seg.speaker or seg.approx or len(seg.text) <= max_chars:
            out.append(seg)
            continue
        parts: list[str] = []
        for sentence in SENTENCE_END.split(seg.text):
            if parts and len(parts[-1]) + len(sentence) < max_chars:
                parts[-1] = f"{parts[-1]} {sentence}"
            else:
                parts.append(sentence)
        total = sum(len(p) for p in parts) or 1
        t = seg.start
        for part in parts:
            span = (seg.end - seg.start) * len(part) / total
            out.append(Segment(t, t + span, part, approx=True))
            t += span
    return out


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _transcribe_chunk(client: OpenAI, chunk: Path, model: str, language: str | None,
                      prompt: str | None, keywords: list[str]) -> tuple[str, list[Segment]]:
    """Kembalikan (teks, segmen) dengan timestamp relatif terhadap potongan."""
    kwargs: dict = {"model": model}
    if language:
        kwargs["language"] = language

    if model == "whisper-1":
        kwargs.update(response_format="verbose_json", timestamp_granularities=["segment"])
    elif model.endswith("-diarize"):
        kwargs.update(response_format="diarized_json", chunking_strategy="auto")
        prompt = None  # model diarize tidak menerima prompt
    else:
        kwargs["response_format"] = "json"
    if model == "gpt-transcribe" and keywords:
        kwargs["keywords"] = keywords
    if prompt:
        kwargs["prompt"] = prompt

    with chunk.open("rb") as fh:
        resp = client.audio.transcriptions.create(file=fh, **kwargs)

    text = (_get(resp, "text") or "").strip()
    segments = [
        Segment(
            start=float(_get(s, "start", 0.0)),
            end=float(_get(s, "end", 0.0)),
            text=(_get(s, "text") or "").strip(),
            speaker=_get(s, "speaker"),
        )
        for s in (_get(resp, "segments") or [])
    ]
    return text, segments


def transcribe(path: str | Path, model: str = DEFAULT_MODEL, language: str | None = "id",
               prompt: str | None = None, api_key: str | None = None,
               progress: ProgressFn | None = None, audio_out: Path | None = None) -> Result:
    """audio_out: jika diisi, mp3 hasil ekstraksi disimpan di sana untuk diputar ulang."""
    src = Path(path)
    if model not in MODELS:
        raise TranscribeError(f"Model tidak dikenal: {model}")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise TranscribeError("OPENAI_API_KEY belum diatur.")
    report = progress or (lambda frac, msg: None)
    client = OpenAI(api_key=key)
    keywords = [k.strip() for k in (prompt or "").split(",") if k.strip()]

    duration = probe_duration(src)
    with tempfile.TemporaryDirectory(prefix="transkrip-") as tmp:
        report(0.02, "Menyiapkan audio…")
        audio = Path(tmp) / "audio.mp3"
        extract_audio(src, audio)
        chunks = split_audio(audio, Path(tmp))

        texts: list[str] = []
        segments: list[Segment] = []
        offset = 0.0
        for i, chunk in enumerate(chunks):
            report(0.05 + 0.9 * i / len(chunks), f"Mentranskrip bagian {i + 1} dari {len(chunks)}…")
            # Ekor teks sebelumnya dikirim sebagai prompt agar sambungan antarpotongan nyambung.
            context = " ".join(filter(None, [prompt, texts[-1][-CONTEXT_CHARS:] if texts else None]))
            text, segs = _transcribe_chunk(client, chunk, model, language, context or None, keywords)
            chunk_len = probe_duration(chunk)

            if not segs and text:
                segs = [Segment(0.0, chunk_len, text)]
            for s in segs:
                if s.speaker and len(chunks) > 1:
                    # Label pembicara tidak konsisten antarpotongan, jadi diberi awalan bagian.
                    s.speaker = f"{i + 1}-{s.speaker}"
                segments.append(Segment(s.start + offset, s.end + offset, s.text, s.speaker))
            texts.append(text)
            offset += chunk_len

        if audio_out:
            shutil.move(audio, audio_out)

    report(1.0, "Selesai")
    return Result(text="\n\n".join(t for t in texts if t), model=model,
                  language=language, duration=duration, segments=paragraphize(segments))


# ---------- format keluaran ----------

def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


AUTO_SPEAKER = re.compile(r"^(\d+-)?[A-Z]$")


def speaker_name(label: str) -> str:
    """Label bawaan model ("A", "2-B") jadi "Pembicara A"; nama yang sudah diganti dibiarkan."""
    return f"Pembicara {label}" if AUTO_SPEAKER.match(label) else label


def _line(seg: Segment) -> str:
    return f"[{speaker_name(seg.speaker)}] {seg.text}" if seg.speaker else seg.text


def segments_to_text(segments: list[Segment]) -> str:
    """Teks utuh dari segmen: paragraf baru tiap ganti pembicara atau ±600 karakter."""
    paras: list[str] = []
    prev_speaker, size = object(), 0
    for s in segments:
        if s.speaker != prev_speaker or size > 600 or not paras:
            paras.append(_line(s))
            size = len(s.text)
        else:
            paras[-1] += " " + s.text
            size += len(s.text)
        prev_speaker = s.speaker
    return "\n\n".join(paras)


def to_txt(r: Result) -> str:
    if any(s.speaker for s in r.segments):
        return "\n".join(f"{_stamp(s.start, '.')[:8]}  {_line(s)}" for s in r.segments) + "\n"
    return segments_to_text(r.segments) + "\n"


def to_srt(r: Result) -> str:
    blocks = [f"{i}\n{_stamp(s.start, ',')} --> {_stamp(s.end, ',')}\n{_line(s)}\n"
              for i, s in enumerate(r.segments, 1)]
    return "\n".join(blocks)


def to_vtt(r: Result) -> str:
    blocks = [f"{_stamp(s.start, '.')} --> {_stamp(s.end, '.')}\n{_line(s)}\n" for s in r.segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_json(r: Result) -> str:
    return json.dumps(r.to_dict(), ensure_ascii=False, indent=2)


FORMATS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def main() -> None:
    ap = argparse.ArgumentParser(description="Transkrip audio/video dengan OpenAI")
    ap.add_argument("file")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("-l", "--language", default="id", help="kode bahasa ISO-639-1, kosongkan = otomatis")
    ap.add_argument("-p", "--prompt", help="istilah/nama khusus agar ejaannya benar")
    ap.add_argument("-f", "--format", default="txt", choices=list(FORMATS))
    ap.add_argument("-o", "--output", help="file keluaran (default: di samping file sumber)")
    args = ap.parse_args()

    result = transcribe(args.file, args.model, args.language or None, args.prompt,
                        progress=lambda f, m: print(f"[{f:4.0%}] {m}", flush=True))
    out = Path(args.output) if args.output else Path(args.file).with_suffix(f".{args.format}")
    out.write_text(FORMATS[args.format](result), encoding="utf-8")
    print(f"Tersimpan: {out}")


if __name__ == "__main__":
    main()
