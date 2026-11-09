"""Web UI transkrip: unggah file, pantau progres, sunting transkrip, unduh TXT/SRT/VTT/JSON."""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from dotenv import set_key
from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import openai

import transcriber as tr

BASE = Path(__file__).parent
ENV_FILE = BASE / ".env"
DATA = Path(os.environ.get("TRANSKRIP_DATA_DIR", BASE / "data"))
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
MEDIA = DATA / "media"
for d in (UPLOADS, RESULTS, MEDIA):
    d.mkdir(parents=True, exist_ok=True)
# Antrean hanya di memori, jadi sisa unggahan dari proses sebelumnya tidak bisa diulang lagi.
for leftover in UPLOADS.iterdir():
    leftover.unlink(missing_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3  # 2 GB; ffmpeg yang mengecilkannya
app.config["TEMPLATES_AUTO_RELOAD"] = True

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
save_lock = threading.Lock()
# Berapa file yang ditranskrip bersamaan; sisanya antre agar tidak kena rate limit OpenAI.
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", 3))
slots = threading.BoundedSemaphore(MAX_PARALLEL)
# File unggahan dan opsi dari transkrip yang gagal, agar bisa diulang tanpa unggah lagi.
failed: dict[str, tuple[Path, str | None, str | None]] = {}

LOOPBACK = {"127.0.0.1", "::1"}


def _is_local() -> bool:
    return request.remote_addr in LOOPBACK


@app.before_request
def require_password():
    """Akses dari komputer lain wajib kata sandi APP_PASSWORD (HTTP Basic Auth)."""
    if _is_local():
        return None
    password = os.environ.get("APP_PASSWORD", "")
    if not password:
        return Response("Akses dari jaringan dimatikan: APP_PASSWORD belum diatur di .env.", 403)
    auth = request.authorization
    if auth and hmac.compare_digest((auth.password or "").encode(), password.encode()):
        return None
    return Response("Butuh kata sandi.", 401, {"WWW-Authenticate": 'Basic realm="Transkrip"'})


def _update(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _meta_path(job_id: str) -> Path:
    return RESULTS / f"{job_id}.json"


def _media_path(job_id: str) -> Path:
    return MEDIA / f"{job_id}.mp3"


def _save(rec: dict) -> None:
    _meta_path(rec["id"]).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def _error_message(e: Exception) -> str:
    """Terjemahkan error OpenAI yang umum jadi pesan yang bisa ditindaklanjuti."""
    if isinstance(e, openai.RateLimitError):
        if getattr(e, "code", None) in ("insufficient_quota", "credit_balance_exhausted") \
                or "credit" in str(e) or "quota" in str(e):
            return "Saldo API OpenAI habis. Isi saldo di platform.openai.com, lalu klik Ulangi."
        return "Terlalu banyak permintaan ke OpenAI. Tunggu sebentar, lalu klik Ulangi."
    if isinstance(e, openai.AuthenticationError):
        return "API key OpenAI ditolak. Periksa di halaman Pengaturan, lalu klik Ulangi."
    if isinstance(e, openai.APIConnectionError):
        return "Tidak bisa terhubung ke OpenAI. Periksa internet, lalu klik Ulangi."
    if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
        return "Server OpenAI sedang bermasalah. Coba lagi beberapa menit lagi."
    return str(e)


def _run(job_id: str, src: Path, model: str, language: str | None, prompt: str | None) -> None:
    try:
        with slots:
            _update(job_id, message="Menyiapkan audio…")
            _transcribe_job(job_id, src, model, language, prompt)
    except Exception as e:  # noqa: BLE001 — pesan apa pun ditampilkan ke pengguna
        failed[job_id] = (src, language, prompt)
        _update(job_id, status="error", message=_error_message(e), retryable=True)
    else:
        src.unlink(missing_ok=True)


def _start(job_id: str, src: Path, model: str, language: str | None, prompt: str | None) -> None:
    threading.Thread(target=_run, daemon=True, args=(job_id, src, model, language, prompt)).start()


def _transcribe_job(job_id: str, src: Path, model: str, language: str | None, prompt: str | None) -> None:
    result = tr.transcribe(src, model, language, prompt, audio_out=_media_path(job_id),
                           progress=lambda f, m: _update(job_id, progress=f, message=m))
    with jobs_lock:
        job = jobs[job_id]
        record = {k: job[k] for k in ("id", "filename", "created")} | {"result": result.to_dict()}
    _save(record)
    _update(job_id, status="done", progress=1.0, message="Selesai")


def _valid_id(job_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)


def _load(job_id: str) -> dict:
    _valid_id(job_id)
    path = _meta_path(job_id)
    if not path.exists():
        abort(404)
    rec = json.loads(path.read_text(encoding="utf-8"))
    # Transkrip lama tersimpan sebagai blok 10 menit; pecah jadi paragraf sekali saja.
    result = tr.Result.from_dict(rec["result"])
    segments = tr.paragraphize(result.segments)
    if len(segments) != len(result.segments):
        result.segments = segments
        rec["result"] = result.to_dict()
        with save_lock:
            _save(rec)
    return rec


def _summary(rec: dict) -> dict:
    r = rec["result"]
    return {"id": rec["id"], "filename": rec["filename"], "created": rec["created"],
            "status": "done", "model": r["model"], "duration": r["duration"],
            "preview": r["text"][:160], "has_media": _media_path(rec["id"]).exists()}


@app.get("/")
def index():
    return render_template("index.html", models=tr.MODELS, default_model=tr.DEFAULT_MODEL,
                           is_local=_is_local(), has_key=bool(os.environ.get("OPENAI_API_KEY")),
                           host=os.environ.get("HOST", "127.0.0.1"))


@app.post("/api/key")
def save_key():
    if not _is_local():
        return jsonify(error="API key hanya bisa diganti dari komputer server."), 403
    key = (request.json or {}).get("key", "").strip()
    if not key.startswith("sk-"):
        return jsonify(error="API key OpenAI biasanya diawali 'sk-'."), 400
    ENV_FILE.touch(mode=0o600, exist_ok=True)
    set_key(str(ENV_FILE), "OPENAI_API_KEY", key)
    os.environ["OPENAI_API_KEY"] = key
    return jsonify(ok=True)


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify(error="Pilih file audio atau video dulu."), 400
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify(error="API key OpenAI belum diatur. Buka Pengaturan."), 400
    model = request.form.get("model", tr.DEFAULT_MODEL)
    if model not in tr.MODELS:
        return jsonify(error="Model tidak dikenal."), 400

    job_id = uuid.uuid4().hex
    name = secure_filename(upload.filename) or "audio"
    src = UPLOADS / f"{job_id}_{name}"
    upload.save(src)

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "filename": upload.filename, "created": time.time(),
                        "status": "running", "progress": 0.0, "message": "Menunggu antrean…",
                        "model": model}
    _start(job_id, src, model, request.form.get("language") or None,
           request.form.get("prompt", "").strip() or None)
    return jsonify(id=job_id)


@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id: str):
    """Ulangi transkrip yang gagal dengan file unggahan yang sama; model boleh diganti."""
    _valid_id(job_id)
    model = (request.json or {}).get("model") if request.is_json else None
    if model and model not in tr.MODELS:
        return jsonify(error="Model tidak dikenal."), 400
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job["status"] != "error" or job_id not in failed:
            return jsonify(error="File ini tidak bisa diulang; unggah ulang filenya."), 409
        src, language, prompt = failed.pop(job_id)
        if not src.exists():
            job["retryable"] = False
            return jsonify(error="File unggahan sudah tidak ada; unggah ulang filenya."), 410
        job.update(status="running", progress=0.0, message="Menunggu antrean…", retryable=False,
                   model=model or job["model"])
        model = job["model"]
    _start(job_id, src, model, language, prompt)
    return jsonify(id=job_id)


@app.get("/api/files")
def list_files():
    with jobs_lock:
        pending = [dict(j) for j in jobs.values() if j["status"] != "done"]
    done = []
    for path in RESULTS.glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        done.append(_summary(rec))
    items = pending + done
    items.sort(key=lambda it: it["created"], reverse=True)
    return jsonify(items)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = dict(jobs.get(job_id) or {})
    if job.get("status") in ("running", "error"):
        return jsonify(job)
    rec = _load(job_id)
    return jsonify(rec | {"status": "done", "has_media": _media_path(job_id).exists()})


@app.put("/api/jobs/<job_id>")
def update_job(job_id: str):
    """Simpan suntingan: judul, teks per segmen, dan nama pembicara."""
    body = request.json or {}
    with save_lock:
        rec = _load(job_id)
        if title := (body.get("filename") or "").strip():
            rec["filename"] = title[:200]
        if "segments" in body:
            result = tr.Result.from_dict(rec["result"])
            edits = body["segments"]
            if not isinstance(edits, list) or len(edits) != len(result.segments):
                return jsonify(error="Jumlah segmen tidak cocok; muat ulang halaman."), 409
            for seg, edit in zip(result.segments, edits):
                seg.text = str(edit.get("text", seg.text)).strip()
                seg.speaker = (str(edit["speaker"]).strip() or None) if edit.get("speaker") else seg.speaker
            result.segments = [s for s in result.segments if s.text]
            result.text = tr.segments_to_text(result.segments)
            rec["result"] = result.to_dict()
        rec["edited"] = time.time()
        _save(rec)
    return jsonify(rec | {"status": "done", "has_media": _media_path(job_id).exists()})


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    _valid_id(job_id)
    with jobs_lock:
        if jobs.get(job_id, {}).get("status") == "running":
            return jsonify(error="Masih diproses."), 409
        jobs.pop(job_id, None)
    if job_id in failed:
        failed.pop(job_id)[0].unlink(missing_ok=True)
    _meta_path(job_id).unlink(missing_ok=True)
    _media_path(job_id).unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>/media")
def media(job_id: str):
    _valid_id(job_id)
    path = _media_path(job_id)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/mpeg", conditional=True)


@app.get("/api/jobs/<job_id>/download/<fmt>")
def download(job_id: str, fmt: str):
    if fmt not in tr.FORMATS:
        abort(404)
    rec = _load(job_id)
    body = tr.FORMATS[fmt](tr.Result.from_dict(rec["result"]))
    stem = Path(rec["filename"]).stem or "transkrip"
    mime = {"json": "application/json", "vtt": "text/vtt"}.get(fmt, "text/plain")
    return Response(body, mimetype=f"{mime}; charset=utf-8", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{secure_filename(stem) or 'transkrip'}.{fmt}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    # HOST=0.0.0.0 di .env agar bisa dibuka dari perangkat lain di jaringan.
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Transkrip berjalan di http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
