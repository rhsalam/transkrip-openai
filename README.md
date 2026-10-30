# Transkrip (OpenAI)

Ubah rekaman audio/video jadi teks dengan API transkripsi OpenAI, lalu baca, sunting, dan ekspor
hasilnya dari browser.

## Fitur

- Unggah banyak file sekaligus (MP3, M4A, WAV, OPUS, MP4, MOV, …). Unggahan diantrekan dan
  transkrip berjalan paralel di server (dibatasi `MAX_PARALLEL`).
- Editor transkrip: pemutar audio yang menyorot kalimat yang sedang diputar, klik kalimat untuk
  lompat, sunting teks, ganti nama pembicara, cari di transkrip.
- Ekspor TXT, SRT, VTT, JSON.
- Bisa dibuka dari perangkat lain di jaringan dengan kata sandi.

## Jalankan

Butuh Python 3.12 dan `ffmpeg` (`brew install ffmpeg`).

```bash
cp .env.example .env   # isi OPENAI_API_KEY, atau tempel lewat halaman Pengaturan
./run.sh
```

Buka http://127.0.0.1:5100.

Dari terminal tanpa web UI:

```bash
.venv/bin/python transcriber.py rekaman.m4a -m whisper-1 -l id -f srt
```

## Pengaturan `.env`

| Variabel | Arti |
|---|---|
| `OPENAI_API_KEY` | API key OpenAI |
| `HOST` | `0.0.0.0` agar bisa dibuka dari perangkat lain; bawaan `127.0.0.1` |
| `APP_PASSWORD` | wajib untuk akses dari perangkat lain (Basic Auth, username bebas) |
| `MAX_PARALLEL` | jumlah file yang ditranskrip bersamaan, bawaan 3 |
| `PORT` | bawaan 5100 |

Akses dari komputer server sendiri tidak butuh kata sandi, dan API key hanya bisa diganti dari sana.

## Model

| Model | Kegunaan |
|---|---|
| `gpt-transcribe` | terbaru; "istilah khusus" (pisahkan dengan koma) dikirim sebagai `keywords` |
| `gpt-4o-transcribe` | default, akurat |
| `gpt-4o-mini-transcribe` | lebih murah |
| `gpt-4o-transcribe-diarize` | label pembicara (A, B, …) |
| `whisper-1` | timestamp per kalimat, terbaik untuk SRT/VTT |

## Cara kerja

File diubah jadi mp3 mono 16 kHz 64 kbps dengan ffmpeg, lalu dipotong per 10 menit agar lolos batas
25 MB dan batas durasi model. Ekor teks potongan sebelumnya dikirim sebagai `prompt` supaya
sambungannya nyambung. Model gpt-4o-* tidak memberi timestamp, jadi teksnya dipecah per 1–3 kalimat
dengan waktu perkiraan (ditandai `~` di editor). Untuk model diarize, label pembicara diberi awalan
nomor potongan (`2-A`) karena label tidak konsisten antarpotongan.

Data tersimpan di `data/`: `results/` (transkrip JSON) dan `media/` (audio untuk pemutar). File
unggahan asli dihapus setelah selesai.
