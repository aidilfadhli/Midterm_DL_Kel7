"""Ambil keluaran Task 3 yang tertanam (base64) di output sel terakhir notebook yang dijalankan di runtime Colab remote.

Pemakaian (dari folder kode/):
    python task3_clustering/extract_outputs.py path/ke/task3_clustering.ipynb            # tempatkan ke lokasi repo
    python task3_clustering/extract_outputs.py path/ke/notebook.ipynb --dest folder_uji   # ekstrak datar ke satu folder

Tanpa --dest: *.csv -> <root repo>/, results.json -> kode/results/task3/, figures/*.png -> laporan/figures/.
"""
import argparse, base64, hashlib, io, json, re, sys, zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("notebook"); ap.add_argument("--dest", default=None)
args = ap.parse_args()

nb = json.load(open(args.notebook, encoding="utf-8"))
text = ""
for c in nb["cells"]:
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            t = o.get("text", "")
            text += "".join(t) if isinstance(t, list) else t
m = list(re.finditer(r"BEGIN_TASK3_PAYLOAD sha256=([0-9a-f]{64}) bytes=(\d+)\n(.*?)\nEND_TASK3_PAYLOAD", text, re.S))
if not m:
    sys.exit("Payload tidak ditemukan di output notebook. Pastikan sel terakhir sudah dijalankan dan EMBED_OUTPUTS = True.")
sha, n, body = m[-1].group(1), int(m[-1].group(2)), m[-1].group(3)
raw = base64.b64decode("".join(body.split()))
if len(raw) != n or hashlib.sha256(raw).hexdigest() != sha:
    sys.exit(f"Payload rusak: ukuran {len(raw)} (harusnya {n}) atau sha256 tidak cocok.")

root = Path(__file__).resolve().parents[2]
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        if name.startswith("/") or ".." in Path(name).parts:
            sys.exit(f"Nama berkas tidak aman di dalam zip: {name}")
        if args.dest:
            out = Path(args.dest) / Path(name).name
        elif name.endswith(".csv"):
            out = root / name
        elif name == "results.json":
            out = root / "kode" / "results" / "task3" / name
        else:
            out = root / "laporan" / name
        out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(z.read(name)); print("ditulis:", out)
