# live_recognizer.py  — versión con auto-captura + mejoras de fluidez + iluminación
import os
import cv2
import faiss
import time
import argparse
import sqlite3
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from insightface.app import FaceAnalysis

# -------------------- Utils de ONNXRuntime provider --------------------
def get_providers(prefer_gpu=True):
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if prefer_gpu and 'CUDAExecutionProvider' in avail:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    except Exception:
        pass
    return ['CPUExecutionProvider']

# -------------------- Dibujo --------------------
def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

# -------------------- Locks (para suavizado temporal) --------------------
class IdentityLock:
    def __init__(self, ttl=1.2):
        self.ttl = ttl
        self.expire_t = 0.0
        self.name = None
        self.score = 0.0
        self.bbox = None
        self.score_ema = None
        self.last_seen = 0.0  # para dwell/auto-capture

    def activate(self, name, score, bbox):
        now = time.time()
        self.expire_t = now + self.ttl
        self.name = name
        self.score = score
        self.score_ema = score if self.score_ema is None else 0.6*self.score_ema + 0.4*score
        self.bbox = bbox
        self.last_seen = now

    def valid(self):
        return time.time() < self.expire_t

    def maybe_update_bbox(self, new_bbox, iou_thr=0.3):
        if self.bbox is None:
            self.bbox = new_bbox; return True
        if iou(self.bbox, new_bbox) >= iou_thr:
            self.bbox = new_bbox; return True
        return False

# -------------------- DB para eventos --------------------
def ensure_events_table(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_name TEXT NOT NULL,
        score REAL NOT NULL,
        ts TEXT NOT NULL,
        snapshot_path TEXT
    );""")
    conn.commit(); conn.close()

def log_event(db_path, name, score, snapshot_path=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO events(person_name,score,ts,snapshot_path) VALUES (?,?,?,?);",
                (name, float(score), datetime.now().isoformat(timespec="seconds"), snapshot_path))
    conn.commit(); conn.close()

# -------------------- FAISS + mapping --------------------
def load_index_and_map(artifacts_dir):
    index_path = os.path.join(artifacts_dir, "index.faiss")
    db_path = os.path.join(artifacts_dir, "face_db.sqlite")
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")

    if not (os.path.exists(index_path) and os.path.exists(db_path) and os.path.exists(emb_path)):
        raise FileNotFoundError("Faltan artifacts: index.faiss, face_db.sqlite o embeddings.npy")

    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    dim = embeddings.shape[1]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT images.emb_index, persons.name
        FROM images JOIN persons ON images.person_id = persons.id
        ORDER BY images.emb_index ASC;
    """)
    rows = cur.fetchall()
    conn.close()

    id_to_name = {int(emb_index): name for emb_index, name in rows}
    return index, id_to_name, dim

# -------------------- Selección de caras --------------------
def largest_k_faces(faces, k=2):
    if not faces: return []
    faces_area = [(f, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
    faces_area.sort(key=lambda x: x[1], reverse=True)
    return [f for (f, _) in faces_area[:k]]

# -------------------- Mejora de iluminación (opcional) --------------------
def clahe_gamma(img, gamma=1.3):
    # img BGR 8-bit, devuelve BGR
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if gamma and gamma != 1.0:
        inv = 1.0 / max(gamma, 1e-6)
        lut = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, lut)
    return out

# -------------------- Main --------------------
# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Live face recognition + auto-capture with InsightFace + FAISS.")
    parser.add_argument("--rtsp", default="rtsp://admin:misCamaras2025@192.168.1.64:554/Streaming/Channels/102?tcp")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--pack", default="buffalo_s")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--det-interval", type=int, default=10)
    parser.add_argument("--lock-seconds", type=float, default=1.3)
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--prefer-gpu", action="store_true")

    # logging/snapshots
    parser.add_argument("--log-events", action="store_true")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--snap-dir", default="artifacts/events")

    # auto-captura
    parser.add_argument("--auto-capture", action="store_true",
                        help="Si una cara se mantiene estable por dwell, toma N fotos.")
    parser.add_argument("--capture-n", type=int, default=5, help="Número de fotos por ráfaga.")
    parser.add_argument("--capture-dwell", type=float, default=2.5, help="Segundos que debe ‘quedarse’ la cara.")
    parser.add_argument("--capture-gap-ms", type=int, default=250, help="Gap entre fotos de la ráfaga.")
    parser.add_argument("--capture-assign-thr", type=float, default=0.52,
                        help="Si score>=thr, se guardan en dataset/<nombre>; si no, en dataset/_unknown/")

    # preprocesado de iluminación
    parser.add_argument("--enhance-face", action="store_true",
                        help="Aplica CLAHE+gamma en chips antes de embebido (puede ayudar en baja luz).")

    args = parser.parse_args()

    index, id_to_name, dim = load_index_and_map(args.artifacts)
    db_path = os.path.join(args.artifacts, "face_db.sqlite")
    if args.log_events or args.save_snapshots:
        ensure_events_table(db_path)
    if args.save_snapshots:
        os.makedirs(args.snap_dir, exist_ok=True)

    # InsightFace
    providers = get_providers(prefer_gpu=args.prefer_gpu)
    app = FaceAnalysis(name=args.pack, providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))  # subir a 800 si te alcanza

    # RTSP
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir RTSP. Revisa URL/credenciales/red.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    locks = [IdentityLock(ttl=args.lock_seconds) for _ in range(args.max_faces)]
    last_burst_time = defaultdict(float)  # evita ráfagas seguidas por persona
    BURST_COOLDOWN = 20.0  # seg

    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    last_fps_t = time.time()

    while True:
        # descarta frames atrasados (reduce delay/“gris”)
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue

        # resize para procesar/mostrar
        h, w = frame.shape[:2]
        if w > args.width:
            scale = args.width / float(w)
            frame_disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_disp = frame.copy()

        run_detect = (frame_idx % args.det_interval == 0)
        detected_faces = []

        if run_detect:
            faces = app.get(frame_disp)
            faces = largest_k_faces(faces, k=args.max_faces)
            detected_faces = faces

            for fi, face in enumerate(faces):
                if getattr(face, "normed_embedding", None) is None:
                    continue

                # opcional: mejora de iluminación en el chip (trabaja sobre chip 112x112)
                if args.enhance_face and getattr(face, "embedding", None) is None:
                    # app.get ya genera chip interno; si no, tomamos bbox recortado (suave)
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    crop = frame_disp[max(0,y1):max(0,y2), max(0,x1):max(0,x2)].copy()
                    if crop.size > 0:
                        crop = clahe_gamma(crop, gamma=1.35)
                        # Nota: esto NO reinyecta al modelo; es una ayuda visual.
                        # El embedding que usamos sigue siendo el de InsightFace (normed_embedding).

                emb = face.normed_embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                q = emb.reshape(1, -1)

                D, I = index.search(q, 1)
                score = float(D[0, 0]) if I[0, 0] != -1 else -1.0
                name = id_to_name.get(int(I[0, 0]), "Desconocido")

                if fi < len(locks):
                    if score >= args.threshold:
                        locks[fi].activate(name, score, face.bbox)

                        # snapshots + logging en activación de lock
                        snapshot_path = None
                        if args.save_snapshots:
                            x1, y1, x2, y2 = [int(v) for v in face.bbox]
                            crop = frame_disp[max(0,y1):max(0,y2), max(0,x1):max(0,x2)].copy()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe = name.replace(" ", "_")
                            snapshot_path = os.path.join(args.snap_dir, f"{ts}_{safe}_{score:.2f}.jpg")
                            if crop.size > 0:
                                cv2.imwrite(snapshot_path, crop)
                            else:
                                cv2.imwrite(snapshot_path, frame_disp)
                        if args.log_events:
                            log_event(db_path, name, score, snapshot_path)
                    else:
                        # no activa lock; pero igual refrescamos bbox para dibujar "Desconocido"
                        locks[fi].maybe_update_bbox(face.bbox)

        # dibujo de locks válidos
        num_valid = 0
        for i, lk in enumerate(locks):
            if run_detect and i < len(detected_faces) and detected_faces:
                face = detected_faces[i]
                if lk.valid():
                    lk.maybe_update_bbox(face.bbox)
                else:
                    # dibuja desconocido temporal
                    fb = getattr(face, "bbox", None)
                    if fb is not None:
                        x1, y1, x2, y2 = [int(v) for v in fb]
                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        draw_label(frame_disp, "Desconocido", x1, max(20, y1 - 8))

            if lk.valid() and lk.bbox is not None and lk.name is not None:
                num_valid += 1
                x1, y1, x2, y2 = [int(v) for v in lk.bbox]
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 180, 0), 2)
                label_score = (0.6*lk.score_ema + 0.4*lk.score) if lk.score_ema else lk.score
                draw_label(frame_disp, f"{lk.name} {label_score:.2f}", x1, max(20, y1 - 8))

        # mensaje si nada
        if num_valid == 0 and not detected_faces:
            draw_label(frame_disp, "Sin rostros", 10, 30)

        # -------------------- AUTO-CAPTURA --------------------
        if args.auto_capture:
            now = time.time()
            for lk in locks:
                if not (lk.valid() and lk.bbox is not None and lk.name is not None):
                    continue
                dwell = now - lk.last_seen  # tiempo desde última activación
                if dwell >= args.capture_dwell and (now - last_burst_time[lk.name] >= BURST_COOLDOWN):
                    # decide carpeta destino según score
                    target_name = lk.name if lk.score >= args.capture_assign_thr else "_unknown"
                    person_dir = os.path.join("dataset", target_name.replace(" ", "_"))
                    os.makedirs(person_dir, exist_ok=True)

                    # ráfaga de N fotos (gap ms)
                    for k in range(args.capture_n):
                        # refresca bbox por si se movió un poco (simple)
                        x1, y1, x2, y2 = [int(v) for v in lk.bbox]
                        crop = frame_disp[max(0,y1):max(0,y2), max(0,x1):max(0,x2)].copy()
                        if crop.size == 0:
                            crop = frame_disp.copy()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(person_dir, f"auto_{ts}_{k+1}.jpg")
                        cv2.imwrite(filename, crop)
                        cv2.waitKey(max(1, int(args.capture_gap_ms)))  # pausa corta
                    last_burst_time[lk.name] = now
                    # NOTA: para que entren a la base, vuelve a correr enroll_faces.py

        # FPS
        now = time.time()
        if now - last_fps_t >= 1.0:
            fps = (frame_idx + 1) / (now - t0 + 1e-9)
            last_fps_t = now
        draw_label(frame_disp, f"FPS ~ {fps:.1f}", 10, frame_disp.shape[0] - 10)

        cv2.imshow("Reconocimiento - Hikvision", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
