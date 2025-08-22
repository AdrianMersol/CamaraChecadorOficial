# live_recognizer.py
import os
import cv2
import faiss
import time
import argparse
import sqlite3
import numpy as np
from collections import deque
from insightface.app import FaceAnalysis

def get_providers(prefer_gpu=True):
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if prefer_gpu and 'CUDAExecutionProvider' in avail:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    except Exception:
        pass
    return ['CPUExecutionProvider']

def draw_label(img, text, x, y):
    # caja con fondo para legibilidad
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def iou(a, b):
    # a, b: [x1,y1,x2,y2]
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union

class IdentityLock:
    def __init__(self, ttl=1.2):
        self.ttl = ttl
        self.expire_t = 0.0
        self.name = None
        self.score = 0.0
        self.bbox = None
        self.score_ema = None

    def activate(self, name, score, bbox):
        now = time.time()
        self.expire_t = now + self.ttl
        self.name = name
        self.score = score
        self.score_ema = score if self.score_ema is None else 0.6*self.score_ema + 0.4*score
        self.bbox = bbox

    def valid(self):
        return time.time() < self.expire_t

    def maybe_update_bbox(self, new_bbox, iou_thr=0.3):
        if self.bbox is None:
            self.bbox = new_bbox
            return True
        if iou(self.bbox, new_bbox) >= iou_thr:
            self.bbox = new_bbox
            return True
        return False

def load_index_and_map(artifacts_dir):
    index_path = os.path.join(artifacts_dir, "index.faiss")
    db_path = os.path.join(artifacts_dir, "face_db.sqlite")
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")

    if not (os.path.exists(index_path) and os.path.exists(db_path) and os.path.exists(emb_path)):
        raise FileNotFoundError("Faltan archivos en artifacts/: index.faiss, face_db.sqlite o embeddings.npy")

    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    dim = embeddings.shape[1]

    # Cargar mapping emb_index -> nombre
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT images.emb_index, persons.name
        FROM images JOIN persons ON images.person_id = persons.id
        ORDER BY images.emb_index ASC;
    """)
    rows = cur.fetchall()
    conn.close()

    # Mapeo a lista para indexado rápido
    id_to_name = {}
    for emb_index, name in rows:
        id_to_name[int(emb_index)] = name

    return index, id_to_name, dim

def largest_k_faces(faces, k=2):
    if not faces:
        return []
    faces_area = [(f, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
    faces_area.sort(key=lambda x: x[1], reverse=True)
    return [f for (f, _) in faces_area[:k]]

def main():
    parser = argparse.ArgumentParser(description="Live face recognition over RTSP using InsightFace + FAISS.")
    parser.add_argument("--rtsp", default="rtsp://admin:misCamaras2025@192.168.1.64:554/Streaming/Channels/102",
                        help="URL RTSP de la cámara Hikvision.")
    parser.add_argument("--artifacts", default="artifacts", help="Carpeta con index.faiss, face_db.sqlite, embeddings.npy.")
    parser.add_argument("--pack", default="buffalo_s", help="Paquete InsightFace (buffalo_s|buffalo_l).")
    parser.add_argument("--width", type=int, default=800, help="Ancho de procesamiento (redimensionado).")
    parser.add_argument("--det-interval", type=int, default=8, help="Detectar/Reconocer cada N frames.")
    parser.add_argument("--lock-seconds", type=float, default=1.2, help="Duración del identity lock.")
    parser.add_argument("--max-faces", type=int, default=2, help="Máximo de rostros por frame a procesar.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Umbral coseno para aceptación.")
    parser.add_argument("--prefer-gpu", action="store_true", help="Usar GPU si está disponible.")
    args = parser.parse_args()

    index, id_to_name, dim = load_index_and_map(args.artifacts)

    providers = get_providers(prefer_gpu=args.prefer_gpu)
    app = FaceAnalysis(name=args.pack, providers=providers)
    # det_size pequeño para rapidez, 640 es buen arranque
    app.prepare(ctx_id=0, det_size=(640, 640))

    # RTSP
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Intento sin flag explícito
        cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el stream RTSP. Revisa URL/credenciales/red.")

    # Reduce latencia si es soportado
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Locks para hasta 2 rostros
    locks = [IdentityLock(ttl=args.lock_seconds) for _ in range(args.max_faces)]

    frame_idx = 0
    t0 = time.time()
    fps = 0.0
    last_fps_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame no válido, reintentando...")
            time.sleep(0.05)
            continue

        # Redimensionado para procesar/mostrar
        h, w = frame.shape[:2]
        scale = 1.0
        if w > args.width:
            scale = args.width / float(w)
            frame_disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_disp = frame.copy()

        # ¿Toca detectar?
        run_detect = (frame_idx % args.det_interval == 0)
        detected_faces = []

        if run_detect:
            faces = app.get(frame_disp)  # BGR
            faces = largest_k_faces(faces, k=args.max_faces)
            detected_faces = faces

            # Para cada cara detectada, buscamos en FAISS
            for fi, face in enumerate(faces):
                if getattr(face, "normed_embedding", None) is None:
                    continue
                emb = face.normed_embedding.astype(np.float32)
                # Normalización defensiva
                norm = np.linalg.norm(emb) + 1e-12
                emb = emb / norm
                q = emb.reshape(1, -1)

                # Búsqueda top-1
                D, I = index.search(q, 1)
                score = float(D[0, 0]) if I[0, 0] != -1 else -1.0
                name = id_to_name.get(int(I[0, 0]), "Desconocido")

                # Actualiza/activa lock correspondiente (por posición fi)
                if fi < len(locks):
                    if score >= args.threshold:
                        # Actualiza lock si bbox coincide o simplemente forza (tenemos máximo 2)
                        locks[fi].activate(name, score, face.bbox)
                    else:
                        # Si no alcanza umbral, pero hay lock válido, no lo tumbamos;
                        # solo no lo reactivamos. Si ya expiró, quedará libre.
                        pass

        # Dibujo: usa locks válidos (y actualiza bbox si hay detecciones)
        num_valid = 0
        for i, lk in enumerate(locks):
            if run_detect and i < len(detected_faces) and detected_faces:
                # Si detectamos y hay cara en esa posición, intenta refrescar bbox aunque sin activar
                # (la activación ya se hizo arriba si superó umbral).
                face = detected_faces[i]
                if lk.valid():
                    if lk.maybe_update_bbox(face.bbox):
                        pass
                else:
                    # Si no hay lock válido, pero detectamos, dibujamos como "Desconocido" temporal
                    if getattr(face, "bbox", None) is not None and getattr(face, "det_score", None) is not None:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        draw_label(frame_disp, "Desconocido", x1, max(20, y1 - 8))

            if lk.valid() and lk.bbox is not None and lk.name is not None:
                num_valid += 1
                x1, y1, x2, y2 = [int(v) for v in lk.bbox]
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 180, 0), 2)
                label = f"{lk.name} { (0.6*lk.score_ema + 0.4*lk.score) if lk.score_ema else lk.score:.2f}"
                draw_label(frame_disp, label, x1, max(20, y1 - 8))

        # Si no hay locks válidos y no se dibujó nada…
        if num_valid == 0 and not detected_faces:
            draw_label(frame_disp, "Sin rostros", 10, 30)

        # FPS
        now = time.time()
        if now - last_fps_t >= 1.0:
            fps = (frame_idx + 1) / (now - t0 + 1e-9)
            last_fps_t = now
        draw_label(frame_disp, f"FPS ~ {fps:.1f}", 10, frame_disp.shape[0] - 10)

        cv2.imshow("Reconocimiento - Hikvision", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
