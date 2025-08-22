# enroll_faces.py
import os
import sys
import cv2
import faiss
import sqlite3
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

# InsightFace
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

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        emb_index INTEGER UNIQUE NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons(id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_emb_index ON images(emb_index);")
    conn.commit()
    return conn

def upsert_person(conn, name):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO persons(name) VALUES(?);", (name,))
    conn.commit()
    cur.execute("SELECT id FROM persons WHERE name=?;", (name,))
    return cur.fetchone()[0]

def insert_image(conn, person_id, file_path, emb_index):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO images(person_id, file_path, emb_index)
        VALUES (?,?,?);
    """, (person_id, file_path, emb_index))
    conn.commit()

def largest_face(faces):
    if not faces:
        return None
    # faces[i].bbox -> [x1,y1,x2,y2]
    areas = [(f, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
    areas.sort(key=lambda x: x[1], reverse=True)
    return areas[0][0]

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)  # fallback
    return img

def main():
    parser = argparse.ArgumentParser(description="Enroll faces into FAISS + SQLite using InsightFace.")
    parser.add_argument("--dataset", default="dataset", help="Carpeta raíz del dataset.")
    parser.add_argument("--artifacts", default="artifacts", help="Carpeta de salida para DB/FAISS/Numpy.")
    parser.add_argument("--pack", default="buffalo_s", help="Paquete InsightFace (buffalo_s|buffalo_l).")
    parser.add_argument("--det-size", default=640, type=int, help="Tamaño del detector SCRFD (cuadrado).")
    parser.add_argument("--prefer-gpu", action="store_true", help="Usar GPU si está disponible.")
    args = parser.parse_args()

    ensure_dirs(args.artifacts)
    db_path = os.path.join(args.artifacts, "face_db.sqlite")
    index_path = os.path.join(args.artifacts, "index.faiss")
    emb_path = os.path.join(args.artifacts, "embeddings.npy")

    # DB
    conn = create_db(db_path)

    # InsightFace (detección + reconocimiento)
    providers = get_providers(prefer_gpu=args.prefer_gpu)
    app = FaceAnalysis(name=args.pack, providers=providers)
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # Recorrido del dataset
    all_embeddings = []
    total_images = 0
    enrolled_images = 0
    persons_count = 0

    # Determinar dimensión del embedding (tras primera cara encontrada)
    emb_dim = None
    emb_index_counter = 0  # mapea fila del índice -> persona/imagen

    people_dirs = [d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]
    people_dirs.sort()

    for person_name in tqdm(people_dirs, desc="Personas"):
        person_dir = os.path.join(args.dataset, person_name)
        person_id = upsert_person(conn, person_name)
        persons_count += 1

        image_files = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            image_files.extend([os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith(ext)])
        image_files.sort()

        for img_path in tqdm(image_files, leave=False, desc=person_name):
            total_images += 1
            img = load_image(img_path)
            if img is None:
                print(f"[WARN] No se pudo leer: {img_path}")
                continue

            # Detecta y reconoce
            faces = app.get(img)
            face = largest_face(faces)
            if face is None or getattr(face, "normed_embedding", None) is None:
                # puede no haber cara o no haber embedding (muy raro con app preparado)
                continue

            emb = face.normed_embedding.astype(np.float32)
            # Normalización defensiva (debería venir normalizado)
            norm = np.linalg.norm(emb) + 1e-12
            emb = emb / norm

            if emb_dim is None:
                emb_dim = emb.shape[0]

            all_embeddings.append(emb)
            insert_image(conn, person_id, img_path, emb_index_counter)
            emb_index_counter += 1
            enrolled_images += 1

    if enrolled_images == 0:
        print("[ERROR] No se enroló ninguna imagen. Verifica tu dataset.")
        sys.exit(1)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(emb_path, embeddings)

    # FAISS (coseno vía IP con vectores normalizados)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    print("====================================")
    print(f"Personas:         {persons_count}")
    print(f"Imágenes totales: {total_images}")
    print(f"Embeddings OK:    {enrolled_images}")
    print(f"Dim embedding:    {embeddings.shape[1]}")
    print(f"DB:               {db_path}")
    print(f"FAISS:            {index_path}")
    print(f"Embeddings.npy:   {emb_path}")
    print("¡Enrolamiento completado!")

if __name__ == "__main__":
    main()
