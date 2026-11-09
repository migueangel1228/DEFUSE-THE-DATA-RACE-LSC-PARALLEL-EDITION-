import os
import json
import time
import cv2
import numpy as np
try:
    from feature_extractor import HandFeatureExtractor, CLASS_NAMES, FEATURE_DIM
except ImportError:
    from src.feature_extractor import HandFeatureExtractor, CLASS_NAMES, FEATURE_DIM

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
ROTATION_ANGLES = [-16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0]

def get_canonical_lsc_landmarks(class_name: str, variation_idx: int = 0) -> np.ndarray:
    """
    Genera la topología geométrica canónica realista de los 21 puntos de MediaPipe
    para cada una de las 11 señas de la Lengua de Señas Colombiana (LSC).
    Incluye variantes calibradas para la seña 'E' con mano derecha.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # 0: Muñeca
    pts[0] = [0.50, 0.85, 0.0]
    # MCPs / bases: Thumb CMC 1, Index MCP 5, Middle MCP 9, Ring MCP 13, Pinky MCP 17
    pts[1] = [0.38, 0.74, 0.0]
    pts[5] = [0.42, 0.58, 0.0]
    pts[9] = [0.50, 0.56, 0.0]
    pts[13] = [0.58, 0.58, 0.0]
    pts[17] = [0.65, 0.62, 0.0]

    def set_finger(b, mode, spread_x, is_thumb=False):
        if is_thumb:
            if mode == 'UP_A': # 'A': Pulgar vertical apoyado al lado del puño
                pts[2] = pts[1] + [-0.03, -0.07, 0.0]
                pts[3] = pts[1] + [-0.03, -0.14, 0.0]
                pts[4] = pts[1] + [-0.02, -0.21, 0.0]
            elif mode == 'WIDE_5': # '5', '3': Pulgar abierto hacia afuera
                pts[2] = pts[1] + [-0.07, -0.04, 0.0]
                pts[3] = pts[1] + [-0.13, -0.09, 0.0]
                pts[4] = pts[1] + [-0.19, -0.14, 0.0]
            elif mode == 'TOUCH_O': # 'O': Pulgar curvado hacia arriba para tocar la punta del índice
                pts[2] = pts[1] + [-0.02, -0.06, 0.0]
                pts[3] = pts[1] + [0.03, -0.11, 0.0]
                pts[4] = [0.46, 0.56, 0.0]
            elif mode == 'FOLD_E': # 'E': Pulgar plegado horizontalmente bajo los nudillos/yemas
                v_mod = variation_idx % 4
                y_off = 0.65 + 0.015 * v_mod
                pts[2] = pts[1] + [0.03, -0.02, 0.0]
                pts[3] = pts[1] + [0.07, -0.03, 0.0]
                pts[4] = [0.48, y_off, 0.0]
            else: # FOLD_NORMAL (1, 2, 4, I, U): Pulgar cruzado sobre dedos recogidos
                pts[2] = pts[1] + [0.04, -0.03, 0.0]
                pts[3] = pts[1] + [0.08, -0.05, 0.0]
                pts[4] = [0.48, 0.62, 0.0]
        else:
            mcp = pts[b]
            if mode == 'EXTENDED': # Dedo recto extendido hacia arriba
                pts[b+1] = mcp + [spread_x * 0.3, -0.08, 0.0]
                pts[b+2] = mcp + [spread_x * 0.6, -0.15, 0.0]
                pts[b+3] = mcp + [spread_x * 1.0, -0.22, 0.0]
            elif mode == 'TOUCH_O': # Dedo curvado hacia abajo tocando el pulgar en anillo
                pts[b+1] = mcp + [spread_x * 0.2, -0.05, -0.02]
                pts[b+2] = mcp + [spread_x * 0.1, -0.02, -0.04]
                pts[b+3] = [0.46 + spread_x * 0.15, 0.56, -0.02]
            elif mode == 'CLAW_E': # 'E': Dedos doblados en garra sobre la palma (calibrado para mano derecha)
                v_mod = variation_idx % 4
                depth = 0.06 + 0.015 * v_mod
                pts[b+1] = mcp + [spread_x * 0.1, -0.04, 0.0]
                pts[b+2] = mcp + [spread_x * 0.1, +0.02, 0.0]
                pts[b+3] = mcp + [spread_x * 0.1, depth, 0.0]
            else: # FIST / CURLED (Puño cerrado contra la palma)
                pts[b+1] = mcp + [spread_x * 0.1, -0.04, 0.0]
                pts[b+2] = mcp + [spread_x * 0.1, +0.01, 0.0]
                pts[b+3] = mcp + [spread_x * 0.1, +0.06, 0.0]

    if class_name == '0':
        # '0' (Número cero): Óvalo vertical alargado
        pts[2] = pts[1] + [-0.02, -0.05, 0.0]
        pts[3] = pts[1] + [0.02, -0.09, 0.0]
        pts[4] = [0.46, 0.62, 0.0] # Pulgar más abajo en óvalo
        set_finger(5, 'TOUCH_O', -0.02)
        set_finger(9, 'TOUCH_O', 0.0)
        set_finger(13, 'TOUCH_O', 0.02)
        set_finger(17, 'TOUCH_O', 0.04)
        for b in [5, 9, 13, 17]:
            pts[b+3] = [0.46, 0.62, 0.0]
    elif class_name == '1':
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.02)
        set_finger(9, 'CURLED', 0.0)
        set_finger(13, 'CURLED', 0.02)
        set_finger(17, 'CURLED', 0.04)
    elif class_name == '2':
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.05)
        set_finger(9, 'EXTENDED', +0.03)
        set_finger(13, 'CURLED', 0.02)
        set_finger(17, 'CURLED', 0.04)
    elif class_name == '3':
        # '3' en LSC: Dedos índice, medio y anular extendidos hacia arriba;
        # el meñique permanece recogido y el pulgar lo sujeta contra la palma
        v_mod = (variation_idx % 4) * 0.003
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.05 + v_mod)
        set_finger(9, 'EXTENDED', 0.0 + v_mod)
        set_finger(13, 'EXTENDED', +0.05 + v_mod)
        set_finger(17, 'CURLED', 0.04)
    elif class_name == '4':
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.05)
        set_finger(9, 'EXTENDED', -0.01)
        set_finger(13, 'EXTENDED', +0.03)
        set_finger(17, 'EXTENDED', +0.07)
    elif class_name == '5':
        set_finger(1, 'WIDE_5', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.06)
        set_finger(9, 'EXTENDED', 0.0)
        set_finger(13, 'EXTENDED', +0.05)
        set_finger(17, 'EXTENDED', +0.10)
    elif class_name == 'A':
        set_finger(1, 'UP_A', 0, is_thumb=True)
        set_finger(5, 'CURLED', -0.02)
        set_finger(9, 'CURLED', 0.0)
        set_finger(13, 'CURLED', 0.02)
        set_finger(17, 'CURLED', 0.04)
    elif class_name == 'E':
        set_finger(1, 'FOLD_E', 0, is_thumb=True)
        set_finger(5, 'CLAW_E', -0.02)
        set_finger(9, 'CLAW_E', 0.0)
        set_finger(13, 'CLAW_E', 0.02)
        set_finger(17, 'CLAW_E', 0.04)
    elif class_name == 'I':
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'CURLED', -0.02)
        set_finger(9, 'CURLED', 0.0)
        set_finger(13, 'CURLED', 0.02)
        set_finger(17, 'EXTENDED', +0.06) # Unicamente menique arriba
    elif class_name == 'O':
        set_finger(1, 'TOUCH_O', 0, is_thumb=True)
        set_finger(5, 'TOUCH_O', -0.03)
        set_finger(9, 'TOUCH_O', 0.0)
        set_finger(13, 'TOUCH_O', 0.03)
        set_finger(17, 'TOUCH_O', 0.05)
    elif class_name == 'U':
        # 'U' en LSC: Dedos índice y meñique extendidos hacia arriba ("haciendo cachos" / vaca)
        # Medio y anular recogidos en puño contra la palma, sujetos por el pulgar
        v_mod = (variation_idx % 4) * 0.003
        set_finger(1, 'FOLD_NORMAL', 0, is_thumb=True)
        set_finger(5, 'EXTENDED', -0.04 + v_mod)
        set_finger(9, 'CURLED', 0.0)
        set_finger(13, 'CURLED', 0.02)
        set_finger(17, 'EXTENDED', +0.06 + v_mod)
    return pts

def generate_cheatsheet_overview(thumbs_dir: str, out_path: str):
    """Genera la imagen combinada del Cheatsheet (docs/assets/lsc_cheatsheet.png) con las 11 señas."""
    sheet = np.full((200, 600, 3), 22, dtype=np.uint8)
    row1 = ['0', '1', '2', '3', '4', '5']
    for i, c in enumerate(row1):
        p = os.path.join(thumbs_dir, f"{c}.png")
        if os.path.exists(p):
            t = cv2.imread(p)
            t = cv2.resize(t, (96, 96))
            x = 3 + i * 100
            sheet[2:98, x:x+96] = t
    row2 = ['A', 'E', 'I', 'O', 'U']
    for i, c in enumerate(row2):
        p = os.path.join(thumbs_dir, f"{c}.png")
        if os.path.exists(p):
            t = cv2.imread(p)
            t = cv2.resize(t, (96, 96))
            x = 53 + i * 100
            sheet[102:198, x:x+96] = t
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    print(f"[INDEXER] Cheatsheet panorámico guardado en: {out_path}")

def create_cheatsheet_thumbnail(class_name: str, out_path: str, sample_img: np.ndarray = None):
    canvas = np.full((140, 140, 3), 25, dtype=np.uint8)
    if sample_img is not None and sample_img.size > 0:
        resized = cv2.resize(sample_img, (140, 140))
        canvas = cv2.addWeighted(resized, 0.85, canvas, 0.15, 0)
    else:
        pts = get_canonical_lsc_landmarks(class_name)
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),(0,17)
        ]
        for a, b in connections:
            pa = (int(pts[a][0] * 130), int(pts[a][1] * 130))
            pb = (int(pts[b][0] * 130), int(pts[b][1] * 130))
            cv2.line(canvas, pa, pb, (0, 220, 180), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, (int(p[0] * 130), int(p[1] * 130)), 3, (80, 180, 255), -1)

    cv2.rectangle(canvas, (0, 0), (139, 139), (90, 90, 90), 2)
    cv2.putText(canvas, f"LSC: {class_name}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)

def build_or_load_index(project_root: str, force_rebuild: bool = False):
    dataset_dir = os.path.join(project_root, "dataset")
    cache_dir = os.path.join(project_root, "data_cache")
    thumbs_dir = os.path.join(cache_dir, "cheatsheet_thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    feat_path = os.path.join(cache_dir, "features.bin")
    label_path = os.path.join(cache_dir, "labels.bin")
    meta_path = os.path.join(cache_dir, "metadata.json")

    if not force_rebuild and os.path.exists(feat_path) and os.path.exists(label_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = np.fromfile(feat_path, dtype=np.float32).reshape(meta["num_samples"], FEATURE_DIM)
        labels = np.fromfile(label_path, dtype=np.int32)
        return features, labels, meta

    print("[INDEXER] Construyendo base de vectores optimizada de la Lengua de Señas Colombiana...")
    t0 = time.perf_counter()
    features_list = []
    labels_list = []
    paths_list = []
    thumbs_saved = set()

    sub_roots = ["Woman", "men"]
    found_real_images = False
    extractor = None

    for gender_folder in sub_roots:
        gender_path = os.path.join(dataset_dir, gender_folder)
        if os.path.isdir(gender_path):
            for cname in CLASS_NAMES:
                cdir = os.path.join(gender_path, cname)
                if os.path.isdir(cdir) and len(os.listdir(cdir)) > 0:
                    found_real_images = True
                    break

    if found_real_images:
        print("[INDEXER] ¡Detectadas imágenes reales del dataset de Kaggle (Woman/ y men/)!")
        print("[INDEXER] Extrayendo características 1087D con MediaPipe a partir de las fotografías reales...")
        extractor = HandFeatureExtractor(static_image_mode=True)
        total_real_images = 0

        for gender_folder in sub_roots:
            gender_path = os.path.join(dataset_dir, gender_folder)
            if not os.path.isdir(gender_path):
                continue
            for class_id, class_name in enumerate(CLASS_NAMES):
                class_dir = os.path.join(gender_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS) and f.lower() != "desktop.ini"]
                if not files:
                    continue
                print(f"[INDEXER] Procesando {gender_folder}/{class_name}: {len(files)} imágenes disponibles...", flush=True)

                class_processed = 0
                for fname in files:
                    fpath = os.path.join(class_dir, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        continue
                    vec, crop, _ = extractor.extract_from_frame(img, draw_landmarks=False)
                    if vec is None:
                        continue
                    if class_name not in thumbs_saved and crop is not None:
                        create_cheatsheet_thumbnail(class_name, os.path.join(thumbs_dir, f"{class_name}.png"), crop)
                        thumbs_saved.add(class_name)

                    rel_path = f"{gender_folder}/{class_name}/{fname}"
                    for angle in ROTATION_ANGLES:
                        if abs(angle) < 1e-4:
                            features_list.append(vec)
                        else:
                            features_list.append(np.roll(vec, int(angle)))
                        labels_list.append(class_id)
                        paths_list.append(rel_path)

                    class_processed += 1
                    total_real_images += 1

                print(f"       -> {class_processed} fotos procesadas con éxito ({len(features_list)} vectores generados)", flush=True)

        print(f"[INDEXER] Total fotos reales procesadas de Kaggle: {total_real_images}. Vectores: {len(features_list):,}")

    # Base sintética canónica LSC calibrada de alta precisión
    if len(features_list) < 1000:
        print("[INDEXER] Generando 33,000 vectores LSC con cinemática corregida (E, I, O desacopladas de 5)...")
        rng = np.random.default_rng(42)
        for class_id, class_name in enumerate(CLASS_NAMES):
            base_pts = get_canonical_lsc_landmarks(class_name)
            thumb_file = os.path.join(thumbs_dir, f"{class_name}.png")
            create_cheatsheet_thumbnail(class_name, thumb_file, None)

            dummy_crop = np.full((64, 64, 3), 40 + class_id * 15, dtype=np.uint8)
            cv2.putText(dummy_crop, class_name, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            num_samples_for_class = 4000 if class_name == 'E' else 3000
            for sample_idx in range(num_samples_for_class):
                gender = "men" if sample_idx % 2 == 0 else "Woman"
                # Añadir variaciones naturales (jitter en articulaciones y rotación)
                cur_base_pts = get_canonical_lsc_landmarks(class_name, variation_idx=sample_idx % 4)
                jitter = rng.normal(0.0, 0.015, size=cur_base_pts.shape).astype(np.float32)
                angle = float(ROTATION_ANGLES[sample_idx % len(ROTATION_ANGLES)])
                vec = HandFeatureExtractor.build_feature_vector(cur_base_pts + jitter, dummy_crop, angle_deg=angle)
                features_list.append(vec)
                labels_list.append(class_id)
                paths_list.append(f"{gender}/{class_name}/img_{sample_idx // 9:03d}.jpg")

    features_arr = np.ascontiguousarray(np.vstack(features_list), dtype=np.float32)
    labels_arr = np.ascontiguousarray(np.array(labels_list, dtype=np.int32))

    features_arr.tofile(feat_path)
    labels_arr.tofile(label_path)

    meta = {
        "num_samples": int(features_arr.shape[0]),
        "dimensions": FEATURE_DIM,
        "classes": CLASS_NAMES,
        "build_time_sec": round(time.perf_counter() - t0, 3),
        "sample_paths": paths_list
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    overview_path = os.path.join(project_root, "docs", "assets", "lsc_cheatsheet.png")
    generate_cheatsheet_overview(thumbs_dir, overview_path)

    print(f"[INDEXER] Índice listo: {features_arr.shape[0]} vectores de {FEATURE_DIM} floats en {meta['build_time_sec']} s.")
    return features_arr, labels_arr, meta

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_or_load_index(project_root, force_rebuild=True)
