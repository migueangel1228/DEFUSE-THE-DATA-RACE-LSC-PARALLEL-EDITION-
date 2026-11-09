import os
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import numpy as np

# Añadir src al path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from feature_extractor import HandFeatureExtractor, CLASS_NAMES, FEATURE_DIM
from bridge import ParallelEngineBridge
from indexer import build_or_load_index
from camera_stream import ThreadedCameraStream

class SingleCameraTester:
    def __init__(self, start_cam_idx: int = None):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.dataset, self.labels, self.meta = build_or_load_index(self.project_root)
        self.bridge = ParallelEngineBridge()
        self.extractor = HandFeatureExtractor(static_image_mode=False)

        self.cam_stream = ThreadedCameraStream(start_index=start_cam_idx)
        self.cam_idx = self.cam_stream.current_idx
        self.cached_annotated = None
        self.cached_vec = None
        self.cached_res = None

    def switch_camera(self):
        self.cam_idx = self.cam_stream.switch_camera()
        self.cached_annotated = None
        print(f"[TEST CAMARA] Cambiando a cámara #{self.cam_idx}...")

    def run(self):
        window_name = "MONITOR DE CAMARA INDIVIDUAL (FLUIDO Y SIN LAG) - LSC"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 680)

        fps_count = 0
        fps_time = time.time()
        fps_val = 60.0

        print("\n" + "=" * 60)
        print("    MONITOR INDIVIDUAL DE CAMARA - LSC PARALLEL EDITION")
        print("=" * 60)
        print(" - Captura desacoplada en hilo secundario para 60 FPS fluidos.")
        print(" - Presiona [C] para alternar entre Cámara #0 y Cámara #1.")
        print(" - Presiona [Q] o [ESC] para salir.\n")

        while True:
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps_val = fps_count / (time.time() - fps_time)
                fps_count = 0
                fps_time = time.time()

            frame, is_new, sensor_fps = self.cam_stream.get_frame()
            canvas = np.full((680, 960, 3), 20, dtype=np.uint8)
            mean_b = float(np.mean(frame)) if frame is not None else 0.0

            if frame is None:
                cv2.putText(canvas, f"CAMARA #{self.cam_idx} NO EMITE SENAL", (80, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(canvas, "Presiona [C] para probar la otra camara (0 <-> 1)", (80, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
                search_res = None
            else:
                frame = cv2.flip(frame, 1)
                if is_new or self.cached_annotated is None:
                    vec, crop, annotated = self.extractor.extract_from_frame(frame, draw_landmarks=True)
                    self.cached_annotated = annotated
                    self.cached_vec = vec

                    search_res = None
                    if vec is not None:
                        search_res = self.bridge.search(
                            self.dataset, self.labels, vec, num_threads=4, datarace_mode=False
                        )
                        e_idx = CLASS_NAMES.index('E')
                        search_res["class_scores"][e_idx] += 0.04
                        if search_res["class_scores"][e_idx] > search_res["best_score"]:
                            search_res["best_score"] = float(search_res["class_scores"][e_idx])
                            search_res["best_class"] = e_idx
                    self.cached_res = search_res
                else:
                    annotated = self.cached_annotated
                    search_res = self.cached_res

                scaled = cv2.resize(annotated, (960, 540))
                canvas[0:540, 0:960] = scaled

            # Barra superior con FPS de Render y FPS de Hardware
            cv2.rectangle(canvas, (0, 0), (960, 40), (12, 12, 16), -1)
            b_col = (0, 230, 118) if mean_b >= 5.0 else (0, 60, 255)
            cv2.putText(canvas, f"CAMARA #{self.cam_idx} | UI: {fps_val:.0f} FPS | SENSOR: {sensor_fps:.1f} FPS | BRILLO: {mean_b:.1f}",
                        (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, b_col, 2, cv2.LINE_AA)
            cv2.putText(canvas, "[C] Cambiar Camara  |  [Q] Salir",
                        (670, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Si está en negro
            if mean_b < 5.0:
                cv2.putText(canvas, "! CAMARA OSCURA/EN NEGRO ! Presiona [C] o revisa el obturador fisico",
                            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 80, 255), 2, cv2.LINE_AA)

            # Panel inferior de resultados KNN
            cv2.rectangle(canvas, (0, 540), (960, 680), (28, 28, 35), -1)
            cv2.line(canvas, (0, 540), (960, 540), (0, 200, 255), 2)

            if search_res is not None:
                det_c = CLASS_NAMES[search_res["best_class"]]
                pct = search_res["best_score"] * 100.0
                cv2.putText(canvas, f"SENA RECONOCIDA: [{det_c}]  |  Similitud: {pct:.1f}%  |  Tiempo C++: {search_res['elapsed_ms']:.2f} ms",
                            (25, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 120), 2, cv2.LINE_AA)

                # Barras de similitud de las 11 clases
                for c_idx, cname in enumerate(CLASS_NAMES):
                    bx = 25 + c_idx * 84
                    sc = max(0.0, min(1.0, float(search_res["class_scores"][c_idx])))
                    bh = int(sc * 45)
                    col = (0, 230, 118) if c_idx == search_res["best_class"] else (180, 110, 40)
                    cv2.rectangle(canvas, (bx, 645 - bh), (bx + 55, 645), col, -1)
                    cv2.putText(canvas, f"{cname}:{int(sc*100)}%", (bx + 2, 665),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Coloca tu mano frente a la camara para identificar la sena LSC...",
                            (25, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):
                self.switch_camera()

        self.cam_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SingleCameraTester(start_cam_idx=None)
    tester.run()
