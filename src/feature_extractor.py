import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    HAS_MEDIAPIPE = False

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', 'A', 'E', 'I', 'O', 'U']
FEATURE_DIM = 1087  # 63 geometricos + 1024 bordes Sobel 32x32

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandFeatureExtractor:
    def __init__(self, static_image_mode: bool = False):
        self.mode = "none"
        self.hands = None
        self.task_detector = None

        if HAS_MEDIAPIPE:
            # 1. Probar API clasica (mp.solutions) si existe
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                try:
                    self.mp_hands = mp.solutions.hands
                    self.mp_draw = mp.solutions.drawing_utils
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=static_image_mode,
                        max_num_hands=1,
                        min_detection_confidence=0.55,
                        min_tracking_confidence=0.55
                    )
                    self.mode = "solutions"
                except Exception:
                    self.hands = None

            # 2. Si no tiene solutions (p. ej. MediaPipe v1+ en Python 3.13), usar Tasks API
            if self.mode == "none":
                try:
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision

                    src_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(src_dir, "hand_landmarker.task")
                    if os.path.exists(model_path):
                        base_options = mp_python.BaseOptions(model_asset_path=model_path)
                        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
                        self.task_detector = vision.HandLandmarker.create_from_options(options)
                        self.mode = "tasks"
                except Exception:
                    self.task_detector = None

    def extract_from_frame(self, bgr_frame: np.ndarray, draw_landmarks: bool = True):
        if bgr_frame is None or bgr_frame.size == 0:
            return None, None, bgr_frame

        h, w, _ = bgr_frame.shape
        annotated = bgr_frame.copy()
        coords = None

        if self.mode == "solutions" and self.hands is not None:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.hands.process(rgb_frame)
                if results and results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    if draw_landmarks and self.mp_draw:
                        self.mp_draw.draw_landmarks(annotated, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            except Exception:
                coords = None

        elif self.mode == "tasks" and self.task_detector is not None:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.task_detector.detect(mp_img)
                if res and res.hand_landmarks and len(res.hand_landmarks) > 0:
                    lms = res.hand_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    if draw_landmarks:
                        for p1, p2 in HAND_CONNECTIONS:
                            x1, y1 = int(coords[p1, 0] * w), int(coords[p1, 1] * h)
                            x2, y2 = int(coords[p2, 0] * w), int(coords[p2, 1] * h)
                            cv2.line(annotated, (x1, y1), (x2, y2), (0, 220, 180), 2, cv2.LINE_AA)
                        for lm in coords:
                            cx, cy = int(lm[0] * w), int(lm[1] * h)
                            cv2.circle(annotated, (cx, cy), 3, (80, 180, 255), -1)
            except Exception:
                coords = None

        if coords is None:
            return None, None, annotated

        # Recorte del Bounding Box de la mano
        xs = (coords[:, 0] * w).astype(np.int32)
        ys = (coords[:, 1] * h).astype(np.int32)
        margin_x = int(max(20, (np.max(xs) - np.min(xs)) * 0.20))
        margin_y = int(max(20, (np.max(ys) - np.min(ys)) * 0.20))
        x1 = max(0, int(np.min(xs) - margin_x))
        y1 = max(0, int(np.min(ys) - margin_y))
        x2 = min(w, int(np.max(xs) + margin_x))
        y2 = min(h, int(np.max(ys) + margin_y))

        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = bgr_frame

        feature_vec = self.build_feature_vector(coords, crop, angle_deg=0.0)
        return feature_vec, crop, annotated

    @staticmethod
    def build_feature_vector(coords_21x3: np.ndarray, crop_bgr: np.ndarray, angle_deg: float = 0.0) -> np.ndarray:
        # 1. Normalizacion geometrica centrada en el centroide de la palma (63 floats)
        # Usar el centro de la palma (promedio de muñeca 0 y bases MCP 5, 9, 17) para desacoplar
        # la direccion de dedos extendidos (apuntan hacia afuera/arriba) de dedos flexionados/recogidos
        # (apuntan hacia adentro/abajo). Esto elimina la falsa confusion de E, I, O con 5.
        palm_center = (coords_21x3[0:1, :] + coords_21x3[5:6, :] + coords_21x3[9:10, :] + coords_21x3[17:18, :]) / 4.0
        centered = coords_21x3 - palm_center

        if abs(angle_deg) > 1e-4:
            rad = np.deg2rad(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot_mat = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            centered = np.dot(centered, rot_mat.T)

        max_dist = float(np.max(np.linalg.norm(centered, axis=1)))
        if max_dist < 1e-6:
            max_dist = 1.0
        geom_vec = (centered / max_dist).flatten().astype(np.float32) * 4.5

        # 2. Silueta de bordes Sobel 32x32 (1024 floats)
        if crop_bgr is None or crop_bgr.size == 0:
            crop_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        if abs(angle_deg) > 1e-4:
            M = cv2.getRotationMatrix2D((16, 16), angle_deg, 1.0)
            resized = cv2.warpAffine(resized, M, (32, 32), borderMode=cv2.BORDER_REPLICATE)

        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy).flatten().astype(np.float32)
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 1e-6:
            mag = mag / mag_norm

        # 3. Fusion y normalizacion L2 unitaria global (1087 floats)
        combined = np.concatenate([geom_vec, mag]).astype(np.float32)
        total_norm = np.linalg.norm(combined)
        if total_norm > 1e-6:
            combined = combined / total_norm
        return combined
