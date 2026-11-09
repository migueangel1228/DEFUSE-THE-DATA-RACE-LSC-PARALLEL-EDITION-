import os
import random
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from bridge import ParallelEngineBridge
    from feature_extractor import HandFeatureExtractor, CLASS_NAMES, FEATURE_DIM
    from indexer import build_or_load_index, get_canonical_lsc_landmarks
    from sound_effects import SoundEffectsManager
    from camera_stream import ThreadedCameraStream
except ImportError:
    from src.bridge import ParallelEngineBridge
    from src.feature_extractor import HandFeatureExtractor, CLASS_NAMES, FEATURE_DIM
    from src.indexer import build_or_load_index, get_canonical_lsc_landmarks
    from src.sound_effects import SoundEffectsManager
    from src.camera_stream import ThreadedCameraStream

class DefuseDataRaceGame:
    def __init__(self, default_camera_idx: int = None):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset, self.labels, self.meta = build_or_load_index(self.project_root)
        self.bridge = ParallelEngineBridge()
        self.extractor = HandFeatureExtractor(static_image_mode=False)
        self.sound = SoundEffectsManager()

        # Cargar miniaturas del Cheatsheet
        self.thumbs = {}
        thumbs_dir = os.path.join(self.project_root, "data_cache", "cheatsheet_thumbs")
        for cname in CLASS_NAMES:
            tpath = os.path.join(thumbs_dir, f"{cname}.png")
            if os.path.exists(tpath):
                self.thumbs[cname] = cv2.imread(tpath)

        # Estado de configuracion paralela (Inicia SIEMPRE en Secuencial: 1 Hilo)
        self.num_threads = 1
        self.datarace_mode = False
        self.baseline_1thread_ms = 14.5

        # Modos de Juego: NORMAL (40s) vs COMPETITIVO (20s - Mitad de tiempo)
        self.game_mode = "NORMAL"  # "NORMAL" o "COMPETITIVO"
        self.best_time_normal = None
        self.best_time_comp = None
        self.defuse_time_achieved = None
        self.is_new_record = False

        # Estado del juego "Defuse the Data-Race"
        self.code_length = 5
        self.target_code = []
        self.current_lock_idx = 0
        self.hold_start_time = None
        self.hold_duration_required = 0.85
        self.bomb_timer_total = 40.0
        self.bomb_start_time = time.time()
        self.game_state = "PLAYING"  # PLAYING, WON, EXPLODED
        self.simulated_class = None
        self.last_hold_notch = 0

        # Gestión de Cámara Asíncrona (Threaded) para 60 FPS garantizados sin lag
        self.cam_stream = ThreadedCameraStream(start_index=default_camera_idx)
        self.camera_index = self.cam_stream.current_idx
        self.show_cam_diag_window = True
        self.cached_query_vec = None
        self.cached_hand_crop = None
        self.cached_annotated = None

        self.reset_game()

    def switch_camera(self):
        new_idx = self.cam_stream.switch_camera()
        self.camera_index = new_idx
        self.cached_annotated = None
        print(f"[CAMARA] Alternando a cámara #{new_idx}...")

    def toggle_game_mode(self):
        if self.game_mode == "NORMAL":
            self.game_mode = "COMPETITIVO"
            self.bomb_timer_total = 20.0  # ¡La mitad del tiempo exacto!
            print("[MODO] Activado MODO COMPETITIVO SPEEDRUN (20 segundos)")
        else:
            self.game_mode = "NORMAL"
            self.bomb_timer_total = 40.0
            print("[MODO] Activado MODO NORMAL (40 segundos)")
        self.reset_game()

    def reset_game(self):
        self.target_code = random.sample(CLASS_NAMES, self.code_length)
        self.current_lock_idx = 0
        self.hold_start_time = None
        self.bomb_start_time = time.time()
        self.game_state = "PLAYING"
        self.last_hold_notch = 0
        self.defuse_time_achieved = None
        self.is_new_record = False
        self.sound.start_tension_loop(
            lambda: max(0.0, self.bomb_timer_total - (time.time() - self.bomb_start_time)),
            lambda: self.game_state
        )

    def show_amdahl_benchmark_window(self, query_vec: np.ndarray):
        print("[BENCHMARK] Ejecutando prueba de escalabilidad con 1, 2, 4 y 8 hilos...")
        res = self.bridge.benchmark(self.dataset, self.labels, query_vec, repetitions=30)
        threads = res["threads"]
        times = res["times_ms"]
        speedups = res["speedups"]
        effs = res["efficiencies"]
        kf_vals = [v for v in res["karp_flatt"] if v > 0]
        serial_est = float(np.mean(kf_vals)) if kf_vals else 0.06

        amdahl_curve = [1.0 / (serial_est + (1.0 - serial_est) / p) for p in threads]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        fig.suptitle(
            f"Laboratorio de Rendimiento Paralelo - LSC Dataset ({self.dataset.shape[0]:,} vectores x {FEATURE_DIM}D)\n"
            f"Fracción Serial Experimental Estimada (e) = {serial_est*100:.2f}%",
            fontsize=12, fontweight="bold"
        )

        # Grafica 1: Tiempo de ejecucion Tp
        axes[0].plot(threads, times, marker="o", linewidth=2.5, color="#ff5722", label="Tiempo Paralelo Tp (ms)")
        axes[0].set_title("Tiempo de Búsqueda por Consulta (ms)")
        axes[0].set_xlabel("Número de Hilos (P)")
        axes[0].set_ylabel("Tiempo (ms)")
        axes[0].set_xticks(threads)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Grafica 2: Speed-up vs Ley de Amdahl
        axes[1].plot(threads, threads, "--", color="gray", label="Speed-up Lineal Ideal (Sp = P)")
        axes[1].plot(threads, amdahl_curve, ":", linewidth=2.2, color="#ffc107", label=f"Ley de Amdahl Teórica (e={serial_est:.3f})")
        axes[1].plot(threads, speedups, marker="s", linewidth=2.5, color="#00e676", label="Speed-up Experimental (T1 / Tp)")
        axes[1].set_title("Speed-up (Sp) y Ley de Amdahl")
        axes[1].set_xlabel("Número de Hilos (P)")
        axes[1].set_ylabel("Speed-up (x veces)")
        axes[1].set_xticks(threads)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Grafica 3: Eficiencia
        axes[2].bar([str(p) for p in threads], [e * 100.0 for e in effs], color="#29b6f6", width=0.55)
        axes[2].axhline(100.0, linestyle="--", color="gray")
        axes[2].set_title("Eficiencia Paralela Ep = (Sp / P) %")
        axes[2].set_xlabel("Número de Hilos (P)")
        axes[2].set_ylabel("Eficiencia (%)")
        axes[2].set_ylim(0, 115)
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        window_name = "DEFUSE THE DATA-RACE | LSC Parallel Search Engine (OpenMP + Python)"
        cam_window_name = "MONITOR DE CAMARA (DIAGNOSTICO) | Defuse the Data-Race"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        cv2.namedWindow(cam_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_window_name, 640, 520)

        # Medir tiempo base con 1 hilo al inicio
        dummy_q = self.dataset[0].copy()
        res_1 = self.bridge.search(self.dataset, self.labels, dummy_q, num_threads=1, datarace_mode=False)
        self.baseline_1thread_ms = max(0.5, res_1["elapsed_ms"])

        running = True
        fps_time = time.time()
        fps_count = 0
        fps_display = 30.0

        while running:
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_count / (time.time() - fps_time)
                fps_count = 0
                fps_time = time.time()

            frame, is_new, sensor_fps = self.cam_stream.get_frame()
            mean_brightness = float(np.mean(frame)) if frame is not None else 0.0

            if frame is None:
                frame = np.full((480, 640, 3), 25, dtype=np.uint8)
                cv2.putText(frame, f"CAMARA #{self.camera_index} NO RESPONDE", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Presiona [C] para probar la otra camara (0 <-> 1)", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                cv2.putText(frame, "O presiona [S] para simular la sena actual", (30, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 118), 1)
                query_vec = None
                hand_crop = None
                annotated_cam = frame
            else:
                frame = cv2.flip(frame, 1)
                if is_new or self.cached_annotated is None:
                    query_vec, hand_crop, annotated_cam = self.extractor.extract_from_frame(frame, draw_landmarks=True)
                    self.cached_query_vec = query_vec
                    self.cached_hand_crop = hand_crop
                    self.cached_annotated = annotated_cam
                else:
                    query_vec = self.cached_query_vec
                    hand_crop = self.cached_hand_crop
                    annotated_cam = self.cached_annotated

            # Simulación con tecla [S]
            is_simulated = False
            if query_vec is None and self.simulated_class is not None:
                pts = get_canonical_lsc_landmarks(self.simulated_class)
                dummy_crop = np.full((64, 64, 3), 40 + CLASS_NAMES.index(self.simulated_class) * 15, dtype=np.uint8)
                query_vec = HandFeatureExtractor.build_feature_vector(pts, dummy_crop, 0.0)
                is_simulated = True

            search_result = None
            if query_vec is not None:
                search_result = self.bridge.search(
                    self.dataset, self.labels, query_vec,
                    num_threads=self.num_threads,
                    datarace_mode=self.datarace_mode
                )

                # Ponderación calibrada para seña 'E' con mano derecha
                e_idx = CLASS_NAMES.index('E')
                if self.game_state == "PLAYING" and self.target_code[self.current_lock_idx] == 'E':
                    search_result["class_scores"][e_idx] += 0.055
                else:
                    search_result["class_scores"][e_idx] += 0.035

                if search_result["class_scores"][e_idx] > search_result["best_score"]:
                    search_result["best_score"] = float(search_result["class_scores"][e_idx])
                    search_result["best_class"] = e_idx

            # Lógica de temporizador y candados
            elapsed_game = time.time() - self.bomb_start_time
            remaining_time = max(0.0, self.bomb_timer_total - elapsed_game)
            if self.game_state == "PLAYING" and remaining_time <= 0.0:
                self.game_state = "EXPLODED"
                self.sound.stop_tension()
                self.sound.play("explosion")

            hold_progress = 0.0
            if self.game_state == "PLAYING" and search_result is not None:
                detected_name = CLASS_NAMES[search_result["best_class"]]
                target_name = self.target_code[self.current_lock_idx]
                score_val = search_result["best_score"]
                is_matching = (detected_name == target_name) or (target_name == 'O' and detected_name == '0') or (target_name == '0' and detected_name == 'O')
                threshold = 0.63 if not is_simulated else 0.58

                if is_matching and score_val >= threshold and not self.datarace_mode:
                    if self.hold_start_time is None:
                        self.hold_start_time = time.time()
                    hold_progress = min(1.0, (time.time() - self.hold_start_time) / self.hold_duration_required)

                    # Sonido de confirmación al sostener
                    notch = int(hold_progress * 4)
                    if notch > self.last_hold_notch:
                        self.last_hold_notch = notch
                        self.sound.play("hold_tick")

                    if hold_progress >= 1.0:
                        self.current_lock_idx += 1
                        self.hold_start_time = None
                        self.last_hold_notch = 0
                        if self.current_lock_idx >= self.code_length:
                            self.game_state = "WON"
                            self.sound.stop_tension()
                            self.sound.play("victory")
                            defuse_time = time.time() - self.bomb_start_time
                            self.defuse_time_achieved = defuse_time
                            # Registrar récord
                            if self.game_mode == "COMPETITIVO":
                                if self.best_time_comp is None or defuse_time < self.best_time_comp:
                                    self.best_time_comp = defuse_time
                                    self.is_new_record = True
                            else:
                                if self.best_time_normal is None or defuse_time < self.best_time_normal:
                                    self.best_time_normal = defuse_time
                                    self.is_new_record = True
                        else:
                            self.sound.play("unlock")
                else:
                    self.hold_start_time = None
                    self.last_hold_notch = 0

            # Renderizar ventana de Diagnóstico de Cámara ("Pestaña" dedicada)
            if self.show_cam_diag_window:
                diag_frame = self._render_camera_diagnostic(
                    annotated_cam, search_result, mean_brightness, fps_display, hand_crop, is_simulated, sensor_fps
                )
                cv2.imshow(cam_window_name, diag_frame)

            # Renderizar Dashboard principal de 1280x720 con efectos de estrés visual
            dashboard = self._render_dashboard(
                annotated_cam, search_result, remaining_time, hold_progress, mean_brightness, fps_display, sensor_fps
            )
            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'):
                running = False
            elif key == ord(' '):
                self.reset_game()
            elif key == 9 or key == ord('t') or key == ord('T'):  # TAB o T para cambiar modo
                self.toggle_game_mode()
            elif key == ord('m') or key == ord('M'):
                self.sound.toggle_mute()
            elif key == ord('c') or key == ord('C'):
                self.switch_camera()
            elif key == ord('v') or key == ord('V'):
                self.show_cam_diag_window = not self.show_cam_diag_window
                if not self.show_cam_diag_window:
                    cv2.destroyWindow(cam_window_name)
                else:
                    cv2.namedWindow(cam_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(cam_window_name, 640, 520)
            elif key in [ord('1'), ord('2'), ord('4'), ord('8')]:
                self.num_threads = int(chr(key))
            elif key == ord('r') or key == ord('R'):
                self.datarace_mode = not self.datarace_mode
                self.sound.play("data_race")
            elif key == ord('b') or key == ord('B'):
                q_bench = query_vec if query_vec is not None else self.dataset[0]
                self.show_amdahl_benchmark_window(q_bench)
            elif key == ord('n') or key == ord('N'):
                if self.game_state == "PLAYING":
                    self.current_lock_idx += 1
                    if self.current_lock_idx >= self.code_length:
                        self.game_state = "WON"
                        self.sound.stop_tension()
                        self.sound.play("victory")
                        self.defuse_time_achieved = time.time() - self.bomb_start_time
                    else:
                        self.sound.play("unlock")
            elif key == ord('s') or key == ord('S'):
                if self.game_state == "PLAYING":
                    self.simulated_class = self.target_code[self.current_lock_idx]
            else:
                self.simulated_class = None

        self.sound.stop_tension()
        if hasattr(self, "cam_stream"):
            self.cam_stream.stop()
        cv2.destroyAllWindows()

    def _render_camera_diagnostic(
        self, cam_img: np.ndarray, search_res: dict, mean_brightness: float,
        fps: float, hand_crop: np.ndarray, is_simulated: bool, sensor_fps: float = 0.0
    ) -> np.ndarray:
        diag = np.full((520, 640, 3), 20, dtype=np.uint8)

        # 1. Feed de cámara escalado (640x360)
        h, w = cam_img.shape[:2]
        scaled_cam = cv2.resize(cam_img, (640, 360))
        diag[0:360, 0:640] = scaled_cam

        # Overlay superior con estado de cámara
        cv2.rectangle(diag, (0, 0), (640, 35), (10, 10, 15), -1)
        cam_status_col = (0, 230, 118) if mean_brightness >= 5.0 else (0, 60, 255)
        sound_status = "AUDIO: [ON]" if self.sound.enabled else "AUDIO: [MUTE]"
        mode_txt = f"[{self.game_mode}]"
        cv2.putText(diag, f"CAM #{self.camera_index} | UI: {fps:.0f} FPS | SENSOR: {sensor_fps:.1f} FPS | {sound_status}",
                    (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, cam_status_col, 2, cv2.LINE_AA)

        # Si el brillo es extremadamente bajo (pantalla negra), desplegar diagnóstico claro
        if mean_brightness < 5.0:
            cv2.rectangle(diag, (20, 50), (620, 340), (0, 0, 180), 2)
            overlay = diag[50:340, 20:620].copy()
            cv2.rectangle(diag, (20, 50), (620, 340), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.2, diag[50:340, 20:620], 0.8, 0, diag[50:340, 20:620])

            cv2.putText(diag, "! ATENCION: SENSOR EN NEGRO !", (40, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)
            tips = [
                "1. Abre la tapa/obturador fisico de privacidad de tu webcam.",
                "2. La camara en oscuridad total baja por hardware a 1 FPS para exponer.",
                "3. Presiona la tecla [C] para alternar entre Camara #0 y Camara #1.",
                "4. Ajustes Windows > Privacidad > Camara (Habilitar acceso apps escritorio).",
                "5. Mientras tanto, presiona [S] para simular la sena actual sin lag."
            ]
            for i, tip in enumerate(tips):
                cv2.putText(diag, tip, (40, 125 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # 2. Panel inferior con información del extractor de características
        cv2.rectangle(diag, (0, 360), (640, 520), (28, 28, 35), -1)
        cv2.line(diag, (0, 360), (640, 360), (0, 200, 255), 2)

        target_name = self.target_code[self.current_lock_idx] if self.game_state == "PLAYING" else "-"
        cv2.putText(diag, f"Candado Objetivo: [{target_name}]", (15, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)

        if is_simulated:
            cv2.putText(diag, f"Modo: SIMULANDO SENA [{self.simulated_class}] (Tecla [S])", (15, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2, cv2.LINE_AA)
        elif search_res is not None:
            best_c = CLASS_NAMES[search_res["best_class"]]
            sc_pct = search_res["best_score"] * 100.0
            is_match = (best_c == target_name) or (target_name == 'O' and best_c == '0') or (target_name == '0' and best_c == 'O')
            match_col = (0, 255, 120) if is_match else (100, 200, 255)
            cv2.putText(diag, f"Sena Detectada: [{best_c}] | Similitud: {sc_pct:.1f}%", (15, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, match_col, 2, cv2.LINE_AA)
        else:
            cv2.putText(diag, "Mano: No detectada (Coloca tu mano frente al lente)", (15, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

        # Miniatura recortada de la mano si existe
        if hand_crop is not None and hand_crop.size > 0:
            thumb_crop = cv2.resize(hand_crop, (90, 90))
            diag[375:465, 530:620] = thumb_crop
            cv2.rectangle(diag, (530, 375), (620, 465), (0, 220, 180), 1)
            cv2.putText(diag, "Recorte 1087D", (525, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Atajos en la barra inferior
        cv2.putText(diag, "[C] Camara  |  [TAB] Modo  |  [M] Mute  |  [S] Simular  |  [N] Saltar",
                    (15, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1, cv2.LINE_AA)

        return diag

    def _render_dashboard(
        self, cam_img: np.ndarray, search_res: dict, remaining_time: float,
        hold_progress: float, mean_brightness: float, fps_display: float = 30.0, sensor_fps: float = 0.0
    ) -> np.ndarray:
        ui = np.full((720, 1280, 3), 18, dtype=np.uint8)

        # 1. ZONA SUPERIOR IZQUIERDA: Camara en vivo (560x380)
        cam_resized = cv2.resize(cam_img, (540, 360))
        ui[20:380, 20:560] = cam_resized
        border_col = (0, 200, 255) if mean_brightness >= 5.0 else (0, 60, 255)
        cv2.rectangle(ui, (20, 20), (560, 380), border_col, 2)

        # Indicador de cámara, atajo [C] y Audio
        sound_badge = "SND: ON" if self.sound.enabled else "SND: MUTE"
        cam_label = f"CAM #{self.camera_index} [C]  |  UI: {fps_display:.0f} FPS  |  SENSOR: {sensor_fps:.1f} FPS  |  {sound_badge} [M]"
        cv2.putText(ui, cam_label, (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        if mean_brightness < 5.0:
            cv2.putText(ui, "! CAMARA EN NEGRO (Shutter cerrado) ! Presiona [C] o [S]", (28, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 2, cv2.LINE_AA)

        # Barra de confirmacion de sena
        bar_w = int(540 * hold_progress)
        cv2.rectangle(ui, (20, 385), (560, 405), (45, 45, 45), -1)
        cv2.rectangle(ui, (20, 385), (20 + bar_w, 405), (0, 230, 118), -1)
        cv2.putText(ui, f"CONFIRMACION DE SENA: {int(hold_progress*100)}%", (28, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. ZONA SUPERIOR DERECHA: Bomba "Defuse the Data-Race" y los 5 Candados
        cv2.rectangle(ui, (580, 20), (1260, 405), (32, 32, 40), -1)
        cv2.rectangle(ui, (580, 20), (1260, 405), (90, 90, 110), 2)

        # Badge del Modo Actual
        is_comp = (self.game_mode == "COMPETITIVO")
        mode_badge_bg = (0, 60, 220) if is_comp else (70, 70, 85)
        mode_badge_txt = "[MODO: COMPETITIVO SPEEDRUN (20s)]" if is_comp else "[MODO: NORMAL (40s)]"
        cv2.rectangle(ui, (950, 30), (1250, 60), mode_badge_bg, -1)
        cv2.putText(ui, mode_badge_txt, (958, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        title_color = (0, 230, 118) if self.game_state == "WON" else ((0, 60, 255) if self.game_state == "EXPLODED" else (0, 210, 255))

        if self.game_state == "WON":
            status_txt = f"¡DESACTIVADA EN {self.defuse_time_achieved:.2f} s! [ESPACIO] Nueva Mision"
        elif self.game_state == "EXPLODED":
            status_txt = "DEADLOCK EXPLOTO! [ESPACIO] Reintentar"
        else:
            status_txt = f"TIEMPO RESTANTE: {remaining_time:04.1f} s"

        cv2.putText(ui, "DEFUSE THE DATA-RACE (CODIGO LSC)", (600, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ui, status_txt, (600, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_color, 2, cv2.LINE_AA)

        # Dibujar los 5 candados y sus miniaturas Cheatsheet
        for idx, cname in enumerate(self.target_code):
            x0 = 600 + idx * 130
            y0 = 115
            unlocked = idx < self.current_lock_idx
            is_current = (idx == self.current_lock_idx and self.game_state == "PLAYING")
            box_col = (0, 200, 83) if unlocked else ((0, 215, 255) if is_current else (75, 75, 85))

            cv2.rectangle(ui, (x0, y0), (x0 + 118, y0 + 95), (22, 22, 26), -1)
            cv2.rectangle(ui, (x0, y0), (x0 + 118, y0 + 95), box_col, 3 if is_current else 2)
            cv2.putText(ui, cname, (x0 + 40, y0 + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, box_col, 3, cv2.LINE_AA)

            # Miniatura de ayuda debajo de cada candado
            if cname in self.thumbs:
                th = cv2.resize(self.thumbs[cname], (118, 118))
                ui[y0 + 105:y0 + 223, x0:x0 + 118] = th

        # Mensaje de Récord o Guía
        cur_rec = self.best_time_comp if is_comp else self.best_time_normal
        rec_txt = f"Record {self.game_mode}: {cur_rec:.2f} s" if cur_rec else f"Sin record en {self.game_mode}"
        cv2.putText(ui, f"{rec_txt}  |  Presiona [TAB] para alternar Modo Normal / Competitivo",
                    (600, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 210, 255), 1, cv2.LINE_AA)

        # 3. ZONA INFERIOR IZQUIERDA: Match del Dataset (men/ y Woman/)
        cv2.rectangle(ui, (20, 420), (560, 700), (28, 28, 34), -1)
        cv2.rectangle(ui, (20, 420), (560, 700), (80, 80, 95), 1)
        cv2.putText(ui, "INSPECTOR DE BUSQUEDA KNN MASIVA (KAGGLE LSC)", (35, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2, cv2.LINE_AA)

        if search_res is not None:
            det_cls = CLASS_NAMES[search_res["best_class"]]
            score_pct = max(0.0, min(100.0, search_res["best_score"] * 100.0))
            matched_path = self.meta["sample_paths"][search_res["best_index"]]
            cv2.putText(ui, f"Sena Detectada: [{det_cls}]  Similitud: {score_pct:.1f}%", (35, 485),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 160), 2, cv2.LINE_AA)
            cv2.putText(ui, f"Archivo Ganador: dataset/{matched_path}", (35, 515),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)

            # Barras de similitud de las 11 clases
            for c_idx, cname in enumerate(CLASS_NAMES):
                bx = 35 + c_idx * 46
                sc = max(0.0, min(1.0, float(search_res["class_scores"][c_idx])))
                bh = int(sc * 110)
                bcol = (0, 230, 118) if c_idx == search_res["best_class"] else (180, 110, 40)
                cv2.rectangle(ui, (bx, 665 - bh), (bx + 32, 665), bcol, -1)
                cv2.putText(ui, cname, (bx + 8, 688), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.putText(ui, "Coloca tu mano frente a la camara para buscar en los 34,000 vectores...",
                        (35, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

        # 4. ZONA INFERIOR DERECHA: Telemetría Paralela OpenMP y Demostrador de Data-Race
        cv2.rectangle(ui, (580, 420), (1260, 700), (28, 28, 34), -1)
        border_col = (0, 0, 255) if self.datarace_mode else ((0, 215, 255) if self.num_threads == 1 else (0, 230, 118))
        cv2.rectangle(ui, (580, 420), (1260, 700), border_col, 2)

        if self.datarace_mode:
            mode_label = "ALERTA: MODO DATA-RACE (SIN #pragma omp critical)"
        elif self.num_threads == 1:
            mode_label = "MODO SECUENCIAL (1 HILO BASE T1) - Presiona [2], [4] u [8]"
        else:
            mode_label = f"MODO PARALELO ACELERADO: {self.num_threads} HILOS OpenMP (MapReduce)"
        cv2.putText(ui, mode_label, (600, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.60, border_col, 2, cv2.LINE_AA)

        if search_res is not None:
            t_ms = max(0.05, search_res["elapsed_ms"])
            if self.num_threads == 1:
                self.baseline_1thread_ms = t_ms
                sp_now = 1.00
                eff_pct = 100.0
                time_line = f"Tiempo Secuencial Base (T1): {t_ms:.2f} ms (1 Hilo de Ejecucion)"
                sp_line = "Speed-up Lineal Base: 1.00x  |  Presiona [2], [4] u [8] para acelerar"
            else:
                sp_now = self.baseline_1thread_ms / t_ms
                eff_pct = min(100.0, (sp_now / float(self.num_threads)) * 100.0)
                time_line = f"Tiempo Paralelo (Tp): {t_ms:.2f} ms  |  Base Secuencial (T1): {self.baseline_1thread_ms:.2f} ms"
                sp_line = f"Speed-up Instantaneo (Sp = T1/Tp): {sp_now:.2f}x  |  Eficiencia Ep: {eff_pct:.1f}%"

            ver_c = search_res["verified_count"]
            exp_c = search_res["expected_count"]
            lost_pct = ((exp_c - ver_c) / float(exp_c)) * 100.0

            cv2.putText(ui, time_line, (600, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ui, sp_line, (600, 522), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2, cv2.LINE_AA)

            cnt_color = (60, 80, 255) if lost_pct > 0.1 else (0, 230, 118)
            cv2.putText(ui, f"Iteraciones Contabilizadas: {ver_c:,} / {exp_c:,} (Perdida por Data-Race: {lost_pct:.1f}%)",
                        (600, 556), cv2.FONT_HERSHEY_SIMPLEX, 0.50, cnt_color, 1, cv2.LINE_AA)

        # Leyenda de controles para el expositor
        controls = [
            "[TAB] o [T]: Alternar Modo Normal (40s) / Competitivo (20s)   |   [M]: Audio On/Mute",
            "[1] [2] [4] [8]: Cambiar hilos OpenMP (Inicia en 1 hilo secuencial)",
            "[R]: Activar / Desactivar Condicion de Carrera (Data-Race Demo)",
            "[B]: Ver Graficas Ley de Amdahl y Eficiencia   |   [C]: Cambiar Camara",
            "[ESPACIO]: Nuevo Codigo   |   [S]: Simular Sena   |   [N]: Saltar Candado"
        ]
        for idx, line in enumerate(controls):
            cv2.putText(ui, line, (600, 595 + idx * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1, cv2.LINE_AA)

        # =========================================================================
        # 5. EFECTOS VISUALES DE ESTRÉS: PARPADEOS ROJOS Y VIBRACIÓN (< 10 SEGUNDOS)
        # =========================================================================
        if self.game_state == "PLAYING" and remaining_time <= 10.0:
            # Frecuencia de parpadeo se acelera conforme baja el tiempo
            freq = 12.0 if remaining_time > 4.5 else 22.0
            pulse = float((np.sin(time.time() * freq) + 1.0) / 2.0)  # oscila 0.0 a 1.0

            # Capa roja transparente en los bordes
            overlay = ui.copy()
            b_thick = int(24 + 14 * pulse)
            # Bordes superior, inferior, izquierdo, derecho
            cv2.rectangle(overlay, (0, 0), (1280, b_thick), (0, 0, 240), -1)
            cv2.rectangle(overlay, (0, 720 - b_thick), (1280, 720), (0, 0, 240), -1)
            cv2.rectangle(overlay, (0, 0), (b_thick, 720), (0, 0, 240), -1)
            cv2.rectangle(overlay, (1280 - b_thick, 0), (1280, 720), (0, 0, 240), -1)

            # Tinte rojo transparente sobre el área de la bomba
            cv2.rectangle(overlay, (580, 20), (1260, 405), (0, 0, 180), -1)

            alpha = float(0.20 + 0.38 * pulse)
            cv2.addWeighted(overlay, alpha, ui, 1.0 - alpha, 0, ui)

            # Banner parpadeante de peligro inminente en los últimos 5 segundos
            if remaining_time <= 5.0:
                shake_x = int(np.random.randint(-4, 5) * pulse)
                shake_y = int(np.random.randint(-3, 4) * pulse)
                M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                ui = cv2.warpAffine(ui, M, (1280, 720), borderMode=cv2.BORDER_REFLECT)

                warning_txt = f"! DETONACION INMINENTE: {remaining_time:04.1f} s !"
                cv2.rectangle(ui, (340, 335), (940, 385), (0, 0, 220), -1)
                cv2.rectangle(ui, (340, 335), (940, 385), (255, 255, 255), 2)
                cv2.putText(ui, warning_txt, (360, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)

        # Pantalla final de victoria con felicitaciones y récord
        if self.game_state == "WON" and self.defuse_time_achieved is not None:
            overlay_win = ui.copy()
            cv2.rectangle(overlay_win, (240, 200), (1040, 520), (15, 25, 20), -1)
            cv2.rectangle(overlay_win, (240, 200), (1040, 520), (0, 230, 118), 3)
            cv2.addWeighted(overlay_win, 0.88, ui, 0.12, 0, ui)

            cv2.putText(ui, "¡¡ BOMBA DESACTIVADA CON EXITO !!", (270, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 240, 120), 3, cv2.LINE_AA)
            cv2.putText(ui, f"TIEMPO LOGRADO: {self.defuse_time_achieved:.2f} s  |  MODO: {self.game_mode}", (280, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            if self.is_new_record:
                cv2.putText(ui, "¡¡ NUEVO RECORD HISTORICO DE DESACTIVACION !!", (280, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 220, 255), 2, cv2.LINE_AA)
            else:
                cur_best = self.best_time_comp if is_comp else self.best_time_normal
                rec_str = f"{cur_best:.2f} s" if cur_best is not None else "-- s"
                cv2.putText(ui, f"Mejor Record en {self.game_mode}: {rec_str}", (280, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 220, 240), 2, cv2.LINE_AA)

            cv2.putText(ui, "Presiona [ESPACIO] para jugar otra mision  |  [TAB] para cambiar Modo", (280, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 215, 255), 1, cv2.LINE_AA)

        return ui

if __name__ == "__main__":
    game = DefuseDataRaceGame()
    game.run()
