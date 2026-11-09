import os
import threading
import time
import cv2
import numpy as np

def find_working_camera_index(preferred: int = 1) -> int:
    """Detecta automáticamente el primer índice de cámara DirectShow que responde."""
    candidates = [preferred, 0, 1, 2]
    seen = set()
    order = [x for x in candidates if not (x in seen or seen.add(x))]
    
    for idx in order:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return idx
        except Exception:
            pass
    return preferred

class ThreadedCameraStream:
    """
    Gestor de captura de video asíncrono y desacoplado del hilo principal.
    Garantiza que la interfaz de usuario corra a 60 FPS sin importar si el sensor
    está bloqueado, en timeout o si la tapa física de privacidad está cerrada.
    """
    def __init__(self, start_index: int = None):
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self.current_idx = start_index if start_index is not None else find_working_camera_index(preferred=1)
        self.latest_frame = None
        self.frame_id = 0
        self.last_consumed_id = -1
        self.sensor_fps = 0.0
        self.thread = None

        self.open_camera(self.current_idx)

    def open_camera(self, idx: int):
        self.stop()
        self.current_idx = idx
        print(f"[CAM_STREAM] Iniciando captura asíncrona desacoplada en cámara #{idx}...")
        
        try:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
        except Exception as exc:
            print(f"[CAM_STREAM] Error abriendo cámara #{idx}: {exc}")
            self.cap = cv2.VideoCapture(idx)

        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception:
                pass

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    fps_counter += 1
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_id += 1
                else:
                    time.sleep(0.015)
            else:
                time.sleep(0.05)

            now = time.time()
            if now - fps_timer >= 1.0:
                with self.lock:
                    self.sensor_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

    def get_frame(self):
        """
        Devuelve (frame, is_new_frame, sensor_fps).
        Operación no bloqueante (< 0.05 milisegundos).
        """
        with self.lock:
            if self.latest_frame is None:
                return None, False, self.sensor_fps
            is_new = (self.frame_id != self.last_consumed_id)
            self.last_consumed_id = self.frame_id
            return self.latest_frame.copy(), is_new, self.sensor_fps

    def switch_camera(self) -> int:
        next_idx = 0 if self.current_idx == 1 else 1
        print(f"[CAM_STREAM] Alternando a cámara #{next_idx}...")
        self.open_camera(next_idx)
        return next_idx

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.6)
        self.thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
