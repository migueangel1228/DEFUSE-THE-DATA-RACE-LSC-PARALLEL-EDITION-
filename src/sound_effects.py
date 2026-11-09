import os
import threading
import time
import wave
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

class SoundEffectsManager:
    """
    Motor de efectos de audio polifónico no bloqueante de latencia ultra baja.
    Utiliza un único OutputStream persistente con callback para mezclar los canales.
    Garantiza que JAMÁS se llame a sd.play() de forma concurrente, eliminando el 100%
    de los bloqueos de PortAudio, tirones de FPS y caídas repentinas de la aplicación.
    """
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.enabled = HAS_AUDIO
        self.volume = 0.95
        self.sounds = {}
        self.lock = threading.Lock()
        self.active_voices = []  # Lista de tuplas: [audio_array, current_sample_index]
        self.stream = None

        if self.enabled:
            self._synthesize_sound_bank()
            self._load_external_explosion()
            self._init_output_stream()

        self.tension_running = False
        self.tension_thread = None
        self.get_time_remaining = None
        self.get_game_state = None

    def _init_output_stream(self):
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                dtype='float32',
                callback=self._mixer_callback,
                blocksize=512
            )
            self.stream.start()
        except Exception as e:
            print(f"[AUDIO] Advertencia: No se pudo abrir el stream de audio ({e}). Modo silencioso activado.")
            self.enabled = False

    def _mixer_callback(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        if not self.enabled:
            outdata[:] = buf.reshape(-1, 1)
            return

        with self.lock:
            remaining_voices = []
            for sound, pos in self.active_voices:
                left = len(sound) - pos
                if left > 0:
                    chunk = min(frames, left)
                    buf[:chunk] += sound[pos:pos+chunk]
                    if pos + chunk < len(sound):
                        remaining_voices.append((sound, pos + chunk))
            self.active_voices = remaining_voices

        # Normalización y protección contra saturación digital
        outdata[:] = np.clip(buf * self.volume, -0.98, 0.98).reshape(-1, 1)

    def _load_external_explosion(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(src_dir, "explosion_loud.wav")
        if os.path.exists(wav_path):
            try:
                with wave.open(wav_path, "rb") as wf:
                    n_frames = wf.getnframes()
                    rate = wf.getframerate()
                    data = wf.readframes(n_frames)
                    audio_16 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    peak = float(np.max(np.abs(audio_16)))
                    if peak > 0.001:
                        audio_16 = audio_16 / peak

                    # Remuestreo limpio a 44,100 Hz para compatibilidad nativa con el mezclador
                    if rate != self.sr:
                        new_len = int(len(audio_16) * self.sr / rate)
                        audio_16 = np.interp(
                            np.linspace(0, len(audio_16), new_len, endpoint=False),
                            np.arange(len(audio_16)),
                            audio_16
                        ).astype(np.float32)

                    self.sounds["explosion"] = (audio_16 * 0.99).astype(np.float32)
                    print(f"[AUDIO] Sonido real de explosión cargado y mezclado al 100% ({len(self.sounds['explosion'])} muestras a {self.sr} Hz).")
                    return
            except Exception as e:
                print(f"[AUDIO] Advertencia al cargar explosion_loud.wav ({e}). Usando sintetizado.")

    def _synthesize_sound_bank(self):
        sr = self.sr

        # 1. Latido sordo de presión / Heartbeat (50-60 Hz contundente)
        t_hb = np.linspace(0, 0.28, int(sr * 0.28), False)
        hb = (np.sin(2 * np.pi * 55 * t_hb) + 0.3 * np.sin(2 * np.pi * 110 * t_hb)) * np.exp(-10 * t_hb)
        self.sounds["heartbeat"] = (hb * 0.95).astype(np.float32)

        # 2. Beep digital tenso / Bomb Tick (1300 Hz metálico penetrante)
        t_tick = np.linspace(0, 0.08, int(sr * 0.08), False)
        tick = (np.sin(2 * np.pi * 1350 * t_tick) + 0.4 * np.sin(2 * np.pi * 2700 * t_tick)) * np.exp(-22 * t_tick)
        self.sounds["tick"] = (tick * 0.90).astype(np.float32)

        # 3. Alarma de Pánico Extremo (< 10s restantes) (1750 Hz + armónico de sonar)
        t_p = np.linspace(0, 0.09, int(sr * 0.09), False)
        panic = (np.sin(2 * np.pi * 1750 * t_p) + 0.6 * np.sin(2 * np.pi * 3500 * t_p)) * np.exp(-18 * t_p)
        self.sounds["panic_beep"] = (panic * 0.95).astype(np.float32)

        # 4. Desbloqueo de Candado / Lock Cracking (Acorde digital ascendente contundente)
        t_u = np.linspace(0, 0.35, int(sr * 0.35), False)
        f_sweep = np.linspace(950, 1850, len(t_u))
        unlock = (0.7 * np.sin(2 * np.pi * f_sweep * t_u) + 0.4 * np.sin(3 * np.pi * f_sweep * t_u)) * np.exp(-8 * t_u)
        self.sounds["unlock"] = (unlock * 0.95).astype(np.float32)

        # 5. Explosión sintética de respaldo
        t_ex = np.linspace(0, 2.5, int(sr * 2.5), False)
        noise = np.random.uniform(-1, 1, len(t_ex)) * np.exp(-2.8 * t_ex)
        rumble = np.sin(2 * np.pi * np.linspace(90, 20, len(t_ex)) * t_ex) * np.exp(-1.5 * t_ex)
        blast = 0.60 * noise + 0.40 * rumble
        self.sounds["explosion"] = (blast * 0.99).astype(np.float32)

        # 6. Fanfarria de Victoria / Bomb Defused! (Acorde triunfal C-E-G-C brillante y heroico)
        t_v = np.linspace(0, 1.6, int(sr * 1.6), False)
        notes = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        victory = np.zeros_like(t_v)
        for i, f in enumerate(notes):
            delay_samples = int(i * 0.10 * sr)
            if delay_samples < len(t_v):
                dt = t_v[:len(t_v) - delay_samples]
                victory[delay_samples:] += np.sin(2 * np.pi * f * dt) * np.exp(-1.8 * dt)
        victory = victory / np.max(np.abs(victory))
        self.sounds["victory"] = (victory * 0.95).astype(np.float32)

        # 7. Glitch de Data-Race (Chispazo eléctrico saturado)
        t_g = np.linspace(0, 0.30, int(sr * 0.30), False)
        sq = np.sign(np.sin(2 * np.pi * 120 * t_g)) * 0.5
        burst = np.random.uniform(-1, 1, len(t_g)) * 0.5
        glitch = (sq + burst) * np.exp(-7 * t_g)
        self.sounds["data_race"] = (glitch * 0.95).astype(np.float32)

        # 8. Micro-beep de progreso al sostener la seña (Hold tick - ultra nítido y rápido)
        t_h = np.linspace(0, 0.04, int(sr * 0.04), False)
        htick = np.sin(2 * np.pi * 1400 * t_h) * np.exp(-30 * t_h)
        self.sounds["hold_tick"] = (htick * 0.55).astype(np.float32)

    def play(self, sound_name: str):
        """Disparo de sonido no bloqueante (tiempo de CPU: < 0.001 ms)."""
        if not self.enabled or sound_name not in self.sounds:
            return
        audio = self.sounds[sound_name]
        with self.lock:
            # Máximo 16 voces simultáneas para evitar saturación
            if len(self.active_voices) < 16:
                self.active_voices.append((audio, 0))

    def start_tension_loop(self, get_time_func, get_state_func):
        self.get_time_remaining = get_time_func
        self.get_game_state = get_state_func
        self.tension_running = True

        if self.tension_thread is None or not self.tension_thread.is_alive():
            self.tension_thread = threading.Thread(target=self._tension_worker, daemon=True)
            self.tension_thread.start()

    def stop_tension(self):
        self.tension_running = False

    def _tension_worker(self):
        last_played = time.time()
        while self.tension_running:
            if not self.enabled or self.get_time_remaining is None or self.get_game_state is None:
                time.sleep(0.05)
                continue

            state = self.get_game_state()
            if state != "PLAYING":
                time.sleep(0.08)
                continue

            rem = max(0.0, self.get_time_remaining())
            now = time.time()

            # Escalado progresivo de tensión sonora
            if rem > 15.0:
                interval = 0.90
                snd = "heartbeat"
            elif rem > 10.0:
                interval = 0.50
                snd = "tick"
            elif rem > 5.0:
                interval = 0.30
                snd = "panic_beep"
            else:
                interval = 0.15
                snd = "panic_beep"

            if now - last_played >= interval:
                self.play(snd)
                last_played = now

            time.sleep(0.015)

    def toggle_mute(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            with self.lock:
                self.active_voices.clear()
        return self.enabled

    def close(self):
        self.stop_tension()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
