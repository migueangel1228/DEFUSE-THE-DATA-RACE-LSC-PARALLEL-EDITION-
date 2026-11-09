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
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.enabled = HAS_AUDIO
        self.volume = 0.95
        self.sounds = {}

        if self.enabled:
            self._synthesize_sound_bank()
            self._load_external_explosion()

        self.tension_running = False
        self.tension_thread = None
        self.get_time_remaining = None
        self.get_game_state = None

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
                    self.sounds["explosion"] = (audio_16 * 0.99).astype(np.float32)
                    self.explosion_rate = rate
                    print(f"[AUDIO] Sonido real de explosión cargado y normalizado al 100% de volumen ({len(audio_16)} muestras a {rate} Hz).")
                    return
            except Exception as e:
                print(f"[AUDIO] Advertencia al cargar explosion_loud.wav ({e}). Usando sintetizado.")
        self.explosion_rate = self.sr

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

        # 5. Explosión de respaldo si no está el archivo WAV (Estallido cinemático profundo)
        t_ex = np.linspace(0, 2.5, int(sr * 2.5), False)
        noise = np.random.uniform(-1, 1, len(t_ex)) * np.exp(-2.8 * t_ex)
        rumble = np.sin(2 * np.pi * np.linspace(90, 20, len(t_ex)) * t_ex) * np.exp(-1.5 * t_ex)
        blast = 0.60 * noise + 0.40 * rumble
        self.sounds["explosion"] = (blast * 1.0).astype(np.float32)
        self.explosion_rate = sr

        # 6. Fanfarria de Victoria / Bomb Defused! (Acorde triunfal C-E-G-C brillante y heroico)
        t_v = np.linspace(0, 1.6, int(sr * 1.6), False)
        notes = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        victory = np.zeros_like(t_v)
        for i, f in enumerate(notes):
            delay_samples = int(i * 0.10 * sr)
            env = np.zeros_like(t_v)
            if delay_samples < len(t_v):
                dt = t_v[:len(t_v) - delay_samples]
                env[delay_samples:] = np.sin(2 * np.pi * f * dt) * np.exp(-1.8 * dt)
            victory += env
        victory = victory / np.max(np.abs(victory))
        self.sounds["victory"] = (victory * 0.95).astype(np.float32)

        # 7. Glitch de Data-Race (Chispazo eléctrico saturado)
        t_g = np.linspace(0, 0.30, int(sr * 0.30), False)
        sq = np.sign(np.sin(2 * np.pi * 120 * t_g)) * 0.5
        burst = np.random.uniform(-1, 1, len(t_g)) * 0.5
        glitch = (sq + burst) * np.exp(-7 * t_g)
        self.sounds["data_race"] = (glitch * 0.95).astype(np.float32)

        # 8. Micro-beep de progreso al sostener la seña (Hold tick)
        t_h = np.linspace(0, 0.05, int(sr * 0.05), False)
        htick = np.sin(2 * np.pi * 1400 * t_h) * np.exp(-25 * t_h)
        self.sounds["hold_tick"] = (htick * 0.60).astype(np.float32)

    def play(self, sound_name: str):
        if not self.enabled or sound_name not in self.sounds:
            return
        try:
            rate = self.explosion_rate if sound_name == "explosion" else self.sr
            audio = np.clip(self.sounds[sound_name] * self.volume, -0.98, 0.98)
            sd.play(audio, rate, blocking=False)
        except Exception:
            pass

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

            # Escalado progresivo de tensión sonora:
            # > 15s: latido sordo de corazón cada 0.9s
            # 10s - 15s: beep de advertencia cada 0.5s
            # 5s - 10s: alarma de presión cada 0.30s
            # < 5s: ¡Pánico extremo acelerado cada 0.15s!
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
                interval = 0.15  # ¡Pulsaciones frenéticas de detonación inminente!
                snd = "panic_beep"

            if now - last_played >= interval:
                self.play(snd)
                last_played = now

            time.sleep(0.015)

    def toggle_mute(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled
