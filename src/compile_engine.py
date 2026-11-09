import os
import platform
import shutil
import subprocess
import sys

def ensure_mingw_in_path():
    system_name = platform.system().lower()
    if "windows" in system_name:
        for p in [r"C:\msys64\mingw64\bin", r"C:\msys64\ucrt64\bin"]:
            if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

def get_gpp_command():
    ensure_mingw_in_path()
    system_name = platform.system().lower()
    if "windows" in system_name:
        candidates = [
            r"C:\msys64\mingw64\bin\g++.exe",
            r"C:\msys64\ucrt64\bin\g++.exe",
            shutil.which("g++"),
            "g++"
        ]
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    res = subprocess.run([cand, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                    if res.returncode == 0:
                        return cand
                except Exception:
                    pass
        return "g++"
    return "g++"

def compile_shared_library():
    ensure_mingw_in_path()
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_path = os.path.join(src_dir, "parallel_engine.cpp")
    system_name = platform.system().lower()
    gpp_bin = get_gpp_command()

    if "windows" in system_name:
        lib_name = "parallel_engine.dll"
        cmd = [gpp_bin, "-O3", "-fopenmp", "-shared", cpp_path, "-o", os.path.join(src_dir, lib_name)]
    elif "darwin" in system_name:
        lib_name = "libparallel_engine.dylib"
        cmd = [gpp_bin, "-O3", "-Xpreprocessor", "-fopenmp", "-lomp", "-shared", "-fPIC", cpp_path, "-o", os.path.join(src_dir, lib_name)]
    else:
        lib_name = "libparallel_engine.so"
        cmd = [gpp_bin, "-O3", "-fopenmp", "-shared", "-fPIC", cpp_path, "-o", os.path.join(src_dir, lib_name)]

    print(f"[COMPILADOR] Ejecutando: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[OK] Librería compilada exitosamente en: {os.path.join(src_dir, lib_name)}")
        return True
    except Exception as exc:
        print(f"[ADVERTENCIA] No se pudo compilar con g++ ({exc}). El puente usará el respaldo paralelo en Python si no encuentra el binario.")
        return False

if __name__ == "__main__":
    compile_shared_library()
