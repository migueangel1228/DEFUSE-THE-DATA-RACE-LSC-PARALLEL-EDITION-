# Directorio del Dataset de Lengua de Señas Colombiana (Kaggle)

Si dispones de los 12.29 GB del dataset completo de Kaggle, descomprímelo dentro de esta carpeta manteniendo la estructura:

```text
dataset/
├── Woman/
│   ├── 0/ ... 5/
│   ├── A/, E/, I/, O/, U/
│   └── desktop.ini          (El indexador lo ignora automáticamente)
├── men/
│   ├── 0/ ... 5/
│   ├── A/, E/, I/, O/, U/
│   └── desktop.ini          (El indexador lo ignora automáticamente)
└── desktop.ini              (El indexador lo ignora automáticamente)
```

Una vez descomprimido, ejecuta `rebuild_index.bat` o `python src/indexer.py` para procesar las imágenes y regenerar la caché en `data_cache/`.

Si aún no has colocado las carpetas `Woman/` y `men/`, no te preocupes: el sistema ya generó automáticamente una base canónica de 33,000 vectores aumentados con las 11 señas oficiales de la LSC para que puedas jugar y realizar los benchmarks de Amdahl y Data-Race de inmediato.
