"""
Detector y Reconocedor de Objetos — con detección de manos primero
===================================================================
Compatible con MediaPipe >= 0.10 (Tasks API).

Flujo:
  1. MediaPipe HandLandmarker detecta manos en el frame completo.
  2. Dentro de la zona de captura (centro), se enmascara la piel de las manos
     para aislar solo el objeto que se sostiene.
  3. Las características se extraen ÚNICAMENTE del objeto (sin piel).

Controles:
  [L] - Modo Aprendizaje   [R] - Modo Reconocimiento
  [N] - Nuevo objeto       [S] - Capturar muestra
  [C] - Limpiar BD         [Q] - Salir
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import numpy as np
import os
import pickle
import time
import urllib.request

# ──────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────
DB_FILE       = "objetos_aprendidos.pkl"
MIN_MUESTRAS  = 8
UMBRAL_SIM    = 0.30
REGION_W      = 340
REGION_H      = 340
SKIN_BLUR     = 15
MIN_OBJ_AREA  = 1500

MODEL_PATH    = "hand_landmarker.task"
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

PIEL_BAJO  = np.array([0,  20,  60],  dtype=np.uint8)
PIEL_ALTO  = np.array([25, 255, 255], dtype=np.uint8)

# Conexiones de mano para dibujo manual (índices de landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


# ──────────────────────────────────────────────
#  DESCARGA DEL MODELO
# ──────────────────────────────────────────────
def descargar_modelo() -> None:
    """Descarga el modelo hand_landmarker.task si no existe."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"Descargando modelo de manos ({MODEL_PATH})...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Modelo descargado correctamente.")
    except Exception as e:
        print(f"  ERROR al descargar el modelo: {e}")
        print(f"  Descárgalo manualmente de:\n  {MODEL_URL}")
        raise


# ──────────────────────────────────────────────
#  INICIALIZACIÓN MEDIAPIPE (Tasks API >= 0.10)
# ──────────────────────────────────────────────
def crear_detector_manos():
    """
    Crea un HandLandmarker en modo IMAGE (síncrono, adecuado para bucle).
    Devuelve el detector listo para usar.
    """
    descargar_modelo()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────
#  DETECCIÓN DE MANOS
# ──────────────────────────────────────────────
def detectar_manos(frame: np.ndarray, detector) -> object:
    """Procesa el frame (BGR) y devuelve HandLandmarkerResult."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect(mp_image)


def obtener_bbox_manos(results, frame_shape: tuple) -> list:
    """
    Devuelve lista de bounding boxes (x1,y1,x2,y2) de cada mano,
    expandidas 15 % para cubrir bien la palma y la muñeca.
    """
    h, w = frame_shape[:2]
    bboxes = []
    if not results.hand_landmarks:
        return bboxes
    for hand_lm in results.hand_landmarks:
        xs = [lm.x * w for lm in hand_lm]
        ys = [lm.y * h for lm in hand_lm]
        pad_x = (max(xs) - min(xs)) * 0.15
        pad_y = (max(ys) - min(ys)) * 0.15
        x1 = max(0, int(min(xs) - pad_x))
        y1 = max(0, int(min(ys) - pad_y))
        x2 = min(w, int(max(xs) + pad_x))
        y2 = min(h, int(max(ys) + pad_y))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def dibujar_manos(frame: np.ndarray, results) -> None:
    """Dibuja landmarks y conexiones de manos sobre el frame (sin mp_draw)."""
    if not results.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_lm in results.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)
            cv2.circle(frame, pt, 3, (0, 150, 200),   1)


# ──────────────────────────────────────────────
#  ZONA DE CAPTURA (ROI)
# ──────────────────────────────────────────────
def coords_roi(frame_shape: tuple) -> tuple:
    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2
    x1 = cx - REGION_W // 2
    y1 = cy - REGION_H // 2
    return x1, y1, x1 + REGION_W, y1 + REGION_H


def recortar_roi(frame: np.ndarray) -> tuple:
    coords = coords_roi(frame.shape)
    x1, y1, x2, y2 = coords
    return frame[y1:y2, x1:x2].copy(), coords


# ──────────────────────────────────────────────
#  MÁSCARA DE PIEL / AISLAMIENTO DEL OBJETO
# ──────────────────────────────────────────────
def mascara_piel(roi: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycr  = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    m_hsv = cv2.inRange(hsv, PIEL_BAJO, PIEL_ALTO)
    m_ycr = cv2.inRange(ycr,
                        np.array([0,  133,  77], dtype=np.uint8),
                        np.array([255, 173, 127], dtype=np.uint8))
    mask = cv2.bitwise_or(m_hsv, m_ycr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SKIN_BLUR, SKIN_BLUR))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.GaussianBlur(mask, (SKIN_BLUR, SKIN_BLUR), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def aislar_objeto(roi: np.ndarray, hand_bboxes_global: list,
                  roi_coords: tuple) -> tuple:
    rx1, ry1, rx2, ry2 = roi_coords
    m_piel = mascara_piel(roi)

    m_manos = np.zeros(roi.shape[:2], dtype=np.uint8)
    for (hx1, hy1, hx2, hy2) in hand_bboxes_global:
        lx1 = max(0, hx1 - rx1)
        ly1 = max(0, hy1 - ry1)
        lx2 = min(REGION_W, hx2 - rx1)
        ly2 = min(REGION_H, hy2 - ry1)
        if lx2 > lx1 and ly2 > ly1:
            m_manos[ly1:ly2, lx1:lx2] = 255

    m_excluir = cv2.bitwise_or(m_piel, m_manos)
    m_objeto  = cv2.bitwise_not(m_excluir)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m_objeto = cv2.morphologyEx(m_objeto, cv2.MORPH_OPEN,  kernel)
    m_objeto = cv2.morphologyEx(m_objeto, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(m_objeto, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    m_final = np.zeros_like(m_objeto)
    area    = 0
    if contornos:
        mayor = max(contornos, key=cv2.contourArea)
        area  = int(cv2.contourArea(mayor))
        if area >= MIN_OBJ_AREA:
            cv2.drawContours(m_final, [mayor], -1, 255, -1)

    roi_objeto = cv2.bitwise_and(roi, roi, mask=m_final)
    return roi_objeto, m_final, area


# ──────────────────────────────────────────────
#  EXTRACCIÓN DE CARACTERÍSTICAS
# ──────────────────────────────────────────────
def extraer_histograma(roi: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mascara,
                        [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def extraer_hog(roi: np.ndarray) -> np.ndarray:
    gris    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gris, (64, 64))
    gx      = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy      = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    cell    = 8
    nc_x    = resized.shape[1] // cell
    nc_y    = resized.shape[0] // cell
    hist_t  = []
    for cy in range(nc_y):
        for cx_ in range(nc_x):
            a = ang[cy*cell:(cy+1)*cell, cx_*cell:(cx_+1)*cell]
            m = mag[cy*cell:(cy+1)*cell, cx_*cell:(cx_+1)*cell]
            h, _ = np.histogram(a, bins=8, range=(0, 2*np.pi), weights=m)
            hist_t.extend(h)
    feat = np.array(hist_t, dtype=np.float32)
    n    = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def extraer_momentos(mascara: np.ndarray) -> np.ndarray:
    M  = cv2.moments(mascara)
    hu = cv2.HuMoments(M).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    n  = np.linalg.norm(hu)
    return hu / n if n > 0 else hu


def extraer_caracteristicas(roi_obj: np.ndarray, mascara: np.ndarray):
    if mascara is None or cv2.countNonZero(mascara) < MIN_OBJ_AREA:
        return None
    return {
        "color":    extraer_histograma(roi_obj, mascara),
        "forma":    extraer_hog(roi_obj),
        "momentos": extraer_momentos(mascara),
    }


# ──────────────────────────────────────────────
#  SIMILITUD Y RECONOCIMIENTO
# ──────────────────────────────────────────────
def similitud_color(h1: np.ndarray, h2: np.ndarray) -> float:
    return cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    )


def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    dot  = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.clip(dot / norm, 0, 1)) if norm > 0 else 0.0


def comparar_muestra(q: dict, m: dict) -> float:
    sc = similitud_color(q["color"],     m["color"])
    sf = similitud_coseno(q["forma"],    m["forma"])
    sh = similitud_coseno(q["momentos"], m["momentos"])
    return 0.50 * sc + 0.30 * sf + 0.20 * sh


def reconocer_objeto(feats: dict, db: dict) -> tuple:
    mejor, score = "Desconocido", 0.0
    for nombre, muestras in db.items():
        if len(muestras) < MIN_MUESTRAS:
            continue
        scores = sorted(
            [comparar_muestra(feats, m) for m in muestras], reverse=True
        )[:5]
        s = float(np.mean(scores))
        if s > score:
            score, mejor = s, nombre
    if score < UMBRAL_SIM:
        return "Desconocido", score
    return mejor, score


# ──────────────────────────────────────────────
#  BASE DE DATOS
# ──────────────────────────────────────────────
def cargar_base_datos() -> dict:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def guardar_base_datos(db: dict) -> None:
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def limpiar_base_datos() -> dict:
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    return {}


def agregar_muestra(db: dict, nombre: str,
                    roi_obj: np.ndarray, mascara: np.ndarray) -> bool:
    feats = extraer_caracteristicas(roi_obj, mascara)
    if feats is None:
        return False
    db.setdefault(nombre, []).append(feats)
    guardar_base_datos(db)
    return True


# ──────────────────────────────────────────────
#  DIBUJO / HUD
# ──────────────────────────────────────────────
def dibujar_zona_captura(frame: np.ndarray, coords: tuple,
                         color: tuple, label: str = "") -> None:
    x1, y1, x2, y2 = coords
    L = 22
    for (sx, sy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx, sy), (sx + dx * L, sy),  color, 2)
        cv2.line(frame, (sx, sy), (sx, sy + dy * L),  color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def dibujar_panel_superior(frame: np.ndarray, modo: str,
                            manos: int, objeto_actual: str,
                            db: dict) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 52), (15, 15, 15), -1)
    color_modo = (0, 210, 100) if modo == "APRENDIZAJE" else (80, 170, 255)
    cv2.putText(frame, f"MODO: {modo}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_modo, 2)
    estado_m = f"Manos: {manos}" if manos else "Sin manos"
    color_m  = (0, 200, 80) if manos else (60, 60, 200)
    cv2.putText(frame, estado_m, (w // 2 - 50, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_m, 2)
    info = f"Obj: {len(db)}  Muestras: {sum(len(v) for v in db.values())}"
    cv2.putText(frame, info, (w - 290, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)


def dibujar_estado_inferior(frame: np.ndarray, objeto_actual: str,
                             db: dict, msg: str = "") -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 38), (w, h), (15, 15, 15), -1)
    if msg:
        cv2.putText(frame, msg, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 60), 1)
    elif objeto_actual:
        n = len(db.get(objeto_actual, []))
        t = f"Aprendiendo: '{objeto_actual}'  [{n}/{MIN_MUESTRAS} muestras]"
        cv2.putText(frame, t, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 100), 1)


def dibujar_leyenda(frame: np.ndarray) -> None:
    controles = [
        ("[Q] Salir",       (100,100,100)),
        ("[C] Limpiar",     (0,100,255)),
        ("[N] Nuevo obj.",  (200,100,255)),
        ("[S] Capturar",    (255,200,0)),
        ("[R] Reconocer",   (80,170,255)),
        ("[L] Aprendizaje", (0,210,100)),
    ]
    h = frame.shape[0]
    for i, (txt, col) in enumerate(controles):
        cv2.putText(frame, txt, (frame.shape[1] - 178, h - 42 - i * 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def dibujar_lista_objetos(frame: np.ndarray, db: dict,
                           objeto_actual: str) -> None:
    y0 = 60
    cv2.putText(frame, "Objetos aprendidos:", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    for i, (nom, muestras) in enumerate(db.items()):
        listo = len(muestras) >= MIN_MUESTRAS
        col   = (0, 210, 100) if listo else (80, 80, 80)
        mark  = "v" if listo else "."
        activo = " <" if nom == objeto_actual else ""
        cv2.putText(frame, f"  {mark} {nom} [{len(muestras)}]{activo}",
                    (10, y0 + 18 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def dibujar_resultado_reconocimiento(frame: np.ndarray,
                                     nombre: str, conf: float,
                                     roi_coords: tuple) -> None:
    _, _, _, ry2 = roi_coords
    h, w = frame.shape[:2]
    color = (60, 220, 60) if nombre != "Desconocido" else (60, 60, 220)
    label = f"{nombre}  {conf*100:.1f}%"
    ts    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cx    = (w - ts[0]) // 2
    cv2.putText(frame, label, (cx, ry2 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    bw, bh = 220, 10
    bx, by = (w - bw) // 2, ry2 + 50
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + bh), color, -1)


def dibujar_contorno_objeto(frame: np.ndarray, mascara: np.ndarray,
                             roi_coords: tuple) -> None:
    if mascara is None or cv2.countNonZero(mascara) == 0:
        return
    rx1, ry1 = roi_coords[0], roi_coords[1]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        c_global = c + np.array([[[rx1, ry1]]])
        cv2.drawContours(frame, [c_global], -1, (0, 255, 200), 2)


def mostrar_flash(frame: np.ndarray) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]),
                  (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)


def dibujar_aviso_sin_objeto(frame: np.ndarray, roi_coords: tuple,
                              area: int, n_manos: int) -> None:
    x1, y1 = roi_coords[0], roi_coords[1]
    if n_manos == 0:
        cv2.putText(frame, "Coloca tus manos con el objeto en la zona",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 1)
    elif area < MIN_OBJ_AREA:
        cv2.putText(frame, "Objeto no visible — acercalo mas",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 80, 255), 1)


# ──────────────────────────────────────────────
#  ENTRADA DE NOMBRE
# ──────────────────────────────────────────────
def pedir_nombre_objeto() -> str:
    nombre = ""
    panel  = np.zeros((80, 440, 3), dtype=np.uint8)
    while True:
        panel[:] = (22, 22, 22)
        cv2.putText(panel, "Nombre del objeto  (ENTER=confirmar  ESC=cancelar)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(panel, f"> {nombre}_",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 100), 2)
        cv2.imshow("Nuevo objeto", panel)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and nombre.strip():
            break
        elif key == 27:
            nombre = ""
            break
        elif key == 8 and nombre:
            nombre = nombre[:-1]
        elif 32 <= key <= 126:
            nombre += chr(key)
    cv2.destroyWindow("Nuevo objeto")
    return nombre.strip()


# ──────────────────────────────────────────────
#  LOOP PRINCIPAL
# ──────────────────────────────────────────────
def main() -> None:
    db               = cargar_base_datos()
    modo             = "APRENDIZAJE"
    objeto_actual    = ""
    flash_ts         = 0.0
    flash_activo     = False
    msg_tmp          = ""
    msg_ts           = 0.0
    ultimo_resultado = ("", 0.0)

    print("Inicializando detector de manos...")
    detector = crear_detector_manos()
    print("  Detector listo.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la camara.")
        detector.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 54)
    print("  Detector de Objetos con Manos  ")
    print("=" * 54)
    print("  [N] Nuevo objeto     [S] Capturar muestra")
    print("  [L] Aprendizaje      [R] Reconocimiento")
    print("  [C] Limpiar BD       [Q] Salir")
    print("=" * 54)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # ── 1. Detectar manos ─────────────────────────────────
        results  = detectar_manos(frame, detector)
        n_manos  = len(results.hand_landmarks) if results.hand_landmarks else 0
        bboxes_m = obtener_bbox_manos(results, frame.shape)

        # ── 2. ROI central ────────────────────────────────────
        roi, roi_coords = recortar_roi(frame)

        # ── 3. Aislar objeto ──────────────────────────────────
        roi_obj, mascara_obj, area_obj = aislar_objeto(roi, bboxes_m, roi_coords)

        # ── 4. Lógica de modo ─────────────────────────────────
        color_zona = (0, 210, 100)

        if modo == "RECONOCIMIENTO":
            color_zona = (80, 170, 255)
            feats = extraer_caracteristicas(roi_obj, mascara_obj)
            if feats:
                nombre, conf = reconocer_objeto(feats, db)
                ultimo_resultado = (nombre, conf)
            dibujar_resultado_reconocimiento(
                frame, ultimo_resultado[0], ultimo_resultado[1], roi_coords)

        # Flash de captura
        if flash_activo and time.time() - flash_ts < 0.12:
            mostrar_flash(frame)
        else:
            flash_activo = False

        # ── 5. Elementos visuales ─────────────────────────────
        dibujar_manos(frame, results)

        for (hx1, hy1, hx2, hy2) in bboxes_m:
            overlay = frame.copy()
            cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 200, 255), -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 200, 255), 1)

        dibujar_contorno_objeto(frame, mascara_obj, roi_coords)

        etiq_zona = "ZONA DE CAPTURA" if n_manos else "Coloca manos aqui"
        dibujar_zona_captura(frame, roi_coords, color_zona, etiq_zona)

        dibujar_aviso_sin_objeto(frame, roi_coords, area_obj, n_manos)

        msg_show = msg_tmp if time.time() - msg_ts < 2.5 else ""
        dibujar_panel_superior(frame, modo, n_manos, objeto_actual, db)
        dibujar_estado_inferior(frame, objeto_actual, db, msg_show)
        dibujar_lista_objetos(frame, db, objeto_actual)
        dibujar_leyenda(frame)

        cv2.putText(frame, f"obj area: {area_obj}px",
                    (roi_coords[0], roi_coords[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1)

        cv2.imshow("Detector de Objetos con Manos", frame)

        # ── 6. Teclado ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Saliendo...")
            break
        elif key == ord('l'):
            modo = "APRENDIZAJE"
            print("[MODO] Aprendizaje")
        elif key == ord('r'):
            modo = "RECONOCIMIENTO"
            ultimo_resultado = ("", 0.0)
            print("[MODO] Reconocimiento")
        elif key == ord('n'):
            nombre = pedir_nombre_objeto()
            if nombre:
                db.setdefault(nombre, [])
                objeto_actual = nombre
                modo = "APRENDIZAJE"
                guardar_base_datos(db)
                msg_tmp = f"Objeto '{nombre}' registrado. Captura con [S]"
                msg_ts  = time.time()
                print(f"[NUEVO] '{nombre}'")
            else:
                print("[INFO] Cancelado.")
        elif key == ord('s'):
            if modo != "APRENDIZAJE":
                msg_tmp = "Cambia a Aprendizaje con [L]"
                msg_ts  = time.time()
            elif not objeto_actual:
                msg_tmp = "Primero registra un objeto con [N]"
                msg_ts  = time.time()
            elif area_obj < MIN_OBJ_AREA:
                msg_tmp = "Objeto no visible en zona. Acercalo mas o agrega luz."
                msg_ts  = time.time()
                print("[AVISO] Objeto demasiado pequeno.")
            else:
                ok = agregar_muestra(db, objeto_actual, roi_obj, mascara_obj)
                if ok:
                    n = len(db[objeto_actual])
                    flash_activo = True
                    flash_ts     = time.time()
                    msg_tmp = f"Muestra {n} capturada para '{objeto_actual}'"
                    msg_ts  = time.time()
                    print(f"[MUESTRA] '{objeto_actual}' -> {n}")
                    if n >= MIN_MUESTRAS:
                        print(f"  OK '{objeto_actual}' listo para reconocerse.")
                else:
                    msg_tmp = "Muestra invalida, intenta de nuevo."
                    msg_ts  = time.time()
        elif key == ord('c'):
            db = limpiar_base_datos()
            objeto_actual    = ""
            ultimo_resultado = ("", 0.0)
            msg_tmp = "Base de datos limpiada."
            msg_ts  = time.time()
            print("[LIMPIAR] BD eliminada.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Cerrado correctamente.")
    print(f"Objetos en BD: {list(db.keys())}")


if __name__ == "__main__":
    main()