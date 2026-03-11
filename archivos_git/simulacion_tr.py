# ============================================================
# SIMULADOR DE DATOS EN TIEMPO REAL
# Simula lecturas del ESP32 cada 5 segundos
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

print("=" * 50)
print("  SIMULADOR TIEMPO REAL - RAS TILAPIA")
print("  Ctrl+C para detener")
print("=" * 50)

# ── CONFIGURACIÓN ─────────────────────────────────────────────
ARCHIVO_SALIDA  = 'datos/lecturas_tiempo_real.csv'
INTERVALO_SEG   = 5      # segundos entre lecturas
MAX_REGISTROS   = 100    # máximo de registros a mantener
TANQUES         = [f'Tanque {i}' for i in range(1, 11)]

# ── ESTADO INICIAL ────────────────────────────────────────────
# Valores iniciales por tanque
estado_tanques = {
    tanque: {
        'ph':   np.random.uniform(6.8, 7.5),
        'temp': np.random.uniform(17.0, 19.0),
        'modo': 'normal',   # normal / bajando / subiendo
        'contador': 0
    }
    for tanque in TANQUES
}

# ── UMBRALES ──────────────────────────────────────────────────
PH_OPTIMO_MIN   = 6.5
PH_OPTIMO_MAX   = 8.5
PH_SUBOPT_MIN   = 6.05
PH_SUBOPT_MAX   = 9.0
TEMP_OPTIMO_MIN = 16.0
TEMP_OPTIMO_MAX = 20.0
TEMP_SUBOPT_MIN = 11.0
TEMP_SUBOPT_MAX = 27.0

def clasificar_estado(ph, temp):
    if temp < TEMP_SUBOPT_MIN or temp > TEMP_SUBOPT_MAX:
        return 2
    if ph < PH_SUBOPT_MIN or ph > PH_SUBOPT_MAX:
        return 2
    if temp < TEMP_OPTIMO_MIN or temp > TEMP_OPTIMO_MAX:
        return 1
    if ph < PH_OPTIMO_MIN or ph > PH_OPTIMO_MAX:
        return 1
    return 0

def simular_lectura(tanque, estado):
    """Simula una lectura realista con tendencias ocasionales"""
    # Cambiar modo aleatoriamente
    if estado['contador'] <= 0:
        rand = np.random.random()
        if rand < 0.02:      # 2% probabilidad de evento crítico
            estado['modo']     = 'critico'
            estado['contador'] = np.random.randint(3, 8)
        elif rand < 0.08:    # 6% probabilidad subóptimo
            estado['modo']     = 'suboptimo'
            estado['contador'] = np.random.randint(2, 5)
        else:
            estado['modo']     = 'normal'
            estado['contador'] = 0
    else:
        estado['contador'] -= 1

    # Simular valor según modo
    if estado['modo'] == 'critico':
        estado['ph']   += np.random.choice([-0.15, 0.15])
        estado['temp'] += np.random.choice([-0.8, 0.8])
    elif estado['modo'] == 'suboptimo':
        estado['ph']   += np.random.choice([-0.08, 0.08])
        estado['temp'] += np.random.choice([-0.4, 0.4])
    else:
        # Volver gradualmente al centro
        estado['ph']   += (7.2 - estado['ph']) * 0.1 + np.random.normal(0, 0.05)
        estado['temp'] += (18.0 - estado['temp']) * 0.1 + np.random.normal(0, 0.15)

    # Limitar valores
    estado['ph']   = np.clip(estado['ph'],   5.0, 10.0)
    estado['temp'] = np.clip(estado['temp'],  8.0, 32.0)

    return round(estado['ph'], 2), round(estado['temp'], 2)

# ── CREAR ARCHIVO SI NO EXISTE ────────────────────────────────
if not os.path.exists(ARCHIVO_SALIDA):
    pd.DataFrame(columns=[
        'Timestamp', 'Tanque', 'pH', 'Temperatura_C', 'Estado'
    ]).to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n✅ Archivo creado: {ARCHIVO_SALIDA}")

# ── LOOP PRINCIPAL ────────────────────────────────────────────
lectura_num = 0
while True:
    try:
        nuevas = []
        ahora  = datetime.now()

        for tanque in TANQUES:
            ph, temp = simular_lectura(tanque, estado_tanques[tanque])
            estado   = clasificar_estado(ph, temp)
            nuevas.append({
                'Timestamp':     ahora.strftime('%Y-%m-%d %H:%M:%S'),
                'Tanque':        tanque,
                'pH':            ph,
                'Temperatura_C': temp,
                'Estado':        estado
            })

        # Leer archivo existente
        df_actual = pd.read_csv(ARCHIVO_SALIDA)

        # Agregar nuevas lecturas
        df_nuevo = pd.concat(
            [df_actual, pd.DataFrame(nuevas)],
            ignore_index=True
        )

        # Mantener solo los últimos MAX_REGISTROS por tanque
        df_nuevo = (df_nuevo
                    .groupby('Tanque')
                    .tail(MAX_REGISTROS)
                    .reset_index(drop=True))

        df_nuevo.to_csv(ARCHIVO_SALIDA, index=False)

        lectura_num += 1
        estados_str = {0: '🟢', 1: '🟡', 2: '🔴'}
        resumen = ' '.join([
            f"T{i+1}:{estados_str[estado_tanques[t]['modo'] == 'critico' and 2 or 0]}"
            for i, t in enumerate(TANQUES[:5])
        ])
        print(f"[{ahora.strftime('%H:%M:%S')}] "
              f"Lectura #{lectura_num} guardada | "
              f"Tanques: {len(TANQUES)}")

        time.sleep(INTERVALO_SEG)

    except KeyboardInterrupt:
        print("\n\n⏹️  Simulador detenido.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(INTERVALO_SEG)