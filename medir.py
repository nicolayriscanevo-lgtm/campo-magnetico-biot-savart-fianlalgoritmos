import time
import platform
import csv #para la tablas de datos
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #para la ley de amdahl
import numba

#importamos funciones de serial y paralelo para poder medir los tiempos de ejecucion y comparar resultados

from serial import (
    curva_helice,
    curva_nudo_trebol,
    discretizar_curva,
    calcular_campo_serial,
)



from paralelo import calcular_campo_paralelo


def obtener_info_hardware():
    """Recopila información básica del sistema y núcleos disponibles."""
    return {
        "procesador": platform.processor() or "CPU x86_64",
        "sistema": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numba": numba.__version__,
        "numpy": np.__version__,
        "hilos_logicos": numba.config.NUMBA_DEFAULT_NUM_THREADS,
    }


def verificar_resultados(tamanos):
    """
    Criterio 4: Verifica cuantitativamente que serial y paralelo producen
    el mismo resultado dentro de tolerancia de punto flotante.
    """
    print("=" * 70)
    print("1. VERIFICACIÓN NUMÉRICA (SERIAL VS PARALELO)")
    print("=" * 70)

    # Usamos un nudo de trébol 3D como geometría de prueba para la verificación
    t = np.linspace(0, 2 * np.pi, 1000) # de 0 a 2 pi crea 1000 puntos equiespaciados de esta linea de trevol parametrizada 
    x_c, y_c, z_c = curva_nudo_trebol(t, R=0.5)
    curva = (x_c, y_c, z_c, *discretizar_curva(x_c, y_c, z_c))
    corriente = 2.5
    z_plano = 0.15

    # Warm-up previo para compilar JIT
    x_w = np.linspace(-1.5, 1.5, 10) #pequeña malla de 10x10 para que compile rapido y no tarde mucho
    calcular_campo_serial(z_plano, x_w, x_w, curva, corriente)
    calcular_campo_paralelo(z_plano, x_w, x_w, curva, corriente, n_hilos=2)

    todo_ok = True
    for nombre, n_malla, n_seg in tamanos:
        t_seg = np.linspace(0, 2 * np.pi, n_seg)
        xc, yc, zc = curva_nudo_trebol(t_seg, R=0.5)
        curva_test = (xc, yc, zc, *discretizar_curva(xc, yc, zc))

        x = np.linspace(-1.5, 1.5, n_malla)
        y = np.linspace(-1.5, 1.5, n_malla)

        Bx_s, By_s, Bz_s = calcular_campo_serial(z_plano, x, y, curva_test, corriente)
        Bx_p, By_p, Bz_p = calcular_campo_paralelo(
            z_plano, x, y, curva_test, corriente, n_hilos=4
        )

        err_max = max(
            np.max(np.abs(Bx_s - Bx_p)),
            np.max(np.abs(By_s - By_p)),
            np.max(np.abs(Bz_s - Bz_p)),
        )
        B_ref = np.max(np.sqrt(Bx_s**2 + By_s**2 + Bz_s**2))
        err_rel = err_max / B_ref if B_ref > 0 else 0.0

        coincide = (
            np.allclose(Bx_s, Bx_p, rtol=1e-12, atol=1e-15)
            and np.allclose(By_s, By_p, rtol=1e-12, atol=1e-15)
            and np.allclose(Bz_s, Bz_p, rtol=1e-12, atol=1e-15)
        )
        todo_ok = todo_ok and coincide

        print(
            f"[{nombre}] Malla {n_malla}x{n_malla}, M={n_seg} -> "
            f"Error abs máx: {err_max:.3e} T | Error rel: {err_rel:.3e} | "
            f"allclose: {'OK' if coincide else 'FALLO'}"
        )

    print(
        "Nota: Diferencias del orden ~1e-16 a 1e-19 T provienen del redondeo "
        "de punto flotante IEEE-754 al cambiar el orden de ejecución entre hilos.\n"
    )
    if not todo_ok:
        raise AssertionError("¡La verificación numérica entre serial y paralelo falló!")








def modelo_amdahl(p, f):
    """Ley de Amdahl: S(p) = 1 / ((1 - f) + f / p)."""
    return 1.0 / ((1.0 - f) + (f / p))


def ejecutar_benchmark(tamanos, lista_hilos, repeticiones=7, archivo_csv="resultados.csv"):
    """
    Criterio 5: Mide tiempos con time.perf_counter, excluye compilación JIT,
    repite >= 5 veces, reporta mediana y dispersión (IQR y std), y exporta CSV.
    """
    print("=" * 70)
    print("2. BENCHMARK DE DESEMPEÑO (TIEMPO, SPEEDUP Y EFICIENCIA)")
    print("=" * 70)

    resultados = []
    z_plano = 0.05
    corriente = 1.0

    for nombre_tam, n_malla, n_seg in tamanos:
        print(f"\n--- Tamaño: {nombre_tam} (Malla {n_malla}x{n_malla} = {n_malla**2:,} pts, M = {n_seg:,} segmentos) ---")
        t = np.linspace(0, 2 * np.pi * 10, n_seg)
        xc, yc, zc = curva_helice(t, R=0.1, paso=0.05)
        curva = (xc, yc, zc, *discretizar_curva(xc, yc, zc))
        x = np.linspace(-0.8, 0.8, n_malla)
        y = np.linspace(-0.8, 0.8, n_malla)

        # 1) Medición Serial (1 llamada warm-up + repeticiones cronometradas)
        _ = calcular_campo_serial(z_plano, x, y, curva, corriente)
        tiempos_serial = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            _ = calcular_campo_serial(z_plano, x, y, curva, corriente)
            t1 = time.perf_counter()
            tiempos_serial.append(t1 - t0)

        t1_mediana = float(np.median(tiempos_serial))
        t1_std = float(np.std(tiempos_serial, ddof=1))
        t1_iqr = float(np.percentile(tiempos_serial, 75) - np.percentile(tiempos_serial, 25))
        print(f"Serial (T1)      -> Mediana: {t1_mediana*1000:8.2f} ms | Std: {t1_std*1000:6.2f} ms")

        resultados.append({
            "tamano": nombre_tam,
            "n_malla": n_malla,
            "n_segmentos": n_seg,
            "operaciones": n_malla * n_malla * n_seg,
            "version": "serial",
            "hilos": 1,
            "mediana_s": t1_mediana,
            "std_s": t1_std,
            "iqr_s": t1_iqr,
            "speedup": 1.0,
            "eficiencia": 1.0,
        })

        # 2) Medición Paralela variando número de hilos p
        for p in lista_hilos:
            # Warm-up para asegurar pool de hilos activo y JIT listo
            _ = calcular_campo_paralelo(z_plano, x, y, curva, corriente, n_hilos=p)
            tiempos_p = []
            for _ in range(repeticiones):
                t0 = time.perf_counter()
                _ = calcular_campo_paralelo(z_plano, x, y, curva, corriente, n_hilos=p)
                t1 = time.perf_counter()
                tiempos_p.append(t1 - t0)

            tp_mediana = float(np.median(tiempos_p))
            tp_std = float(np.std(tiempos_p, ddof=1))
            tp_iqr = float(np.percentile(tiempos_p, 75) - np.percentile(tiempos_p, 25))
            speedup = t1_mediana / tp_mediana
            eficiencia = speedup / p

            print(
                f"Paralelo (p={p:2d})  -> Mediana: {tp_mediana*1000:8.2f} ms | "
                f"Std: {tp_std*1000:6.2f} ms | Speedup: {speedup:5.2f}x | "
                f"Eficiencia: {eficiencia*100:5.1f}%"
            )

            resultados.append({
                "tamano": nombre_tam,
                "n_malla": n_malla,
                "n_segmentos": n_seg,
                "operaciones": n_malla * n_malla * n_seg,
                "version": "paralelo",
                "hilos": p,
                "mediana_s": tp_mediana,
                "std_s": tp_std,
                "iqr_s": tp_iqr,
                "speedup": speedup,
                "eficiencia": eficiencia,
            })

    # Guardar resultados en CSV
    campos = list(resultados[0].keys())
    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nResultados exportados exitosamente a '{archivo_csv}'.")
    return resultados


def graficar_desempeno(resultados, nucleos_fisicos=6):
    """
    Genera gráficas de alta resolución (300 dpi) para informe y póster:
    1) Tiempo de ejecución vs hilos
    2) Speedup vs hilos con línea ideal y ajuste teórico de Ley de Amdahl
    3) Eficiencia paralela vs hilos
    """
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    tamanos_unicos = list(dict.fromkeys(r["tamano"] for r in resultados))
    colores = ["#1f77b4", "#d62728", "#2ca02c"]
    marcadores = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax_t, ax_s, ax_e = axes

    max_hilos = max(r["hilos"] for r in resultados)
    p_continuo = np.linspace(1, max_hilos, 200)

    print("\n" + "=" * 70)
    print("3. AJUSTE DE LEY DE AMDAHL (ESTIMACIÓN DE FRACCIÓN PARALELIZABLE f)")
    print("=" * 70)

    for idx, tam in enumerate(tamanos_unicos):
        filas = [r for r in resultados if r["tamano"] == tam and r["version"] == "paralelo"]
        p_vals = np.array([r["hilos"] for r in filas], dtype=float)
        t_vals = np.array([r["mediana_s"] * 1000.0 for r in filas])  # en ms
        t_err = np.array([r["std_s"] * 1000.0 for r in filas])
        s_vals = np.array([r["speedup"] for r in filas])
        e_vals = np.array([r["eficiencia"] * 100.0 for r in filas])

        n_malla = filas[0]["n_malla"]
        n_seg = filas[0]["n_segmentos"]
        etiqueta = f"{tam} ({n_malla}x{n_malla}, M={n_seg})"
        c = colores[idx % len(colores)]
        m = marcadores[idx % len(marcadores)]

        # Ajuste de Amdahl solo hasta núcleos físicos (donde no hay SMT/HyperThreading)
        mask_fisicos = p_vals <= nucleos_fisicos
        popt, _ = curve_fit(
            modelo_amdahl,
            p_vals[mask_fisicos],
            s_vals[mask_fisicos],
            bounds=(0.0, 1.0),
        )
        f_est = popt[0]
        s_max_teorico = 1.0 / (1.0 - f_est) if f_est < 1.0 else np.inf
        print(
            f"[{tam}] Fracción paralelizable estimada f = {f_est*100:.2f}% "
            f"(fracción serial 1-f = {(1-f_est)*100:.2f}%) -> "
            f"Speedup máximo teórico (p->inf): {s_max_teorico:.2f}x"
        )

        # Panel 1: Tiempo (ms)
        ax_t.errorbar(
            p_vals, t_vals, yerr=t_err, marker=m, color=c, lw=2, capsize=4, label=etiqueta
        )

        # Panel 2: Speedup + Amdahl
        ax_s.plot(p_vals, s_vals, marker=m, color=c, lw=2, label=f"{etiqueta} (Medido)")
        ax_s.plot(
            p_continuo,
            modelo_amdahl(p_continuo, f_est),
            linestyle="--",
            color=c,
            alpha=0.75,
            lw=1.5,
            label=f"Amdahl {tam} (f={f_est:.3f})",
        )

        # Panel 3: Eficiencia (%)
        ax_e.plot(p_vals, e_vals, marker=m, color=c, lw=2, label=etiqueta)

    # Líneas de referencia (Ideal y límite de núcleos físicos)
    ax_s.plot(p_continuo, p_continuo, "k:", lw=1.8, label="Speedup Ideal (S = p)")
    ax_e.axhline(100.0, color="k", linestyle=":", lw=1.8, label="Eficiencia Ideal (100%)")

    for ax in axes:
        ax.axvline(
            nucleos_fisicos,
            color="gray",
            linestyle="-.",
            alpha=0.6,
            label=f"Núcleos físicos ({nucleos_fisicos})" if ax == ax_t else None,
        )
        ax.set_xlabel("Número de hilos (p)")
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_t.set_ylabel("Tiempo mediano (ms)")
    ax_t.set_title("Tiempo de Ejecución vs. Hilos")
    ax_t.legend()

    ax_s.set_ylabel("Speedup (S = T1 / Tp)")
    ax_s.set_title("Escalabilidad y Ajuste de Ley de Amdahl")
    ax_s.legend(fontsize=8.5)

    ax_e.set_ylabel("Eficiencia Paralela E = S / p (%)")
    ax_e.set_title("Eficiencia vs. Número de Hilos")
    ax_e.set_ylim(0, 115)
    ax_e.legend()

    plt.tight_layout()
    plt.savefig("desempeno_speedup_eficiencia.png", dpi=300)
    plt.close(fig)
    print("Gráfica guardada: 'desempeno_speedup_eficiencia.png'")


def graficar_fenomeno_fisico():
    """
    Genera una figura compuesta de alta resolución (300 dpi) con el campo magnético B
    en el plano XY para tres geometrías representativas (Solenoide, Espiral y Nudo de Trébol),
    lista para el informe y el póster científico.
    """
    from serial import curva_espiral

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    corriente = 5.0

    # Caso 1: Solenoide finito (hélice horizontal)
    t1 = np.linspace(0, 2 * np.pi * 12, 2400)
    x1, y1, z1 = curva_helice(t1, R=0.15, paso=0.04)
    x1 -= 0.24
    c1 = (x1, y1, z1, *discretizar_curva(x1, y1, z1))

    # Caso 2: Espiral de Arquímedes plana
    t2 = np.linspace(0, 6 * np.pi, 2000)
    x2, y2, z2 = curva_espiral(t2, R0=0.02, a=0.015)
    c2 = (x2, y2, z2, *discretizar_curva(x2, y2, z2))

    # Caso 3: Nudo de trébol 3D
    t3 = np.linspace(0, 2 * np.pi, 2000)
    x3, y3, z3 = curva_nudo_trebol(t3, R=0.12)
    c3 = (x3, y3, z3, *discretizar_curva(x3, y3, z3))

    casos = [
        (axes[0], c1, "Solenoide Horizontal (12 vueltas)", 0.45, 0.0),
        (axes[1], c2, "Espiral de Arquímedes (3 vueltas)", 0.45, 0.05),
        (axes[2], c3, "Nudo de Trébol 3D (z = 0.05 m)", 0.45, 0.05),
    ]

    for ax, curva, titulo, L, z_eval in casos:
        x = np.linspace(-L, L, 250)
        y = np.linspace(-L, L, 250)
        Bx, By, Bz = calcular_campo_paralelo(z_eval, x, y, curva, corriente)
        mag_uT = np.sqrt(Bx**2 + By**2 + Bz**2) * 1e6

        strm = ax.streamplot(
            x, y, Bx, By, color=mag_uT, cmap="viridis", density=1.4, linewidth=1.1
        )
        cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|B| (µT)")

        ax.plot(curva[0], curva[1], "r--", lw=1.2, alpha=0.8, label="Proyección XY del cable")
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(titulo)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("simulacion_campo_magnetico.png", dpi=300)
    plt.close(fig)
    print("Gráfica guardada: 'simulacion_campo_magnetico.png'")


def inicio():
    info = obtener_info_hardware()
    print("=" * 70)
    print("ENTORNO DE EJECUCIÓN Y HARDWARE")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    # Tres tamaños de problema (N_malla x N_malla puntos y M segmentos)
    tamanos_verificacion = [
        ("Pequeño", 80, 500),
        ("Mediano", 200, 1500),
        ("Grande", 400, 3000),
    ]

    tamanos_benchmark = [
        ("Pequeño", 250, 1500),    # 62,500 * 1,500 = 9.37e7 interacciones
        ("Mediano", 500, 3000),    # 250,000 * 3,000 = 7.50e8 interacciones
        ("Grande", 800, 4000),     # 640,000 * 4,000 = 2.56e9 interacciones
    ]

    # Hilos a evaluar: 1 hasta los núcleos físicos (6) + hilos lógicos SMT (8, 12)
    hilos_max = info["hilos_logicos"]
    lista_hilos = [p for p in [1, 2, 4, 6, 8, 12] if p <= hilos_max]
    if hilos_max not in lista_hilos:
        lista_hilos.append(hilos_max)

    verificar_resultados(tamanos_verificacion)
    resultados = ejecutar_benchmark(
        tamanos=tamanos_benchmark,
        lista_hilos=lista_hilos,
        repeticiones=7,
        archivo_csv="resultados.csv",
    )
    graficar_desempeno(resultados, nucleos_fisicos=6)
    graficar_fenomeno_fisico()



inicio()