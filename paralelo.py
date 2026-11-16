import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numba
from numba import njit, prange

# Importamos utilidades de entrada y geometrías desde serial para evitar duplicidad
from serial import (
    mu0,
    evaluar,
    pedir_opcion,
    construir_curva,
    discretizar_curva,
)


# ----------------------------------------------------------------------
# Núcleo numérico: Ley de Biot-Savart (Versión Paralela con prange)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _campo_paralelo_kernel(z, x, y, x_c, y_c, z_c, dx, dy, dz, k):
    """Reparte las filas (eje y) de la malla 2D entre los hilos disponibles mediante prange."""
    ny, nx = y.shape[0], x.shape[0]
    Bx = np.zeros((ny, nx), dtype=np.float64)
    By = np.zeros((ny, nx), dtype=np.float64)
    Bz = np.zeros((ny, nx), dtype=np.float64)
    m_total = x_c.shape[0]

    for i in prange(ny):
        yi = y[i]
        for j in range(nx):
            xj = x[j]
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for m in range(m_total):
                rx = xj - x_c[m]
                ry = yi - y_c[m]
                rz = z - z_c[m]
                r2 = rx * rx + ry * ry + rz * rz + 1e-12
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                sx += (dy[m] * rz - dz[m] * ry) * inv_r3
                sy += (dz[m] * rx - dx[m] * rz) * inv_r3
                sz += (dx[m] * ry - dy[m] * rx) * inv_r3
            Bx[i, j] = k * sx
            By[i, j] = k * sy
            Bz[i, j] = k * sz
    return Bx, By, Bz


def calcular_campo_paralelo(z, x, y, curva, corriente=1.0, n_hilos=None):
    """
    Calcula el campo magnético en paralelo usando Numba prange.
    Si se especifica n_hilos, configura el pool de hilos de Numba antes de ejecutar.
    """
    if n_hilos is not None:
        numba.set_num_threads(int(n_hilos))

    x_c, y_c, z_c, dx, dy, dz = (np.ascontiguousarray(a, dtype=np.float64) for a in curva)
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    k = mu0 * corriente / (4.0 * np.pi)
    return _campo_paralelo_kernel(float(z), x_arr, y_arr, x_c, y_c, z_c, dx, dy, dz, k)


# ----------------------------------------------------------------------
# Visualización interactiva
# ----------------------------------------------------------------------
def graficar(x_c, y_c, z_c, corriente):
    curva = (x_c, y_c, z_c, *discretizar_curva(x_c, y_c, z_c))
    L = max(1.3 * max(np.abs(x_c).max(), np.abs(y_c).max()), 0.5)
    z_min, z_max = z_c.min(), z_c.max()
    margen = max(0.5 * L, 0.25 * (z_max - z_min))
    z_ini = 0.5 * (z_min + z_max)

    x = np.linspace(-L, L, 400)
    y = np.linspace(-L, L, 400)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2, right=0.85)
    cax = fig.add_axes([0.88, 0.25, 0.02, 0.6])

    def dibujar(z):
        ax.clear()
        cax.clear()
        Bx, By, Bz = calcular_campo_paralelo(z, x, y, curva, corriente)
        mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        strm = ax.streamplot(x, y, Bx, By, color=mag * 1e6, cmap="viridis", density=1.5)
        fig.colorbar(strm.lines, cax=cax, label="|B| (µT)")
        ax.plot(x_c, y_c, "r--", lw=1, alpha=0.7, label="Proyección de la curva")
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"Campo magnético [Paralelo: {numba.get_num_threads()} hilos] "
            f"en z = {z:.2f} m (I = {corriente:g} A)"
        )
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    ax_z = fig.add_axes([0.2, 0.07, 0.6, 0.03])
    slider = Slider(ax_z, "z (m)", z_min - margen, z_max + margen, valinit=z_ini)
    slider.on_changed(dibujar)

    dibujar(z_ini)
    plt.show()


def inicio():
    print(f"Numba configurado con {numba.get_num_threads()} hilos disponibles.")
    while True:
        opcion = pedir_opcion()
        if opcion == 5:
            print("Hasta luego.")
            break
        entrada = input("Ingrese la corriente en amperios (A) [1 por defecto]: ").strip()
        try:
            corriente = float(evaluar(entrada)) if entrada else 1.0
        except Exception:
            corriente = 1.0
            print("Valor inválido. Se asignará un valor por defecto de 1 A.")
        x_c, y_c, z_c = construir_curva(opcion)
        graficar(x_c, y_c, z_c, corriente)


if __name__ == "__main__":
    inicio()