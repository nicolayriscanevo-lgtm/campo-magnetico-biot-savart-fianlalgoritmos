import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit

mu0 = 4e-7 * np.pi  # Permeabilidad del vacío (T·m/A)

ENTORNO = {
    "pi": np.pi, "e": np.e,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
}


# ----------------------------------------------------------------------
# Utilidades de entrada
# ----------------------------------------------------------------------
def evaluar(expr, **variables):
    """Evalúa una expresión matemática con funciones de numpy (ej. '4*pi', 'cos(t)')."""
    return eval(expr, {"__builtins__": {}}, {**ENTORNO, **variables})


def leer_float(mensaje, positivo=True):
    while True:
        try:
            valor = float(evaluar(input(mensaje).strip()))
            if positivo and valor <= 0:
                print("Error: el valor debe ser mayor que 0.")
                continue
            return valor
        except Exception:
            print("Error: debe ingresar un número válido (puede usar pi, ej. 2*pi).")


def leer_entero(mensaje, minimo=1):
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"Error: el valor debe ser al menos {minimo}.")
                continue
            return valor
        except ValueError:
            print("Error: debe ingresar un número entero.")


def leer_intervalo():
    while True:
        try:
            texto = input("Ingrese el intervalo de t (ej. 0, 4*pi): ")
            t_min, t_max = (float(evaluar(p.strip())) for p in texto.split(","))
            if t_max <= t_min:
                print("Error: el valor máximo debe ser mayor que el mínimo.")
                continue
            return t_min, t_max
        except Exception:
            print("Error: formato inválido. Use 'min, max' (ej. 0, 4*pi).")


def leer_expresion(mensaje, t_prueba):
    """Pide una expresión en función de t y verifica que se pueda evaluar."""
    while True:
        expr = input(mensaje).strip()
        try:
            evaluar(expr, t=t_prueba)
            return expr
        except Exception as err:
            print(f"Error: expresión inválida ({err}). Ejemplo: 2*cos(t)")


# ----------------------------------------------------------------------
# Curvas paramétricas y discretización
# ----------------------------------------------------------------------
def curva_helice(t, R, paso):
    """Hélice / solenoide horizontal (eje a lo largo de x): paso = avance en x por vuelta (m)."""
    x = paso * t / (2 * np.pi)
    y = R * np.cos(t)
    z = R * np.sin(t)
    return x, y, z


def curva_espiral(t, R0, a):
    """Espiral de Arquímedes plana (z = 0): radio = R0 + a*t."""
    rho = R0 + a * t
    return rho * np.cos(t), rho * np.sin(t), np.zeros_like(t)


def curva_nudo_trebol(t, R):
    """Nudo de trébol 3D (sin solución analítica del campo)."""
    x = R * (np.sin(t) + 2 * np.sin(2 * t))
    y = R * (np.cos(t) - 2 * np.cos(2 * t))
    z = R * (-np.sin(3 * t))
    return x, y, z


def curva_arbitraria(t, ex, ey, ez):
    """Curva dada por tres expresiones en t escritas por el usuario."""
    unos = np.ones_like(t)
    return (evaluar(ex, t=t) * unos,
            evaluar(ey, t=t) * unos,
            evaluar(ez, t=t) * unos)


def discretizar_curva(x_c, y_c, z_c):
    """Elementos de corriente dl = (dx, dy, dz) a lo largo de la curva."""
    return np.gradient(x_c), np.gradient(y_c), np.gradient(z_c)


# ----------------------------------------------------------------------
# Núcleo numérico: Ley de Biot-Savart (Versión Serial)
# ----------------------------------------------------------------------
@njit(cache=True)
def _campo_serial_kernel(z, x, y, x_c, y_c, z_c, dx, dy, dz, k):
    """Evalúa B = (Bx, By, Bz) secuencialmente recorriendo toda la malla 2D."""
    ny, nx = y.shape[0], x.shape[0]
    Bx = np.zeros((ny, nx), dtype=np.float64)
    By = np.zeros((ny, nx), dtype=np.float64)
    Bz = np.zeros((ny, nx), dtype=np.float64)
    m_total = x_c.shape[0]

    for i in range(ny):
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


def calcular_campo_serial(z, x, y, curva, corriente=1.0):
    """Interfaz limpia para calcular el campo magnético en versión serial."""
    x_c, y_c, z_c, dx, dy, dz = (np.ascontiguousarray(a, dtype=np.float64) for a in curva)
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    k = mu0 * corriente / (4.0 * np.pi)
    return _campo_serial_kernel(float(z), x_arr, y_arr, x_c, y_c, z_c, dx, dy, dz, k)


# ----------------------------------------------------------------------
# Menú y visualización interactiva
# ----------------------------------------------------------------------
def pedir_opcion():
    while True:
        print("\n[SERIAL] Gráfico de campo magnético en el plano XY")
        print("1. Campo magnético de una hélice (solenoide horizontal)")
        print("2. Campo magnético de una espiral")
        print("3. Campo magnético de un nudo de trébol")
        print("4. Campo magnético de una curva paramétrica arbitraria")
        print("5. Salir")
        try:
            opcion = int(input("Seleccione una opción: "))
            if 1 <= opcion <= 5:
                return opcion
            print("Error: debe ingresar un número entre 1 y 5.")
        except ValueError:
            print("Error: debe ingresar un número entero.")


def construir_curva(opcion):
    if opcion == 1:
        r = leer_float("Ingrese el radio del solenoide (m): ")
        l_cm = leer_float("Ingrese la longitud del solenoide (cm): ")
        n = leer_entero("Ingrese el número de vueltas del solenoide: ")
        l = l_cm / 100.0
        paso = l / n
        t = np.linspace(0, 2 * np.pi * n, 200 * n)
        x_c, y_c, z_c = curva_helice(t, r, paso)
        return x_c - l / 2, y_c, z_c

    if opcion == 2:
        r0 = leer_float("Ingrese el radio inicial de la espiral (m): ", positivo=False)
        a = leer_float("Ingrese el crecimiento del radio por radián (m/rad): ")
        t_min, t_max = leer_intervalo()
        t = np.linspace(t_min, t_max, 2000)
        return curva_espiral(t, r0, a)

    if opcion == 3:
        r = leer_float("Ingrese el factor de escala del nudo de trébol (m): ")
        t_min, t_max = leer_intervalo()
        t = np.linspace(t_min, t_max, 2000)
        return curva_nudo_trebol(t, r)

    t_min, t_max = leer_intervalo()
    print("Escriba x(t), y(t), z(t) usando t, pi, sin, cos, exp, sqrt, etc.")
    t = np.linspace(t_min, t_max, 2000)
    ex = leer_expresion("x(t) = ", t)
    ey = leer_expresion("y(t) = ", t)
    ez = leer_expresion("z(t) = ", t)
    return curva_arbitraria(t, ex, ey, ez)


def graficar(x_c, y_c, z_c, corriente):
    curva = (x_c, y_c, z_c, *discretizar_curva(x_c, y_c, z_c))
    L = max(1.3 * max(np.abs(x_c).max(), np.abs(y_c).max()), 0.5)
    z_min, z_max = z_c.min(), z_c.max()
    margen = max(0.5 * L, 0.25 * (z_max - z_min))
    z_ini = 0.5 * (z_min + z_max)

    x = np.linspace(-L, L, 150)
    y = np.linspace(-L, L, 150)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2, right=0.85)
    cax = fig.add_axes([0.88, 0.25, 0.02, 0.6])

    def dibujar(z):
        ax.clear()
        cax.clear()
        Bx, By, Bz = calcular_campo_serial(z, x, y, curva, corriente)
        mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        strm = ax.streamplot(x, y, Bx, By, color=mag * 1e6, cmap="viridis", density=1.5)
        fig.colorbar(strm.lines, cax=cax, label="|B| (µT)")
        ax.plot(x_c, y_c, "r--", lw=1, alpha=0.7, label="Proyección de la curva")
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Campo magnético [Serial] en z = {z:.2f} m (I = {corriente:g} A)")
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    ax_z = fig.add_axes([0.2, 0.07, 0.6, 0.03])
    slider = Slider(ax_z, "z (m)", z_min - margen, z_max + margen, valinit=z_ini)
    slider.on_changed(dibujar)

    dibujar(z_ini)
    plt.show()


def inicio():
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