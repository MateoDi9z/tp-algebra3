import matplotlib.pyplot as plt


def graficar_listas(res: list, t: list, title: str):
    """
        Graficar par de listas
        :param title: Titulo del grafio
        :param res lista de resoluciones
        :param t lista de tiempo
    """

    # Crear gráfico
    plt.plot(res, t, marker='o', linestyle='-', color='blue', label='Resolución vs n')
    plt.xlabel('Resoluciones')
    plt.ylabel('Tiempo (ms)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()

def graficar_errores(res: list, errors: list, title: str):
    # Crear gráfico
    plt.plot(res, errors, marker='o', linestyle='-', color='blue', label='Resolución vs n')
    plt.xlabel('Resoluciones')
    plt.ylabel('Error')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()

"""
import matplotlib.pyplot as plt


def graficar_listas(res: list, t: list, errors: list, title: str) -> None:
        Graficar par de listas
        :param res lista de resoluciones
        :param t lista de tiempos
        :param errors lista de errores
        :param title titulo

    # Crear gráfico
    scatter = plt.scatter(res, t, c=errors, cmap='viridis', s=100, edgecolors='k')
    plt.plot(res, t, marker='o', linestyle='-', alpha=0.5, color='gray', label='Resolución vs n')

    # Añadir barra de color
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error')

    plt.xlabel('Resoluciones')
    plt.ylabel('Tiempo (ms)')
    plt.title(title)
    plt.grid(True)

    plt.legend()
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()

"""