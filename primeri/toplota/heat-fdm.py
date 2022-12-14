import numpy as np
import matplotlib.pyplot as plt

def heatFTCS(nt=10, nx=20, alpha=0.3, L=1, tmax=0.1):

    h = L / (nx - 1)
    k = tmax / (nt - 1)
    r = alpha * k / h**2

    x = np.linspace(0, L, nx)
    t = np.linspace(0, tmax, nt)
    U = np.zeros((nx, nt))

    U[:, 0] = np.sin(np.pi * x / L)

    for m in range(1, nt):
        for i in range(1, nx-1):
            U[i, m] = r * U[i - 1, m - 1] + (1-2*r) * U[i, m-1] + r * U[i+1, m-1]

    ue = np.sin(np.pi * x / L) * \
        np.exp(-t[nt - 1] * alpha * (np.pi / L) * (np.pi / L))

    _, ax = plt.subplots()
    ax.plot(x, U[:, nt - 1], 'o--', label='FTCS')
    ax.plot(x, ue, '-', label='Exact')
    plt.xlabel("x")
    plt.ylabel("u")
    ax.legend()
    plt.show()

heatFTCS()
