from problems.taylor_green2D import TaylorGreen2D
from problems.taylor_green3D import TaylorGreen3D
from problems.conservation_properties3D import ConservationProperties3D
from solvers.dfNS_palha import compute_sol
import matplotlib.pyplot as plt

from math import pi
d = 3 # int(input("Spatial dimension ? "))

if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    deg = 2
    n_t = 5
    Delta_t = 1/100
    t_f = n_t * Delta_t
    options = {"n_el": 3, "t_fin": t_f, "n_t": n_t}
    if d == 2:
        taylorgreen = TaylorGreen2D(options)
    else:
        taylorgreen = TaylorGreen3D(options)
    tvec_int, tvec_stag, H_pr, H_dl, H_ex, E_pr, E_dl, E_ex,\
    wP_pr, wP_dl, wP_ex,pP_pr, pP_dl, pP_ex, div_pr,div_dl = compute_sol(taylorgreen, deg, n_t, t_f)
    # cons_prop = ConservationProperties3D(options)
    # tvec_int, tvec_stag, H_pr, H_dl, H_ex, wP_pr, wP_dl, wP_ex = compute_sol(cons_prop, deg, n_t, t_f)
    plt.subplot(2,3,1)
    plt.plot(tvec_stag, H_pr, 'b', label="H primal")
    plt.plot(tvec_int, H_dl, 'r', label="H dual")
    plt.plot(tvec_stag, H_ex, 'g', label="H exact")
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(tvec_stag, E_pr, 'b', label="E primal")
    plt.plot(tvec_int, E_dl, 'r', label="E dual")
    plt.plot(tvec_stag, E_ex, 'g', label="E exact")
    plt.legend()

    plt.subplot(2,3,3)
    plt.plot(tvec_stag, wP_pr, 'b', label="w at P primal")
    plt.plot(tvec_int, wP_dl, 'r', label="w at P dual")
    plt.plot(tvec_stag, wP_ex, 'g', label="w at P exact")
    plt.legend()

    plt.subplot(2,3,4)
    plt.plot(tvec_stag, pP_pr, 'b', label="p at P primal")
    plt.plot(tvec_int, pP_dl, 'r', label="p at P dual")
    plt.plot(tvec_stag, pP_ex, 'g', label="p at P exact")
    plt.legend()

    plt.subplot(2,3,5)
    plt.plot(tvec_stag[1:], div_pr[1:], 'b', label="div(u) primal")
    plt.plot(tvec_int[1:], div_dl[1:], 'r', label="div(u) dual")
    plt.legend()

    plt.show()
