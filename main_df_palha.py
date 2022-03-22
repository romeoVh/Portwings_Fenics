from problems.taylor_green2D import TaylorGreen2D
from problems.taylor_green3D import TaylorGreen3D
from problems.conservation_properties3D import ConservationProperties3D
from solvers.dfNS_palha import compute_sol
import matplotlib.pyplot as plt

from math import pi
d = 2 # int(input("Spatial dimension ? "))

if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    deg = 1
    n_t =1000
    Delta_t = 1/1000
    t_f = n_t * Delta_t
    options = {"n_el": 16, "t_fin": t_f, "n_t": n_t}
    if d == 2:
        taylorgreen = TaylorGreen2D(options)
    else:
        taylorgreen = TaylorGreen3D(options)
    tvec_int, tvec_stag, H_pr, H_dl, H_ex, wP_pr, wP_dl, wP_ex = compute_sol(taylorgreen, deg, n_t, t_f)
    # cons_prop = ConservationProperties3D(options)
    # tvec_int, tvec_stag, H_pr, H_dl, H_ex, wP_pr, wP_dl, wP_ex = compute_sol(cons_prop, deg, n_t, t_f)
    plt.figure()
    plt.plot(tvec_stag, H_pr, 'b', label="H primal")
    plt.plot(tvec_int, H_dl, 'r', label="H dual")
    plt.plot(tvec_int, H_ex, 'g', label="H exact")
    plt.legend()


    plt.figure()
    plt.plot(tvec_stag, wP_pr, 'b', label="w at P primal")
    plt.plot(tvec_int, wP_dl, 'r', label="w at P dual")
    plt.plot(tvec_int, wP_ex, 'g', label="w at P exact")
    plt.legend()
    plt.show()
