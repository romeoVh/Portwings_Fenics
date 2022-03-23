from problems.periodicEuler2D_exact import ExactEuler2D
from problems.conservation_properties3D import ConservationProperties3D
from solvers.dfEuler_palha import compute_sol
import matplotlib.pyplot as plt
import numpy as np
from math import pi
d = 2 # int(input("Spatial dimension ? "))

if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    deg = 3
    n_t = 5
    Delta_t = 1/10
    t_f = n_t * Delta_t
    options = {"n_el": 3, "t_fin": t_f, "n_t": n_t}

    # problem = ExactEuler2D(options)
    problem = ConservationProperties3D(options)
    results = compute_sol(problem, deg, n_t, t_f)

    tvec_int = results["tspan_int"]
    tvec_stag = results["tspan_stag"]

    H_ex = results["energy_ex"]
    H_pr = results["energy_pr"]
    H_dl = results["energy_dl"]

    E_ex = results["enstrophy_ex"]
    E_pr = results["enstrophy_pr"]
    E_dl = results["enstrophy_dl"]

    Hel_ex = results["helicity_ex"]
    Hel_pr = results["helicity_pr"]
    Hel_dl = results["helicity_dl"]

    uP_ex = results["uP_ex"]
    uP_pr = results["uP_pr"]
    uP_dl = results["uP_dl"]

    wP_ex = results["wP_ex"]
    wP_pr = results["wP_pr"]
    wP_dl = results["wP_dl"]

    pdynP_ex = results["pdynP_ex"]
    pdynP_pr = results["pdynP_pr"]
    pdynP_dl = results["pdynP_dl"]

    divu_pr_L2 = results["divu_pr_L2"]
    divu_dl_L2 = results["divu_dl_L2"]

    plt.figure()
    plt.plot(tvec_int[1:], H_pr[1:], 'b', label="H primal")
    plt.plot(tvec_int[1:], H_dl[1:], 'r', label="H dual")
    if problem.exact:
        plt.plot(tvec_int, H_ex, 'g', label="H exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int[1:], Hel_pr[1:], 'b', label="E primal")
    plt.plot(tvec_int[1:], Hel_dl[1:], 'r', label="E dual")
    if problem.exact:
        plt.plot(tvec_int, Hel_ex, 'g', label="E exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, E_pr, 'b', label="E primal")
    plt.plot(tvec_int, E_dl, 'r', label="E dual")
    if problem.exact:
        plt.plot(tvec_int, E_ex, 'g', label="E exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, uP_pr[:, 0], 'b', label="ux at P primal")
    plt.plot(tvec_int, uP_dl[:, 0], 'r', label="ux at P dual")
    if problem.exact:
        plt.plot(tvec_int, uP_ex[:, 0], 'g', label="ux at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, uP_pr[:, 1], 'b', label="uy at P primal")
    plt.plot(tvec_int, uP_dl[:, 1], 'r', label="uy at P dual")
    if problem.exact:
        plt.plot(tvec_int, uP_ex[:, 1], 'g', label="uy at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, wP_pr[:, 0], 'b', label="w at P primal")
    plt.plot(tvec_int, wP_dl[:, 0], 'r', label="w at P dual")
    if problem.exact:
        plt.plot(tvec_int, wP_ex[:, 0], 'g', label="w at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, pdynP_pr, 'b', label="p dyn at P primal")
    plt.plot(tvec_stag, pdynP_dl, 'r', label="p dyn at P dual")
    if problem.exact:
        plt.plot(tvec_int, pdynP_ex, 'g', label="p dyn at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, divu_pr_L2, 'b', label="L2 norm div u primal")
    plt.plot(tvec_stag, divu_dl_L2, 'r', label="L2 norm div u primal")
    plt.legend()
    plt.show()
