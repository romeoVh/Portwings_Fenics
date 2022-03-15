from problems.taylor_green2D import *
from solvers.dfNS_palha import compute_sol


if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    deg = 1
    n_t = 10
    t_f = 1
    options = {"n_el": 30, "t_fin": t_f, "n_t": n_t}
    taylorgreen2D = TaylorGreen2D(options)

    t_vec, H_pr, H_dl, H_ex, wP_pr, wP_dl, wP_ex = compute_sol(taylorgreen2D, deg, n_t, t_f)

    plt.figure()
    plt.plot(t_vec, H_pr, 'b', label="H primal")
    plt.plot(t_vec, H_dl, 'r', label="H dual")
    plt.plot(t_vec, H_ex, 'g', label="H exact")
    plt.legend()

    plt.figure()
    plt.plot(t_vec, wP_pr, 'b', label="w at P primal")
    plt.plot(t_vec, wP_dl, 'r', label="w at P dual")
    plt.plot(t_vec, wP_ex, 'g', label="w at P exact")
    plt.legend()
    plt.show()
