from problems.taylor_green2D import *
from solvers.dfNS_palha import compute_sol


if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    options = {"n_el": 10, "n_t": 4, "t_fin": 1}
    taylorgreen2D = TaylorGreen2D(options)

    compute_sol(taylorgreen2D, 1, n_t, t_fin=1)

    plt.show()