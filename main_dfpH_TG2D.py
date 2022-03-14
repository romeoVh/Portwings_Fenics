from __future__ import print_function
from problems.taylor_green2D import *
from solvers.dfpH_NS_solver_first_explicit import *

def post_processing_pH_NS_taylorgreen2D(outputs_arr, taylorgreen2D, stagger_time=False):
    if stagger_time:
        t_range = np.roll(taylorgreen2D.t_vec, 1) + taylorgreen2D.dt / 2.0
        t_range[0] = 0.0
    else:
        t_range = taylorgreen2D.t_vec
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.bar(t_range, outputs_arr[:, 0], width=float(taylorgreen2D.dt) / 2)
    plt.title("L2 error of v_t")
    plt.subplot(2, 4, 2)
    plt.bar(t_range, outputs_arr[:, 1], width=float(taylorgreen2D.dt) / 2)
    plt.title("L2 error of w_t")
    plt.subplot(2, 4, 3)
    plt.bar(t_range, outputs_arr[:, 2], width=float(taylorgreen2D.dt) / 2)
    plt.title("L2 error of p_t")
    plt.subplot(2, 4, 4)
    plt.plot(t_range, outputs_arr[:, 3])
    plt.plot(t_range, outputs_arr[:, 6])
    plt.legend(['H_ex_t', 'H_t'])
    plt.subplot(2, 4, 5)
    plt.plot(t_range, outputs_arr[:, 4])
    plt.plot(t_range, outputs_arr[:, 7])
    plt.legend(['E_ex_t', 'E_t'])
    plt.subplot(2, 4, 6)
    plt.plot(t_range, outputs_arr[:, 5])
    plt.plot(t_range, outputs_arr[:, 8])
    plt.legend(['Hel_ex_t', 'Hel_t'])
    plt.subplot(2, 4, 7)
    plt.plot(t_range, outputs_arr[:, 9])
    plt.title("divergence error of vector field")


if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D
    options = {"n_el": 20, "n_t": 300, "t_fin": 5}
    taylorgreen2D = TaylorGreen2D(options)

    options = {"pol_deg":1,"couple_primal_dual":True}
    # couple_primal_dual= True
    # --> supplies weak boundary conditions of primal (dual) as outputs of dual (primal)
    pH_NS = DualFieldPHNSSolver(options)

    pH_NS.solve(taylorgreen2D)
    post_processing_pH_NS_taylorgreen2D(pH_NS.outputs_arr_primal, taylorgreen2D)

    plt.show()



