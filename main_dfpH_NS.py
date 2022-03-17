from __future__ import print_function
from problems.beltrami_problem import *
from problems.channel_problem import *
from problems.driven_cavity_problem import *
from problems.taylor_green2D import *
from solvers.dfpH_NS_solver import *
from solvers.ipcs_solver import *


def post_processing_ipcs_beltrami(ipcs, beltrami):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.bar(beltrami.t_vec, ipcs.outputs_arr[:, 0], width=float(beltrami.dt) / 2)
    plt.title("L2 error of u_t")
    plt.subplot(2, 2, 2)
    plt.bar(beltrami.t_vec, ipcs.outputs_arr[:, 2], width=float(beltrami.dt) / 2)
    plt.title("L2 error of p_t")
    plt.subplot(2, 2, 3)
    plt.plot(beltrami.t_vec, ipcs.outputs_arr[:, 3])
    plt.plot(beltrami.t_vec, ipcs.outputs_arr[:, 4])
    plt.legend(['H_ex_t', 'H_t'])
    plt.subplot(2, 2, 4)
    plt.plot(beltrami.t_vec, ipcs.outputs_arr[:, 5])
    plt.title("divergence error of vector field")

def post_processing_ipcs_2d(ipcs, problem):
    plt.figure()
    plt.subplot(2, 2, 1)
    u_plot = plot(ipcs.u_t)
    plt.title("Velocity plot at t=t_fin")
    plt.colorbar(u_plot)
    plt.subplot(2, 2, 2)
    p_plot = plot(ipcs.p_t)
    plt.title("Pressure plot at t=t_fin")
    plt.colorbar(p_plot)
    plt.subplot(2, 2, 3)
    plt.plot(problem.t_vec, ipcs.outputs_arr[:, 0])
    plt.title("H_t")
    plt.subplot(2, 2, 4)
    plt.plot(problem.t_vec, ipcs.outputs_arr[:, 1])
    plt.title("divergence error of vector field")

def post_processing_pH_NS_beltrami(outputs_arr, beltrami,stagger_time = False):
    if stagger_time:
        t_range = np.roll(beltrami.t_vec, 1) + beltrami.dt / 2.0
        t_range[0] = 0.0
    else:
        t_range = beltrami.t_vec
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.bar(t_range, outputs_arr[:, 0], width=float(beltrami.dt) / 2)
    plt.title("L2 error of v_t")
    plt.subplot(2, 4, 2)
    plt.bar(t_range, outputs_arr[:, 1], width=float(beltrami.dt) / 2)
    plt.title("L2 error of w_t")
    plt.subplot(2, 4, 3)
    plt.bar(t_range, outputs_arr[:, 2], width=float(beltrami.dt) / 2)
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

def post_processing_pH_NS_2d(pH_sys,outputs_arr, problem,stagger_time = False):
    plt.figure()
    plt.subplot(2,3, 1)
    v_plot = plot(pH_sys.v_t)
    plt.title("Velocity plot at t=t_fin")
    plt.colorbar(v_plot)
    plt.subplot(2, 3, 2)
    v_plot = plot(pH_sys.w_t)
    plt.title("Vorticity plot at t=t_fin")
    plt.colorbar(v_plot)
    plt.subplot(2, 3, 3)
    p_plot = plot(pH_sys.p_t)
    plt.title("Pressure plot at t=t_fin")
    plt.colorbar(p_plot)
    plt.subplot(2, 3, 4)
    plt.plot(problem.t_vec, outputs_arr[:, 0])
    plt.title("H_t")
    plt.subplot(2, 3, 5)
    plt.plot(problem.t_vec, outputs_arr[:, 1])
    plt.title("divergence error of vector field")

def post_processing_pH_NS_taylorgreen(outputs_arr, taylorgreen, stagger_time=False):
    if stagger_time:
        t_range = np.roll(taylorgreen.t_vec, 1) + taylorgreen.dt / 2.0
        t_range[0] = 0.0
    else:
        t_range = taylorgreen.t_vec

    V = 8*pi**3
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t_range, outputs_arr[:, 0])
    plt.title("Energy")
    plt.subplot(1, 3, 2)
    plt.plot(t_range, outputs_arr[:, 1])
    plt.title('Enstrophy')
    plt.subplot(1, 3, 3)
    plt.plot(t_range, outputs_arr[:, 2])
    plt.title("divergence error of vector field")


if __name__ == '__main__':
    # 1. Select Problem:

    # Beltrami 3D problem
    # options = {"n_el":5,"n_t":100,"t_fin":.1}
    # beltrami = BeltramiProblem(options)
    #
    # # Channel 2D problem
    # options = {"n_el": 8, "n_t": 50, "t_fin": 1}
    # channel = ChannelProblem(options)
    #
    # # Driven Cavity 2D problem
    # options = {"n_el": 10, "n_t": 100, "t_fin": 2}
    # cavity = DrivenCavityProblem(options)

    # Driven Cavity 2D problem
    # options = {"n_el": 1, "n_t": 10, "t_fin":.002}
    # taylorgreen = TaylorGreen(options)
    # 2. Select Solver:

    # Taylor Green 2D
    options = {"n_el": 5, "n_t": 4, "t_fin": 1}
    taylorgreen2D = TaylorGreen2D(options)
    # options = {"pol_deg":2}
    # ipcs = IPCS_Solver(options)

    # ipcs.solve(beltrami)
    # post_processing_ipcs_beltrami(ipcs,beltrami)

    # ipcs.solve(channel)
    # post_processing_ipcs_2d(ipcs, channel)

    # ipcs.solve(cavity)
    # post_processing_ipcs_2d(ipcs, cavity)

    options = {"pol_deg":1,"couple_primal_dual":True}
    # couple_primal_dual= True
    # --> supplies weak boundary conditions of primal (dual) as outputs of dual (primal)
    pH_NS = DualFieldPHNSSolver(options)

    # pH_NS.solve(beltrami)
    # post_processing_pH_NS_beltrami(pH_NS.outputs_arr_primal, beltrami)
    # post_processing_pH_NS_beltrami(pH_NS.outputs_arr_dual, beltrami, True)

    pH_NS.solve(taylorgreen2D)
    post_processing_pH_NS_taylorgreen(pH_NS.outputs_arr_primal, taylorgreen2D)
    post_processing_pH_NS_taylorgreen(pH_NS.outputs_arr_dual, taylorgreen2D,True)
    #
    # # Work in progress
    # pH_NS.solve(cavity)
    # post_processing_pH_NS_2d(pH_NS.p_h_primal,pH_NS.outputs_arr_primal, cavity)
    # post_processing_pH_NS_2d(pH_NS.pH_dual, pH_NS.outputs_arr_dual, cavity)

    plt.show()



# Log book - Observations
# divergence is zero only for pol_deg = 1 for pH_NS solver


