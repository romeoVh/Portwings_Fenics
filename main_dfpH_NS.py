from __future__ import print_function
from problems.beltrami_problem import *
from problems.channel_problem import *
from problems.driven_cavity_problem import *
from solvers.dfpH_NS_solver import *
from solvers.ipcs_solver import *


def post_processing_ipcs_beltrami(ipcs, beltrami):
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
    plt.show()

def post_processing_ipcs_2d(ipcs, problem):
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
    plt.show()

def post_processing_pH_NS_beltrami(pH_NS, beltrami):
    plt.subplot(2, 3, 1)
    plt.bar(beltrami.t_vec, pH_NS.outputs_arr_primal[:, 0], width=float(beltrami.dt) / 2)
    plt.title("L2 error of v_t")
    plt.subplot(2, 3, 2)
    plt.bar(beltrami.t_vec, pH_NS.outputs_arr_primal[:, 1], width=float(beltrami.dt) / 2)
    plt.title("L2 error of w_t")
    plt.subplot(2, 3, 3)
    plt.bar(beltrami.t_vec, pH_NS.outputs_arr_primal[:, 2], width=float(beltrami.dt) / 2)
    plt.title("L2 error of p_t")
    plt.subplot(2, 3, 4)
    plt.plot(beltrami.t_vec, pH_NS.outputs_arr_primal[:, 3:5])
    plt.legend(['H_ex_t', 'H_t'])
    plt.subplot(2, 3, 5)
    plt.plot(beltrami.t_vec, pH_NS.outputs_arr_primal[:, 5])
    plt.title("divergence error of vector field")
    plt.show()

if __name__ == '__main__':
    # 1. Select Problem:

    # Beltrami 3D problem
    # options = {"n_el":2,"n_t":100,"t_fin":1.0}
    # beltrami = BeltramiProblem(options)

    # Channel 2D problem
    options = {"n_el": 10, "n_t": 500, "t_fin": 6}
    channel = ChannelProblem(options)

    # Driven Cavity 2D problem
    #options = {"n_el": 15, "n_t": 500, "t_fin": 10}
    #cavity = DrivenCavityProblem(options)

    # 2. Select Solver:

    options = {"pol_deg":2}
    ipcs = IPCS_Solver(options)

    # ipcs.solve(beltrami)
    # post_processing_ipcs_beltrami(ipcs,beltrami)

    ipcs.solve(channel)
    post_processing_ipcs_2d(ipcs, channel)

    #ipcs.solve(cavity)
    #post_processing_ipcs_2d(ipcs, cavity)

    # options = {"pol_deg":1}
    # pH_NS = DualFieldPHNSSolver(options)
    # pH_NS.solve(beltrami)
    # post_processing_pH_NS_beltrami(pH_NS, beltrami)


# Log book - Observations
# divergence is zero only for pol_deg = 1 for pH_NS solver


