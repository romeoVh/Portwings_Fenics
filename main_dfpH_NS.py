from __future__ import print_function
from problems.beltrami_problem import *
from problems.channel_problem import *
from problems.driven_cavity_problem import *
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
    plt.subplot(2, 3, 1)
    plt.bar(t_range, outputs_arr[:, 0], width=float(beltrami.dt) / 2)
    plt.title("L2 error of v_t")
    plt.subplot(2, 3, 2)
    plt.bar(t_range, outputs_arr[:, 1], width=float(beltrami.dt) / 2)
    plt.title("L2 error of w_t")
    plt.subplot(2, 3, 3)
    plt.bar(t_range, outputs_arr[:, 2], width=float(beltrami.dt) / 2)
    plt.title("L2 error of p_t")
    plt.subplot(2, 3, 4)
    plt.plot(t_range, outputs_arr[:, 3:5])
    plt.legend(['H_ex_t', 'H_t'])
    plt.subplot(2, 3, 5)
    plt.plot(t_range, outputs_arr[:, 5])
    plt.title("divergence error of vector field")

if __name__ == '__main__':
    # 1. Select Problem:

    # Beltrami 3D problem
    options = {"n_el":2,"n_t":75,"t_fin":0.5}
    beltrami = BeltramiProblem(options)

    # Channel 2D problem
    # options = {"n_el": 8, "n_t": 50, "t_fin": 1}
    # channel = ChannelProblem(options)

    # Driven Cavity 2D problem
    # options = {"n_el": 15, "n_t": 500, "t_fin": 10}
    # cavity = DrivenCavityProblem(options)

    # 2. Select Solver:

    # options = {"pol_deg":2}
    # ipcs = IPCS_Solver(options)

    # ipcs.solve(beltrami)
    # post_processing_ipcs_beltrami(ipcs,beltrami)

    # ipcs.solve(channel)
    # post_processing_ipcs_2d(ipcs, channel)

    # ipcs.solve(cavity)
    # post_processing_ipcs_2d(ipcs, cavity)

    options = {"pol_deg":1,"stagger_time":True,"couple_primal_dual":True}
    # Options couple_primal_dual + time_staggering should be always True
    # Only the Beltrami problem can be tested with either option false
    # In future, both options should be made True by default
    pH_NS = DualFieldPHNSSolver(options)
    pH_NS.solve(beltrami)
    post_processing_pH_NS_beltrami(pH_NS.outputs_arr_primal, beltrami)
    post_processing_pH_NS_beltrami(pH_NS.outputs_arr_dual, beltrami,True)
    plt.show()



# Log book - Observations
# divergence is zero only for pol_deg = 1 for pH_NS solver


