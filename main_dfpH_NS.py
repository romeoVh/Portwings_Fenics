from __future__ import print_function
from problems.beltrami3d_problem import *
from problems.channel_problem import *
from solvers.dfpH_NS_solver import *
from solvers.ipcs_solver_new import *


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

def post_processing_ipcs_channel(ipcs, channel):
    plt.subplot(2, 2, 1)
    plot(ipcs.u_t)
    plt.title("Velocity plot at t=t_fin")
    plt.subplot(2, 2, 2)
    plot(ipcs.p_t)
    plt.title("Pressure plot at t=t_fin")
    plt.subplot(2, 2, 3)
    plt.plot(channel.t_vec, ipcs.outputs_arr[:, 0])
    plt.title("H_t")
    plt.subplot(2, 2, 4)
    plt.plot(channel.t_vec, ipcs.outputs_arr[:, 1])
    plt.title("divergence error of vector field")
    plt.show()


if __name__ == '__main__':
    # 1. Select Problem:

    # Beltrami 3D problem
    options = {"n_el":2,"n_t":100,"t_fin":1.0}
    beltrami = BeltramiProblem(options)

    # Channel 2D problem
    # options = {"n_el": 10, "n_t": 500, "t_fin": 6}
    # channel = ChannelProblem(options)

    # 2. Select Solver:

    options = {"pol_deg":2}
    ipcs = IPCS_Solver(options)

    ipcs.solve(beltrami)
    post_processing_ipcs_beltrami(ipcs,beltrami)

    #ipcs.solve(channel)
    #post_processing_ipcs_channel(ipcs, channel)

    #options = {"pol_deg":1}
    #pH_NS = DualFieldPHNSSolver(options)
    #pH_NS.solve(beltrami)


# Log book - Observations
# divergence is zero only for pol_deg = 1


