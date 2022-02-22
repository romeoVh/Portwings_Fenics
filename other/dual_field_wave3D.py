## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from tools_plotting import setup
from tqdm import tqdm
# from time import sleep
from matplotlib.ticker import FormatStrFormatter


path_fig = "/home/andrea/Pictures/PythonPlots/DualField_wave3D/"
bc_case = "_DN"
geo_case = "_3D"


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    def m_form32(v_3, p_3, v_2, u_2):
        m_form = inner(v_3, p_3) * dx + inner(v_2, u_2) * dx

        return m_form

    def m_form10(v_1, u_1, v_0, p_0):
        m_form = inner(v_1, u_1) * dx + inner(v_0, p_0) * dx

        return m_form

    def j_form32(v_3, p_3, v_2, u_2):
        j_form = dot(v_3, div(u_2)) * dx - dot(div(v_2), p_3) * dx

        return j_form

    def j_form10(v_1, u_1, v_0, p_0):
        j_form = dot(v_1, grad(p_0)) * dx - dot(grad(v_0), u_1) * dx

        return j_form

    def bdflow32(v_2, p_0):
        b_form = dot(v_2, n_ver) * p_0 * ds
        return b_form

    def bdflow10(v_0, u_2):
        b_form = v_0 * dot(u_2, n_ver) * ds
        return b_form

    L = 1/2
    mesh = BoxMesh(n_el, n_el, n_el, 1, 1/2, 1/2)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", tetrahedron, deg)
    P_1 = FiniteElement("N1curl", tetrahedron, deg)
    # P_1 = FiniteElement("N1curl", tetrahedron, deg)
    P_2 = FiniteElement("RT", tetrahedron, deg)
    # Integral evaluation on Raviart-Thomas and NED for deg=3 completely freezes interpolation
    # P_2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
    P_3 = FiniteElement("DG", tetrahedron, deg - 1)

    V_3 = FunctionSpace(mesh, P_3)
    V_1 = FunctionSpace(mesh, P_1)

    V_0 = FunctionSpace(mesh, P_0)
    V_2 = FunctionSpace(mesh, P_2)

    V_32 = V_3 * V_2
    V_10 = V_1 * V_0

    # print(V_0.dim())
    # print(V_1.dim())
    # print(V_2.dim())
    # print(V_3.dim())

    v_32 = TestFunction(V_32)
    v_3, v_2 = split(v_32)

    v_10 = TestFunction(V_10)
    v_1, v_0 = split(v_10)

    e_32 = TrialFunction(V_32)
    p_3, u_2 = split(e_32)

    e_10 = TrialFunction(V_10)
    u_1, p_0 = split(e_10)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y, z = SpatialCoordinate(mesh)

    om_x = 1
    om_y = 1
    om_z = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 0
    phi_y = 0
    phi_z = 0
    phi_t = 0

    dt = Constant(t_fin / n_t)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

    dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

    grad_gxyz = as_vector([dgxyz_x,
                           dgxyz_y,
                           dgxyz_z]) # grad(gxyz)


    p_ex = gxyz * dft
    u_ex = grad_gxyz * ft

    p_ex_1 = gxyz * dft_1
    u_ex_1 = grad_gxyz * ft_1

    u_ex_mid = 0.5 * (u_ex + u_ex_1)
    p_ex_mid = 0.5 * (p_ex + p_ex_1)

    p0_3 = interpolate(p_ex, V_3)
    u0_2 = interpolate(u_ex, V_2)
    u0_1 = interpolate(u_ex, V_1)
    p0_0 = interpolate(p_ex, V_0)

    if bd_cond == "D":
        bc_D = [DirichletBC(V_10.sub(1), p_ex_1, "on_boundary")]
        bc_D_nat = None

        bc_N = None
        bc_N_nat = [DirichletBC(V_32.sub(1), u_ex_1, "on_boundary")]

    elif bd_cond == "N":
        bc_N = [DirichletBC(V_32.sub(1), u_ex_1, "on_boundary")]
        bc_N_nat = None

        bc_D = None
        bc_D_nat = [DirichletBC(V_10.sub(1), p_ex_1, "on_boundary")]
    else:
        bc_D = [DirichletBC(V_10.sub(1), p_ex_1, 1), \
                DirichletBC(V_10.sub(1), p_ex_1, 3),
                DirichletBC(V_10.sub(1), p_ex_1, 5)]

        bc_D_nat = [DirichletBC(V_10.sub(1), p_ex_1, 2), \
                    DirichletBC(V_10.sub(1), p_ex_1, 4), \
                    DirichletBC(V_10.sub(1), p_ex_1, 6)]

        bc_N = [DirichletBC(V_32.sub(1), u_ex_1, 2), \
                DirichletBC(V_32.sub(1), u_ex_1, 4),
                DirichletBC(V_32.sub(1), u_ex_1, 6)]

        bc_N_nat = [DirichletBC(V_32.sub(1), u_ex_1, 1), \
                    DirichletBC(V_32.sub(1), u_ex_1, 3), \
                    DirichletBC(V_32.sub(1), u_ex_1, 5)]

    dofs10_D = []
    dofs32_D = []

    if bc_D is not None:
        for ii in range(len(bc_D)):
            nodes10_D = V_1.dim() + bc_D[ii].nodes
            nodes32_D = V_3.dim() + bc_N_nat[ii].nodes

            dofs10_D = dofs10_D + list(nodes10_D)
            dofs32_D = dofs32_D + list(nodes32_D)


    dofs10_D = list(set(dofs10_D))
    dofs32_D = list(set(dofs32_D))

    # print("dofs on Gamma_D for 10")
    # print(dofs10_D)
    # print("dofs on Gamma_D for 32")
    # print(dofs32_D)

    dofs10_N = []
    dofs32_N = []

    if bc_N is not None:
        for ii in range(len(bc_N)):
            nodes32_N = V_3.dim() + bc_N[ii].nodes
            nodes10_N = V_1.dim() + bc_D_nat[ii].nodes

            dofs32_N = dofs32_N + list(nodes32_N)
            dofs10_N = dofs10_N + list(nodes10_N)

    dofs32_N = list(set(dofs32_N))
    dofs10_N = list(set(dofs10_N))

    for element in dofs10_D:
        if element in dofs10_N:
            dofs10_N.remove(element)

    for element in dofs32_N:
        if element in dofs32_D:
            dofs32_D.remove(element)

    # print("dofs on Gamma_N for 10")
    # print(dofs10_N)
    # print("dofs on Gamma_N for 32")
    # print(dofs32_N)


    Ppoint = (L/5, L/5, L/5)

    p_0P = np.zeros((1+n_t,))
    p_0P[0] = interpolate(p_ex, V_0).at(Ppoint)

    p_3P = np.zeros((1+n_t, ))
    p_3P[0] = interpolate(p_ex, V_3).at(Ppoint)

    e0_32 = Function(V_32, name="e_32 initial")
    e0_10 = Function(V_10, name="e_10 initial")

    e0_32.sub(0).assign(p0_3)
    e0_32.sub(1).assign(u0_2)

    e0_10.sub(0).assign(u0_1)
    e0_10.sub(1).assign(p0_0)


    en_32 = Function(V_32, name="e_32 n")
    en_32.assign(e0_32)

    en_10 = Function(V_10, name="e_10 n")
    en_10.assign(e0_10)

    enmid_32 = Function(V_32, name="e_32 n+1/2")
    enmid_10 = Function(V_10, name="e_10 n+1/2")

    en1_10 = Function(V_10, name="e_10 n+1")
    en1_32 = Function(V_32, name="e_32 n+1")

    pn_3, un_2 = en_32.split()
    un_1, pn_0 = en_10.split()

    pnmid_3, unmid_2 = enmid_32.split()
    unmid_1, pnmid_0 = enmid_10.split()

    pn1_3, un1_2 = en1_32.split()
    un1_1, pn1_0 = en1_10.split()

    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_10 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hn_31 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_1, un_1) * dx)
    Hn_02 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_2, un_2) * dx)

    Hn_3210 = 0.5 * (dot(pn_0, pn_3) * dx + dot(un_2, un_1) * dx)

    Hn_ex = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))

    Hdot_n = 1/dt*(dot(pnmid_0, pn1_3 - pn_3) * dx(domain=mesh) + dot(unmid_2, un1_1 - un_1) * dx(domain=mesh))

    bdflow_midn = pnmid_0 * dot(unmid_2, n_ver) * ds(domain=mesh)

    y_nmid_ess10 = 1 / dt * m_form10(v_1, un1_1 - un_1, v_0, pn1_0 - pn_0) \
                   - j_form10(v_1, unmid_1, v_0, pnmid_0)
    u_nmid_nat10 = bdflow10(v_0, unmid_2)


    y_nmid_ess32 = 1 / dt * m_form32(v_3, pn1_3 - pn_3, v_2, un1_2 - un_2) \
              - j_form32(v_3, pnmid_3, v_2, unmid_2)
    u_nmid_nat32 = bdflow32(v_2, pnmid_0)

    bdflow_n = pn_0 * dot(un_2, n_ver) * ds(domain=mesh)
    bdflow_ex_n = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))

    H_31_vec = np.zeros((1 + n_t,))
    H_02_vec = np.zeros((1 + n_t,))

    H_3210_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((n_t,))

    bdflow_mid_vec = np.zeros((n_t,))

    bdflow10_mid_vec = np.zeros((n_t,))
    bdflow32_mid_vec = np.zeros((n_t,))

    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    H_ex_vec = np.zeros((1 + n_t,))

    errL2_p_3_vec = np.zeros((1 + n_t,))
    errL2_u_1_vec = np.zeros((1 + n_t,))

    errL2_p_0_vec = np.zeros((1 + n_t,))
    errL2_u_2_vec = np.zeros((1 + n_t,))

    errHcurl_u_1_vec = np.zeros((1 + n_t,))
    errH1_p_0_vec = np.zeros((1 + n_t,))
    errHdiv_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    errH_32_vec = np.zeros((1 + n_t,))
    errH_10_vec = np.zeros((1 + n_t,))
    errH_3210_vec = np.zeros((1 + n_t,))

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)

    H_31_vec[0] = assemble(Hn_31)
    H_02_vec[0] = assemble(Hn_02)

    H_3210_vec[0] = assemble(Hn_3210)

    H_ex_vec[0] = assemble(Hn_ex)

    errH_32_vec[0] = np.abs(H_32_vec[0] - H_ex_vec[0])
    errH_10_vec[0] = np.abs(H_10_vec[0] - H_ex_vec[0])
    errH_3210_vec[0] = np.abs(H_3210_vec[0] - H_ex_vec[0])

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)

    errL2_p_3_vec[0] = errornorm(p_ex, p0_3, norm_type="L2")
    errL2_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    errL2_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="L2")
    errL2_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="L2")

    errHcurl_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="Hcurl")
    errH1_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="H1")
    errHdiv_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="Hdiv")

    err_p30_vec[0] = np.sqrt(assemble(inner(p0_3 - p0_0, p0_3 - p0_0) * dx))
    err_u12_vec[0] = np.sqrt(assemble(inner(u0_2 - u0_1, u0_2 - u0_1) * dx))

    ## Settings of intermediate variables and matrices for the 2 linear systems

    a_form10 = m_form10(v_1, u_1, v_0, p_0) - 0.5*dt*j_form10(v_1, u_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, u_2) - 0.5*dt*j_form32(v_3, p_3, v_2, u_2)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_2 = interpolate(u_ex_mid, V_2)
        input_0 = interpolate(p_ex_mid, V_0)

        ## Integration of 32 system (Dirichlet natural)

        A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

        b_form32 = m_form32(v_3, pn_3, v_2, un_2) + dt*(0.5*j_form32(v_3, pn_3, v_2, un_2) + bdflow32(v_2, input_0))
        b_vec32 = assemble(b_form32)

        solve(A_32, en1_32, b_vec32, solver_parameters=params)

        ## Integration of 10 system (Neumann natural)

        A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')

        b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt * (0.5 * j_form10(v_1, un_1, v_0, pn_0) + bdflow10(v_0, input_2))

        b_vec10 = assemble(b_form10)

        solve(A_10, en1_10, b_vec10, solver_parameters=params)

        # Computation of energy rate and fluxes

        enmid_10.assign(0.5 * (en_10 + en1_10))
        enmid_32.assign(0.5 * (en_32 + en1_32))

        Hdot_vec[ii] = assemble(Hdot_n)

        bdflow_mid_vec[ii] = assemble(bdflow_midn)

        yhat_10 = assemble(y_nmid_ess10).vector().get_local()[dofs10_D]
        u_midn_10 = enmid_10.vector().get_local()[dofs10_D]

        uhat_10 = assemble(u_nmid_nat10).vector().get_local()[dofs10_N]
        y_midn_10 = enmid_10.vector().get_local()[dofs10_N]

        bdflow10_nat = np.dot(uhat_10, y_midn_10)
        bdflow10_ess = np.dot(yhat_10, u_midn_10)
        bdflow10_mid_vec[ii] = bdflow10_nat + bdflow10_ess

        yhat_32 = assemble(y_nmid_ess32).vector().get_local()[dofs32_N]
        u_midn_32 = enmid_32.vector().get_local()[dofs32_N]

        uhat_32 = assemble(u_nmid_nat32).vector().get_local()[dofs32_D]
        y_midn_32 = enmid_32.vector().get_local()[dofs32_D]

        bdflow32_nat = np.dot(uhat_32, y_midn_32)
        bdflow32_ess = np.dot(yhat_32, u_midn_32)
        bdflow32_mid_vec[ii] = bdflow32_nat + bdflow32_ess


        # New assign

        en_32.assign(en1_32)
        en_10.assign(en1_10)

        un_1, pn_0 = en_10.split()
        pn_3, un_2 = en_32.split()

        bdflow_vec[ii+1] = assemble(bdflow_n)

        H_32_vec[ii+1] = assemble(Hn_32)
        H_10_vec[ii+1] = assemble(Hn_10)

        H_31_vec[ii+1] = assemble(Hn_31)
        H_02_vec[ii+1] = assemble(Hn_02)

        H_3210_vec[ii+1] = assemble(Hn_3210)

        p_3P[ii+1] = pn_3.at(Ppoint)
        p_0P[ii+1] = pn_0.at(Ppoint)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        H_ex_vec[ii + 1] = assemble(Hn_ex)

        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)

        # print(bdflow_ex_vec[ii+1])
        errH_32_vec[ii + 1] = np.abs(H_32_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_10_vec[ii + 1] = np.abs(H_10_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_3210_vec[ii + 1] = np.abs(H_3210_vec[ii + 1] - H_ex_vec[ii + 1])

        errL2_p_3_vec[ii + 1] = errornorm(p_ex, pn_3, norm_type="L2")
        errL2_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="L2")
        errL2_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="L2")
        errL2_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="L2")

        errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="Hcurl")
        errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="H1")
        errHdiv_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="Hdiv")

        err_p30_vec[ii + 1] = np.sqrt(assemble(inner(pn_3 - pn_0, pn_3 - pn_0) * dx))
        err_u12_vec[ii + 1] = np.sqrt(assemble(inner(un_2 - un_1, un_2 - un_1) * dx))

        #     p_3P[ii + 1] = pn_3.at(Ppoint)
        #     p_0P[ii + 1] = pn_0.at(Ppoint)
        #
        # err_p3.assign(pn_3 - interpolate(p_ex, V_3))
        # err_p0.assign(pn_0 - interpolate(p_ex, V_0))
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(err_p3, axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("Error $p_3$")
        # fig.colorbar(contours)
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(err_p0, axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("Error $p_0$")
        # fig.colorbar(contours)
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(interpolate(p_ex, V_3), axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("$p_3$ Exact")
        # fig.colorbar(contours)
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(interpolate(p_ex, V_0), axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("$p_0$ Exact")
        # fig.colorbar(contours)
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(pn_3, axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("$P_3$")
        # fig.colorbar(contours)
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # contours = trisurf(pn_0, axes=axes, cmap="inferno")
        # axes.set_aspect("auto")
        # axes.set_title("$P_0$")
        # fig.colorbar(contours)
        #
        # print(r"Initial and final 32 energy:")
        # print(r"Inital: ", H_32_vec[0])
        # print(r"Final: ", H_32_vec[-1])
        # print(r"Delta: ", H_32_vec[-1] - H_32_vec[0])
        #
        # print(r"Initial and final 10 energy:")
        # print(r"Inital: ", H_10_vec[0])
        # print(r"Final: ", H_10_vec[-1])
        # print(r"Delta: ", H_10_vec[-1] - H_10_vec[0])
        #
        # plt.figure()
        # plt.plot(t_vec, p_3P, 'r-', label=r'$p_3$')
        # plt.plot(t_vec, p_0P, 'b-', label=r'$p_0$')
        # plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
        #          * np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
        # plt.xlabel(r'Time [s]')
        # plt.title(r'$p$ at ' + str(Ppoint))
        # plt.legend()

        # err_p_3 = np.sqrt(np.sum(float(dt) * np.power(err_p_3_vec, 2)))
        # err_u_1 = np.sqrt(np.sum(float(dt) * np.power(err_u_1_vec, 2)))
        # err_p_0 = np.sqrt(np.sum(float(dt) * np.power(err_p_0_vec, 2)))
        # err_u_2 = np.sqrt(np.sum(float(dt) * np.power(err_u_2_vec, 2)))
        #
        # err_p_3 = max(err_p_3_vec)
        # err_u_1 = max(err_u_1_vec)
        #
        # err_p_0 = max(err_p_0_vec)
        # err_u_2 = max(err_u_2_vec)
        #
        # err_p30 = max(err_p30_vec)
        # err_u12 = max(err_u12_vec)

    errL2_p_3 = errL2_p_3_vec[-1]
    errL2_u_1 = errL2_u_1_vec[-1]

    errL2_p_0 = errL2_p_0_vec[-1]
    errL2_u_2 = errL2_u_2_vec[-1]

    errHcurl_u_1 = errHcurl_u_1_vec[-1]

    errH1_p_0 = errH1_p_0_vec[-1]
    errHdiv_u_2 = errHdiv_u_2_vec[-1]

    err_p30 = err_p30_vec[-1]
    err_u12 = err_u12_vec[-1]

    errH_3210 = errH_3210_vec[-1]
    errH_10 = errH_10_vec[-1]
    errH_32 = errH_32_vec[-1]

    int_bd_flow = np.zeros((1 + n_t,))

    for i in range(n_t):
        int_bd_flow[i+1] = int_bd_flow[i] + dt*bdflow_mid_vec[i]

    H_df_vec = H_3210_vec[0] + int_bd_flow

    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "energy_df": H_df_vec, "energy_3210": H_3210_vec,\
                "energy_32": H_32_vec, "energy_01": H_10_vec, "energy_31": H_31_vec, "energy_02": H_02_vec, \
                "power": Hdot_vec, "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "int_flow": int_bd_flow, \
                "flow_mid": bdflow_mid_vec, "flow10_mid": bdflow10_mid_vec, "flow32_mid": bdflow32_mid_vec,\
                "err_p3": errL2_p_3, "err_u1": [errL2_u_1, errHcurl_u_1], \
                "err_p0": [errL2_p_0, errH1_p_0], "err_u2": [errL2_u_2, errHdiv_u_2], "err_p30": err_p30, \
                "err_u12": err_u12, "err_H": [errH_3210, errH_10, errH_32]}

    return dict_res


bd_cond = input("Enter bc: ")
save_plots = input("Save plots? ")

n_elem = 4
pol_deg = 3

n_time = 200
t_fin = 5

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)

t_vec = results["t_span"]
Hdot_vec = results["power"]

bdflow_vec = results["flow"]
bdflow_mid = results["flow_mid"]

bdflow10_mid = results["flow10_mid"]
bdflow32_mid = results["flow32_mid"]
int_bdflow = results["int_flow"]

H_df = results["energy_df"]
H_3210 = results["energy_3210"]

H_32 = results["energy_32"]
H_01 = results["energy_01"]

H_31 = results["energy_31"]
H_02 = results["energy_02"]

H_ex = results["energy_ex"]
bdflow_ex_vec = results["flow_ex"]

errL2_p3 = results["err_p3"]
errL2_u1, errHcurl_u1 = results["err_u1"]
errL2_p0, errH1_p0 = results["err_p0"]
errL2_u2, errHdiv_u2 = results["err_u2"]

err_Hs, err_H10, err_H32 = results["err_H"]



plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_vec - bdflow_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$P -<e^\partial_{h}, f^\partial_{h}>_{\partial M}$')
plt.title(r'Power balance conservation')

if save_plots:
    plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt - bdflow10_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$\dot{H}^{\widehat{3}1} - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_q} - \mathbf{u}^p \widehat{\mathbf{y}}^q$')
plt.title(r'Conservation law $\dot{H}^{3\widehat{1}}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_bal10" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_32)/dt - bdflow32_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$\dot{H}^{32} - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_p} - \mathbf{u}^q \widehat{\mathbf{y}}^p$')
plt.title(r'Conservation law $\dot{H}^{\widehat{3}1}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_bal32" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
ax = plt.gca()
plt.plot(t_vec, bdflow_vec - bdflow_ex_vec, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$<e^\partial_{h}, f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}, f^\partial_{\mathrm{ex}}>_{\partial M}$')
plt.title(r'Discrete and exact boundary flow')

if save_plots:
    plt.savefig(path_fig + "bd_flow" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_01)/dt - bdflow_mid), '-v', label=r"$\dot{H}_{h}^{3\widehat{1}}$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_32)/dt - bdflow_mid), '--', label=r"$\dot{H}_{h}^{\widehat{3}1}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_02)/dt - bdflow_mid), '-.+', label=r"$\dot{H}^{02}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_31)/dt - bdflow_mid), '--*', label=r"$\dot{H}^{31}$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_3210)/dt - bdflow_mid), '-.', label=r'$\frac{\dot{H}_{T, h}}{2}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\dot{H}_h - <e^\partial_{h}, f^\partial_{h}>_{\partial M}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "dHdt" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs((H_01 - H_01[0]) - (H_ex-H_ex[0])), '-v', label=r'$\Delta H_{h}^{3\widehat{1}}$')
plt.plot(t_vec, np.abs((H_32 - H_32[0]) - (H_ex-H_ex[0])), '--', label=r'$\Delta H_{h}^{\widehat{3}1}$')
# plt.plot(t_vec, np.abs((H_02 - H_02[0]) - (H_ex-H_ex[0])), '--+', label=r'$\Delta H^{02}$')
# plt.plot(t_vec, np.abs((H_31 - H_31[0]) - (H_ex-H_ex[0])), '--*', label=r'$\Delta H^{31}$')
plt.plot(t_vec, np.abs((H_3210 - H_3210[0]) - (H_ex-H_ex[0])), '-.', label=r'$\frac{\Delta H_{T, h}}{2}$')
plt.plot(t_vec, np.abs(int_bdflow - (H_ex-H_ex[0])), '-.+', label=r'$\int_0^t P_h(\tau) d\tau$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\Delta H_h - \Delta H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "deltaH" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs(H_01 - H_ex), '-v', label=r'$H_h^{3\widehat{1}}$')
plt.plot(t_vec, np.abs(H_32 - H_ex), '--', label=r'$H_h^{\widehat{3}1}$')
# plt.plot(t_vec, np.abs(H_02 - H_ex), '--+', label=r'$H^{02}$')
# plt.plot(t_vec, np.abs(H_31 - H_ex), '--*', label=r'$H^{31}$')
plt.plot(t_vec, np.abs(H_3210 - H_ex), '-.', label=r'$\frac{H_{T, h}}{2}$')
# plt.plot(t_vec, np.abs(H_df - H_ex), '-.+', label=r'$H_{\mathrm{df}}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|H_h - H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "H" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()


# print("Error L2 p3: " + str(errL2_p3))
#
# print("Error L2 u1: " + str(errL2_u1))
# print("Error Hcurl u1: " + str(errHcurl_u1))
#
# print("Error L2 p0: " + str(errL2_p0))
# print("Error H1 p0: " + str(errH1_p0))
#
# print("Error L2 u2: " + str(errL2_u2))
# print("Error Hdiv u2: " + str(errHdiv_u2))
#
# print("Error Hs: " + str(err_Hs))
# print("Error H_10: " + str(err_H10))
# print("Error H_32: " + str(err_H32))
