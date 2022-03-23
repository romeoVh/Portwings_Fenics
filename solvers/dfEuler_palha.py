from fenics import *
from time import time
from tqdm import tqdm
import numpy as np
from .utilities.operators import *
import matplotlib.pyplot as plt
from vedo.dolfin import plot

def explicit_step_primal_incompressible(dt_0, problem, u_n, wT_n, V_pr):

    chi_pr = TestFunction(V_pr)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_pr)

    x_pr = TrialFunction(V_pr)
    u_pr, w_pr, p_pr = split(x_pr)

    a1_form_vel = (1 / dt_0) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr) \
                  - 0.5*wcross1_form(chi_u_pr, u_pr, wT_n, problem.dimM)
    a2_form_vor = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_form_p = - adj_divu_form(chi_p_pr, u_pr)
    A0_pr = assemble(a1_form_vel + a2_form_vor + a3_form_p)

    b1_form_vel = (1 / dt_0) * m_form(chi_u_pr, u_n) + 0.5*wcross1_form(chi_u_pr, u_n, wT_n, problem.dimM)
    b0_pr = assemble(b1_form_vel)

    x_sol = Function(V_pr)

    solve(A0_pr, x_sol.vector(), b0_pr, "gmres", "icc")

    return x_sol

def compute_sol(problem, pol_deg, n_t, t_fin=1):
    # Implementation of the dual field formulation for periodic navier stokes
    mesh = problem.mesh
    problem.init_mesh()

    # Primal trimmed polynomial finite element families
    ufl_cell = mesh.ufl_cell()
    P_1 = FiniteElement("N1curl", ufl_cell, pol_deg)
    P_0 = FiniteElement("CG", ufl_cell, pol_deg)
    if problem.dimM == 3:
        P_2 = FiniteElement("RT", ufl_cell, pol_deg)
    elif problem.dimM == 2:
        P_2 = FiniteElement("DG", ufl_cell, pol_deg - 1)

    P_primal = MixedElement([P_1, P_2, P_0])

    # Dual trimmed polynomial finite element families
    PT_n1 = FiniteElement("RT", ufl_cell, pol_deg)
    PT_n = FiniteElement("DG", ufl_cell, pol_deg - 1)
    if problem.dimM == 3:
        PT_n2 = FiniteElement("N1curl", ufl_cell, pol_deg)
    elif problem.dimM == 2:
        PT_n2 = FiniteElement("CG", ufl_cell, pol_deg)

    P_dual = MixedElement([PT_n1, PT_n2, PT_n])

    # Define primal function spaces for periodic bcs
    if problem.periodic==True:
        print("Periodic domain")
        V_1 = FunctionSpace(mesh, P_1, constrained_domain=problem.boundary_conditions())
        V_2 = FunctionSpace(mesh, P_2, constrained_domain=problem.boundary_conditions())
        V_0 = FunctionSpace(mesh, P_0, constrained_domain=problem.boundary_conditions())
        V_primal = FunctionSpace(mesh, P_primal, constrained_domain=problem.boundary_conditions())  # V_1 x V_2 x V_0
        # Dual function spaces
        VT_n1 = FunctionSpace(mesh, PT_n1, constrained_domain=problem.boundary_conditions())
        VT_n2 = FunctionSpace(mesh, PT_n2, constrained_domain=problem.boundary_conditions())
        VT_n = FunctionSpace(mesh, PT_n, constrained_domain=problem.boundary_conditions())
        V_dual = FunctionSpace(mesh, P_dual, constrained_domain=problem.boundary_conditions())  # VT_n-1 x VT_n-2 x VT_n
    else:
        V_1 = FunctionSpace(mesh, P_1)
        V_2 = FunctionSpace(mesh, P_2)
        V_0 = FunctionSpace(mesh, P_0)
        V_primal = FunctionSpace(mesh, P_primal)  # V_1 x V_2 x V_0
        # Dual function spaces
        VT_n1 = FunctionSpace(mesh, PT_n1)
        VT_n2 = FunctionSpace(mesh, PT_n2)
        VT_n = FunctionSpace(mesh, PT_n)
        V_dual = FunctionSpace(mesh, P_dual)  # VT_n-1 x VT_n-2 x VT_n
    print("Function Space dimensions, Primal - Dual: ", [V_primal.dim(), V_dual.dim()])

    # Define Function assigners
    fa_primal = FunctionAssigner(V_primal, [V_1, V_2, V_0])
    fa_dual = FunctionAssigner(V_dual, [VT_n1, VT_n2, VT_n])
    # Set initial condition at t=0
    xprimal_0 = Function(V_primal, name="x_0 primal")
    xdual_0 = Function(V_dual, name="x_0 dual")

    fa_primal.assign(xprimal_0, problem.initial_conditions(V_1, V_2, V_0))
    fa_dual.assign(xdual_0, problem.initial_conditions(VT_n1, VT_n2, VT_n))

    dt = Constant(t_fin / n_t)
    tvec_int = np.linspace(0, n_t * float(dt), 1 + n_t)
    tvec_stag = np.zeros((n_t+1))
    tvec_stag[1:] = np.linspace(float(dt)/2, float(dt)*(n_t - 1/2), n_t)

    u_pr_0, w_pr_0, p_pr_0 = xprimal_0.split(deepcopy=True)
    u_dl_0, w_dl_0, p_dl_0 = xdual_0.split(deepcopy=True)


    xprimal_n12 = explicit_step_primal_incompressible(dt/2, problem, u_pr_0, w_dl_0, V_primal)

    print("Explicit step solved")

    u_pr_12, w_pr_12, p_pr_init = xprimal_n12.split(deepcopy=True)
    # Primal intermediate variables
    xprimal_n32 = Function(V_primal, name="u, w at n+3/2, p at n+1")

    xprimal_n1 = Function(V_primal, name="u, w at n+1, p at n+1/2")

    # Dual intermediate variables
    xdual_n = Function(V_dual, name="uT, wT at n, pT at n-1/2")
    xdual_n.assign(xdual_0)

    xdual_n1 = Function(V_dual, name="u, w at n+1, p at n+1/2")

    # Kinetic energy and Enstrophy definition
    # Primal
    H_pr_vec = np.zeros((n_t + 1, ))
    H_pr_0 = 0.5*dot(u_pr_0, u_pr_0) * dx
    H_pr_vec[0] = assemble(H_pr_0)

    E_pr_vec = np.zeros((n_t + 1, ))
    E_pr_0 = 0.5 * dot(w_pr_0, w_pr_0) * dx
    E_pr_vec[0] = assemble(E_pr_0)

    # Dual
    u_dl_0, w_dl_0, p_dl_0 = xdual_0.split(deepcopy=True)

    H_dl_vec = np.zeros((n_t + 1, ))
    H_dl_0 = 0.5*dot(u_dl_0, u_dl_0) * dx
    H_dl_vec[0] = assemble(H_dl_0)

    E_dl_vec = np.zeros((n_t + 1,))
    E_dl_0 = 0.5 * dot(w_dl_0, w_dl_0) * dx
    E_dl_vec[0] = assemble(E_dl_0)

    # Incompressibility constraint
    div_u_pr_L2vec = np.zeros((n_t + 1,))
    div_u_dl_L2vec = np.zeros((n_t + 1,))

    divu_pr_0 = div(u_pr_0)**2 * dx
    divu_dl_0 = div(u_dl_0)**2 * dx

    div_u_pr_L2vec[0] = np.sqrt(assemble(divu_pr_0))
    div_u_dl_L2vec[0] = np.sqrt(assemble(divu_dl_0))

    # Compute vorticity at a given point to check correctness of the solver
    u_pr_P_vec = np.zeros((n_t + 1, problem.dimM))
    u_dl_P_vec = np.zeros((n_t + 1, problem.dimM))

    if problem.dimM ==2:
        w_pr_P_vec = np.zeros((n_t + 1, 1))
        w_dl_P_vec = np.zeros((n_t + 1, 1))
    elif problem.dimM ==3:
        w_pr_P_vec = np.zeros((n_t + 1, problem.dimM))
        w_dl_P_vec = np.zeros((n_t + 1, problem.dimM))

    pdyn_pr_P_vec = np.zeros((n_t + 1, ))
    pdyn_dl_P_vec = np.zeros((n_t + 1, ))

    # Only in 3D Helicity
    Hel_pr_vec = np.zeros((n_t + 1,))
    Hel_dl_vec = np.zeros((n_t + 1,))

    if problem.dimM == 2:
        point_P = (1 / 3, 5 / 7)
    elif problem.dimM == 3:
        point_P = (1 / 3, 5 / 7, 3 / 7)

        Hel_pr_0 = dot(u_pr_0, w_dl_0) * dx
        Hel_dl_0 = dot(u_dl_0, w_pr_0) * dx

        Hel_pr_vec[0] = assemble(Hel_pr_0)
        Hel_dl_vec[0] = assemble(Hel_dl_0)

    # Primal
    u_pr_P_vec[0, :] = u_pr_0(point_P)
    w_pr_P_vec[0, :] = w_pr_0(point_P)
    pdyn_pr_P_vec[0] = p_pr_0(point_P) + 0.5*np.dot(u_pr_P_vec[0, :], u_pr_P_vec[0, :])
    # Dual
    u_dl_P_vec[0, :] = u_dl_0(point_P)
    w_dl_P_vec[0, :] = w_dl_0(point_P)
    pdyn_dl_P_vec[0] = p_dl_0(point_P) + 0.5*np.dot(u_dl_P_vec[0, :], u_dl_P_vec[0, :])

    # Exact quantities
    # Energy and Vorticity at P
    H_ex_vec = np.zeros((n_t + 1, ))
    E_ex_vec = np.zeros((n_t + 1,))
    Hel_ex_vec = np.zeros((n_t + 1,))

    u_ex_P_vec = np.zeros((n_t + 1, problem.dimM))

    if problem.dimM == 2:
        w_ex_P_vec = np.zeros((n_t + 1, 1))
    elif problem.dimM == 3:
        w_ex_P_vec = np.zeros((n_t + 1, problem.dimM))

    p_ex_P_vec = np.zeros((n_t + 1,))
    pdyn_ex_P_vec = np.zeros((n_t + 1,))

    if problem.exact == True:
        u_ex_0, w_ex_0, p_ex_0, H_ex_0, E_ex_0, Hel_ex_0 = problem.init_outputs(0)

        H_ex_vec[0] = assemble(H_ex_0)
        E_ex_vec[0] = assemble(E_ex_0)
        if problem.dimM == 3:
            Hel_ex_vec[0] = assemble(Hel_ex_0)
        u_ex_P_vec[0, :] = u_ex_0(point_P)
        w_ex_P_vec[0, :] = w_ex_0(point_P)
        p_ex_P_vec[0] = p_ex_0(point_P)
        pdyn_ex_P_vec[0] = p_ex_P_vec[0] + 0.5*np.dot(u_ex_P_vec[0, :], u_ex_P_vec[0, :])

    # Primal Test and trial functions definition
    chi_primal = TestFunction(V_primal)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_primal)

    x_primal = TrialFunction(V_primal)
    u_pr, w_pr, p_pr = split(x_primal)

    # Static part of the primal A operator
    a1_primal_static = (1/dt) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr)
    a2_primal_static = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_primal_static = - adj_divu_form(chi_p_pr, u_pr)

    A_primal_static = assemble(a1_primal_static + a2_primal_static + a3_primal_static)

    # Primal Test and trial functions definition
    chi_dual = TestFunction(V_dual)
    chi_u_dl, chi_w_dl, chi_p_dl = split(chi_dual)

    x_dual = TrialFunction(V_dual)
    u_dl, w_dl, p_dl = split(x_dual)

    # Static part of the dual A operator
    a1_dual_static = (1/dt) * m_form(chi_u_dl, u_dl) - adj_gradp_form(chi_u_dl, p_dl)
    a2_dual_static = m_form(chi_w_dl, w_dl) - adj_curlu_form(chi_w_dl, u_dl, problem.dimM)
    a3_dual_static = - divu_form(chi_p_dl, u_dl)

    A_dual_static = assemble(a1_dual_static + a2_dual_static + a3_dual_static)

    # Time loop from 1 onwards
    for ii in tqdm(range(1, n_t+1)):

        # Solve dual system for n+1
        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split(deepcopy=True)
        a_dual_dynamic = - 0.5*wcross2_form(chi_u_dl, u_dl, w_pr_n12, problem.dimM)
        A_dual_dynamic = assemble(a_dual_dynamic)
        A_dual = A_dual_static + A_dual_dynamic

        u_dl_n, w_dl_n, p_dl_12n = xdual_n.split(deepcopy=True)

        b1_dual = (1/dt) * m_form(chi_u_dl, u_dl_n) + 0.5*wcross2_form(chi_u_dl, u_dl_n, w_pr_n12, problem.dimM)
        bvec_dual = assemble(b1_dual)
        solve(A_dual, xdual_n1.vector(), bvec_dual, "gmres", "icc")

        u_dl_n1, w_dl_n1, p_dl_n12 = xdual_n1.split(deepcopy=True)


        # Solve primal system at n_32
        a_primal_dynamic = - 0.5*wcross1_form(chi_u_pr, u_pr, w_dl_n1, problem.dimM)
        A_primal_dynamic = assemble(a_primal_dynamic)
        A_primal = A_primal_static + A_primal_dynamic

        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split(deepcopy=True)
        b1_primal = (1/dt) * m_form(chi_u_pr, u_pr_n12) + 0.5*wcross1_form(chi_u_pr, u_pr_n12, w_dl_n1, problem.dimM)
        bvec_primal = assemble(b1_primal)
        solve(A_primal, xprimal_n32.vector(), bvec_primal, "gmres", "icc")

        xprimal_n1.assign(0.5*(xprimal_n12 + xprimal_n32))
        u_pr_n1, w_pr_n1, p_pr_n12 = xprimal_n1.split(deepcopy=True)

        u_pr_n32, w_pr_n32, p_pr_n1 = xprimal_n32.split(deepcopy=True)

        H_dl_n1 = 0.5 * dot(u_dl_n1, u_dl_n1) * dx
        H_dl_vec[ii] = assemble(H_dl_n1)

        H_pr_n1 = 0.5 * dot(u_pr_n1, u_pr_n1) * dx
        H_pr_vec[ii] = assemble(H_pr_n1)

        E_dl_n1 = 0.5 * dot(w_dl_n1, w_dl_n1) * dx
        E_dl_vec[ii] = assemble(E_dl_n1)

        E_pr_n1 = 0.5 * dot(w_pr_n1, w_pr_n1) * dx
        E_pr_vec[ii] = assemble(E_pr_n1)

        divu_pr_n1 = div(u_pr_n1) ** 2 * dx
        divu_dl_n1 = div(u_dl_n1) ** 2 * dx

        div_u_pr_L2vec[ii] = np.sqrt(assemble(divu_pr_n1))
        div_u_dl_L2vec[ii] = np.sqrt(assemble(divu_dl_n1))

        if problem.dimM ==3:
            Hel_pr_n1 = dot(u_pr_n1, w_dl_n1) * dx
            Hel_dl_n1 = dot(u_dl_n1, w_pr_n1) * dx

            Hel_pr_vec[ii] = assemble(Hel_pr_n1)
            Hel_dl_vec[ii] = assemble(Hel_dl_n1)

        u_pr_P_vec[ii, :] = u_pr_n1(point_P)
        w_pr_P_vec[ii, :] = w_pr_n1(point_P)
        pdyn_pr_P_vec[ii] = p_pr_n1(point_P)

        u_dl_P_vec[ii, :] = u_dl_n1(point_P)
        w_dl_P_vec[ii, :] = w_dl_n1(point_P)
        pdyn_dl_P_vec[ii] = p_dl_n12(point_P)

        xdual_n.assign(xdual_n1)
        xprimal_n12.assign(xprimal_n32)

        # Reassign dual, primal, exact

    # Compute exact energy and vorticity
    if problem.exact == True:
        for ii in tqdm(range(1, n_t + 1)):
            t_act = ii * dt
            u_ex_t, w_ex_t, p_ex_t, H_ex_t, E_ex_t, Hel_ex_t = problem.init_outputs(t_act)
            H_ex_vec[ii] = assemble(H_ex_t)
            E_ex_vec[ii] = assemble(E_ex_t)
            if problem.dimM == 3:
                Hel_ex_vec[ii] = assemble(Hel_ex_t)

            u_ex_P_vec[ii, :] = u_ex_t(point_P)
            w_ex_P_vec[ii, :] = w_ex_t(point_P)
            p_ex_P_vec[ii] = p_ex_t(point_P)
            pdyn_ex_P_vec[ii] = p_ex_P_vec[ii] + 0.5 * np.dot(u_ex_P_vec[ii, :], u_ex_P_vec[ii, :])

    dict_res = {"tspan_int": tvec_int, "tspan_stag": tvec_stag, \
                "energy_ex": H_ex_vec, "energy_pr": H_pr_vec, "energy_dl": H_dl_vec, \
                "enstrophy_ex": E_ex_vec, "enstrophy_pr": E_pr_vec, "enstrophy_dl": E_dl_vec, \
                "helicity_ex": Hel_ex_vec, "helicity_pr": Hel_pr_vec, "helicity_dl": Hel_dl_vec, \
                "uP_ex": u_ex_P_vec, "uP_pr": u_pr_P_vec, "uP_dl": u_dl_P_vec, \
                "wP_ex": w_ex_P_vec, "wP_pr": w_pr_P_vec, "wP_dl": w_dl_P_vec, \
                "pdynP_ex": pdyn_ex_P_vec, "pdynP_pr": pdyn_pr_P_vec, "pdynP_dl": pdyn_dl_P_vec, \
                "divu_pr_L2" : div_u_pr_L2vec, "divu_dl_L2" : div_u_dl_L2vec}

    return dict_res
