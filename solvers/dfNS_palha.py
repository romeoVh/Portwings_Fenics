from fenics import *
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from vedo.dolfin import plot

def explicit_step_primal_incompressible(dt_0, problem, x_n, wT_n, V_pr):
    u_n = x_n[0]
    w_n = x_n[1]

    chi_pr = TestFunction(V_pr)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_pr)

    x_pr = TrialFunction(V_pr)
    u_pr, w_pr, p_pr = split(x_pr)

    a1_form_vel = (1 / dt_0) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr)
    a2_form_vor = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_form_p = - adj_divu_form(chi_p_pr, u_pr)
    A0_pr = assemble(a1_form_vel + a2_form_vor + a3_form_p)

    b1_form_vel = (1 / dt_0) * m_form(chi_u_pr, u_n) + wcross1_form(chi_u_pr, u_n, wT_n, problem.dimM) \
                 + adj_curlw_form(chi_u_pr, w_n, problem.dimM, problem.Re)
    b0_pr = assemble(b1_form_vel)

    x_sol = Function(V_pr)

    solve(A0_pr, x_sol.vector(), b0_pr, "gmres", "amg")

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
    tvec_stag = np.linspace(float(dt)/2, float(dt)*(n_t + 1/2), n_t+1)

    u_pr_0, w_pr_0, p_pr_0 = xprimal_0.split(deepcopy=True)
    w_dl_0 = xdual_0.split(deepcopy=True)[1]

    x_0 = [u_pr_0, w_pr_0, p_pr_0]

    xprimal_n12 = explicit_step_primal_incompressible(dt / 2, problem, x_0, w_dl_0, V_primal)

    print("Explicit step solved")

    u_pr_12, w_pr_12, p_pr_init = xprimal_n12.split(deepcopy=True)


    # Primal intermediate variables
    xprimal_n32 = Function(V_primal, name="u, w at n+3/2, p at n+1")

    # xprimal_n1 = Function(V_primal, name="u, w at n+1, p at n+1/2")


    # Dual intermediate variables
    xdual_n = Function(V_dual, name="uT, wT at n, pT at n-1/2")
    xdual_n.assign(xdual_0)

    xdual_n1 = Function(V_dual, name="u, w at n+1, p at n+1/2")

    # Kinetic energy definition
    # Primal
    H_pr_vec = np.zeros((n_t + 1))
    H_pr_12 = 0.5*dot(u_pr_12, u_pr_12) * dx
    H_pr_vec[0] = assemble(H_pr_12)

    # Dual
    u_dl_0, w_dl_0, p_dl_0 = xdual_0.split(deepcopy=True)
    H_dl_vec = np.zeros((n_t + 1))
    H_dl_0 = 0.5*dot(u_dl_0, u_dl_0) * dx
    H_dl_vec[0] = assemble(H_dl_0)

    # Compute vorticity at a given point to check correctness of the solver
    w_pr_P_vec = np.zeros((n_t + 1))
    w_dl_P_vec = np.zeros((n_t + 1))
    if problem.dimM == 2:
        point_P = (1/3, 5/7)
        # Primal
        w_pr_P_vec[0] = w_pr_12(point_P)
        # Dual
        w_dl_P_vec[0] = w_dl_0(point_P)
    else:
        point_P = (1 / 3, 5 / 7, 3/7)


    # Exact quantities
    # Energy and Vorticity at P
    H_ex_vec = np.zeros((n_t + 1))
    w_ex_P_vec = np.zeros((n_t + 1))

    if problem.exact == True:
        u_ex_0, w_ex_0, p_ex_0, H_ex_0, E_ex_0, Ch_ex_0 = problem.init_outputs(0)
        H_ex_vec[0] = assemble(H_ex_0)
        w_ex_P_vec[0] = w_ex_0(point_P)

    # Primal Test and trial functions definition
    chi_primal = TestFunction(V_primal)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_primal)

    x_primal = TrialFunction(V_primal)
    u_pr, w_pr, p_pr = split(x_primal)

    # Static part of the primal A operator
    a1_primal_static = (1/dt) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr) \
                       - 0.5*adj_curlw_form(chi_u_pr, w_pr, problem.dimM, problem.Re)
    a2_primal_static = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_primal_static = - adj_divu_form(chi_p_pr, u_pr)

    A_primal_static = assemble(a1_primal_static + a2_primal_static + a3_primal_static)

    # Primal Test and trial functions definition
    chi_dual = TestFunction(V_dual)
    chi_u_dl, chi_w_dl, chi_p_dl = split(chi_dual)

    x_dual = TrialFunction(V_dual)
    u_dl, w_dl, p_dl = split(x_dual)

    # Static part of the dual A operator
    a1_dual_static = (1/dt) * m_form(chi_u_dl, u_dl) - adj_gradp_form(chi_u_dl, p_dl) \
                       - 0.5 * curlw_form(chi_u_dl, w_dl, problem.dimM, problem.Re)
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

        b1_dual = (1/dt) * m_form(chi_u_dl, u_dl_n) + 0.5*wcross2_form(chi_u_dl, u_dl_n, w_pr_n12, problem.dimM) \
                  + 0.5*curlw_form(chi_u_dl, w_dl_n, problem.dimM, problem.Re)
        bvec_dual = assemble(b1_dual)
        solve(A_dual, xdual_n1.vector(), bvec_dual, "gmres", "amg")

        u_dl_n1, w_dl_n1, p_dl_n12 = xdual_n1.split(deepcopy=True)
        H_dl_n1 = 0.5 * dot(u_dl_n1, u_dl_n1) * dx
        H_dl_vec[ii] = assemble(H_dl_n1)

        # Solve primal system at n_32
        a_primal_dynamic = - 0.5*wcross1_form(chi_u_pr, u_pr, w_dl_n1, problem.dimM)
        A_primal_dynamic = assemble(a_primal_dynamic)
        A_primal = A_primal_static + A_primal_dynamic

        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split(deepcopy=True)
        b1_primal = (1/dt) * m_form(chi_u_pr, u_pr_n12) + 0.5*wcross1_form(chi_u_pr, u_pr_n12, w_dl_n1, problem.dimM) \
                    + 0.5*adj_curlw_form(chi_u_pr, w_pr_n12, problem.dimM, problem.Re)
        bvec_primal = assemble(b1_primal)
        solve(A_primal, xprimal_n32.vector(), bvec_primal, "gmres", "amg")

        # xprimal_n1.assign(0.5*(xprimal_n12 + xprimal_n32))
        # u_pr_n1, w_pr_n1, p_pr_n1 = xprimal_n1.split(deepcopy=True)
        # H_pr_n1 = 0.5 * dot(u_pr_n1, u_pr_n1) * dx

        u_pr_n32, w_pr_n32, p_pr_n1 = xprimal_n32.split(deepcopy=True)
        H_pr_n32 = 0.5 * dot(u_pr_n32, u_pr_n32) * dx
        H_pr_vec[ii] = assemble(H_pr_n32)

        xdual_n.assign(xdual_n1)
        xprimal_n12.assign(xprimal_n32)

        if problem.dimM==2:
            w_dl_P_vec[ii] = w_dl_n1(point_P)
            w_pr_P_vec[ii] = w_pr_n32(point_P)

        # Reassign dual, primal, exact

    # Compute exact energy and vorticity
    if problem.exact == True:
        for ii in tqdm(range(1, n_t + 1)):
            t_act = ii * dt
            u_ex_t, w_ex_t, p_ex_t, H_ex_t, E_ex_t, Ch_ex_t = problem.init_outputs(t_act)
            H_ex_vec[ii] = assemble(H_ex_t)
            w_ex_P_vec[ii] = w_ex_t(point_P)

    return tvec_int, tvec_stag,  H_pr_vec, H_dl_vec, H_ex_vec, w_pr_P_vec, w_dl_P_vec, w_ex_P_vec

# Common forms
def m_form(chi_i, alpha_i):
    form = inner(chi_i,alpha_i) * dx
    return form

def curl2D(v):
    return v[1].dx(0) - v[0].dx(1)

def rot2D(w):
    return as_vector((w.dx(1), -w.dx(0)))


# Primal system forms
def wcross1_form(chi_1, v_1, wT_n2, dimM):
    if dimM==3:
        form = inner(chi_1,cross(v_1, wT_n2)) * dx
    elif dimM==2:
        form = wT_n2*dot(chi_1, as_vector([v_1[1], -v_1[0]])) * dx
    return form


def gradp_form(chi_1, p_0):
    form = -inner(chi_1,grad(p_0)) * dx
    return form

def adj_curlw_form(chi_1, w_2, dimM, Re):
    # if dimM==3:
    #     form = -1./Re*inner(curl(chi_1),w_2) * dx
    # elif dimM==2:
    #     form = -1./Re*dot(curl2D(chi_1),w_2) * dx
    # return form
    return 0

def adj_divu_form(chi_0, v_1):
    form = inner(grad(chi_0),v_1) * dx
    return form

def curlu_form(chi_2, v_1, dimM):
    if dimM==3:
        form = inner(chi_2,curl(v_1)) * dx
    elif dimM==2:
        form = dot(chi_2,curl2D(v_1)) * dx
    return form

def tantrace_w_form(chi_1, wT_n2, n_vec, dimM, Re):
    if dimM==3:
        form = 1./Re*dot(cross(chi_1,wT_n2),n_vec) * ds
    elif dimM==2:
        form = 1./Re*wT_n2*dot(as_vector((chi_1[1], -chi_1[0])), n_vec) * ds
    return form

def normtrace_v_form(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms
def wcross2_form(chi_2, vT_2, w_2, dimM):
    if dimM==3:
        form = inner(chi_2,cross(vT_2, w_2)) * dx
    elif dimM==2:
        form = w_2*dot(chi_2, as_vector([vT_2[1], -vT_2[0]])) * dx

    return form

def adj_gradp_form(chi_2,pT_3):
    form = inner(div(chi_2),pT_3) * dx
    return form

def curlw_form(chi_2,wT_1,dimM, Re):
    # if dimM == 3:
    #     form = -1./Re*inner(chi_2, curl(wT_1)) * dx
    # elif dimM == 2:
    #     form = -1./Re*dot(chi_2, rot2D(wT_1)) * dx
    #     # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    # return form
    return 0

def divu_form(chi_3, vT_2):
    form = -inner(chi_3, div(vT_2)) * dx
    return form

def adj_curlu_form(chi_1, vT_2, dimM):
    if dimM == 3:
        form = inner(curl(chi_1), vT_2) * dx
    elif dimM == 2:
        form = dot(rot2D(chi_1), vT_2) * dx
    return form

def dirtrace_p_form(chi_2, p_0, n_vec):
    form = -p_0*dot(chi_2,n_vec) * ds
    return form

def tantrace_v_form(chi_1, v_1, n_vec, dimM):
    if dimM == 3:
        form = -dot(cross(chi_1, v_1), n_vec) * ds
    elif dimM == 2:
        form = chi_1*dot(as_vector((v_1[1], -v_1[0])), n_vec) * ds
    return form

