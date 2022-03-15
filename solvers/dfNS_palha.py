from fenics import *
from time import time
from tqdm import tqdm
import numpy as np

def explicit_step_primal(dt_0, problem, x_n, V_vel, V_vor):
    v_n = x_n[0]
    w_n = x_n[1]
    p_n = x_n[2]

    chi_1 = TestFunction(V_vel)
    u_1 = TrialFunction(V_vel)

    a_form_vel = 1 / dt_0 * m_form(chi_1, u_1)
    A_vel = assemble(a_form_vel)

    b_form_vel = (1 / dt_0) * m_form(chi_1, v_n) + wcross1_form(problem.dimM, chi_1, v_n, w_n) \
                 + gradp_form(chi_1, p_n) + adj_curlw_form(problem.dimM, chi_1, w_n, problem.Re)
    b_vel = assemble(b_form_vel)

    v_sol = Function(V_vel)
    solve(A_vel, v_sol.vector(), b_vel)

    chi_w = TestFunction(V_vor)
    w_trial = TrialFunction(V_vor)

    a_form_vor = m_form(chi_w, w_trial)
    A_vor = assemble(a_form_vor)

    b_form_vor = curlu_form(problem.dimM, chi_w, v_sol)
    b_vor = assemble(b_form_vor)

    w_sol = Function(V_vor)

    solve(A_vor, w_sol.vector(), b_vor)

    return v_sol, w_sol


def compute_sol(problem, pol_deg, n_t, t_fin=1):
    # Implementation of the dual field formulation for periodic navier stokes
    mesh = problem.mesh
    problem.init_mesh()

    ufl_cell = mesh.ufl_cell()
    P_1 = FiniteElement("N1curl", ufl_cell, pol_deg)
    P_0 = FiniteElement("CG", ufl_cell, pol_deg)
    if problem.dimM == 3:
        P_2 = FiniteElement("RT", ufl_cell, pol_deg)
    elif problem.dimM == 2:
        P_2 = FiniteElement("DG", ufl_cell, pol_deg - 1)

    # Define dual mixed elements
    PT_n1 = FiniteElement("RT", ufl_cell, pol_deg)
    PT_n = FiniteElement("DG", ufl_cell, pol_deg - 1)
    if problem.dimM == 3:
        PT_n2 = FiniteElement("N1curl", ufl_cell, pol_deg)
    elif problem.dimM == 2:
        PT_n2 = FiniteElement("CG", ufl_cell, pol_deg)

    P_primal = MixedElement([P_1, P_2, P_0])
    P_dual = MixedElement([PT_n1, PT_n2, PT_n])

    # Define function spaces
    V_1 = FunctionSpace(mesh, P_1)
    V_2 = FunctionSpace(mesh, P_2)
    V_0 = FunctionSpace(mesh, P_0)
    V_primal = FunctionSpace(mesh, P_primal)  # V_1 x V_2 x V_0

    VT_n1 = FunctionSpace(mesh, PT_n1)
    VT_n2 = FunctionSpace(mesh, PT_n2)
    VT_n = FunctionSpace(mesh, PT_n)
    V_dual = FunctionSpace(mesh, P_dual)  # VT_n-1 x VT_n-2 x VT_n
    print("Function Space dimensions, Primal - Dual: ", [V_primal.dim(), V_dual.dim()])

    # Define Function assigners
    fa_primal = FunctionAssigner(V_primal, [V_1, V_2, V_0])
    fa_dual = FunctionAssigner(V_dual, [VT_n1, VT_n2, VT_n])
    # Set initial condition at t=0
    x_init = Function(V_primal, name="x_0 primal")
    xT_init = Function(V_dual, name="x_0 dual")

    fa_primal.assign(x_init, problem.initial_conditions(V_1, V_2, V_0))
    fa_dual.assign(xT_init, problem.initial_conditions(VT_n1, VT_n2, VT_n))

    dt = Constant(t_fin / n_t)

    v_0, w_0, p_0 = x_init.split(deepcopy=True)
    x_0 = [v_0, w_0, p_0]
    v_half, w_half = explicit_step_primal(dt / 2, problem, x_0, V_1, V_2)

    tvec_dual = np.linspace(0, n_t * float(dt), 1 + n_t)

    return 1

# Common forms
def m_form(chi_i, alpha_i):
    form = inner(chi_i,alpha_i) * dx
    return form


# Primal system forms
def wcross1_form(dimM,chi_1, v_1, wT_n2):
    if dimM==3:
        form = inner(chi_1,cross(v_1, wT_n2)) * dx
    elif dimM==2:
        form = dot(wT_n2, v_1[1]*chi_1[0] - v_1[0]*chi_1[1]) * dx
    return form

def gradp_form(chi_1, p_0):
    form = -inner(chi_1,grad(p_0)) * dx
    return form

def adj_curlw_form(dimM,chi_1, w_2, Re):
    if dimM==3:
        form = -1./Re*inner(curl(chi_1),w_2) * dx
    elif dimM==2:
        form = -1./Re*dot(curl2D(chi_1),w_2) * dx
    return form

def adj_divu_form(chi_0, v_1):
    form = -inner(grad(chi_0),v_1) * dx
    return form

def curlu_form(dimM,chi_2, v_1):
    if dimM==3:
        form = inner(chi_2,curl(v_1)) * dx
    elif dimM==2:
        form = dot(chi_2,curl2D(v_1)) * dx
    return form

def tantrace_w_form(dimM, chi_1, wT_n2, n_vec, Re):
    if dimM==3:
        form = 1./Re*dot(cross(chi_1,wT_n2),n_vec) * ds
    elif dimM==2:
        form = 1./Re*wT_n2*dot(as_vector((chi_1[1], -chi_1[0])), n_vec) * ds
    return form

def normtrace_v_form(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms
def wcross2_form(dimM,chi_2, vT_2, w_2):
    if dimM==3:
        form = inner(chi_2,cross(vT_2, w_2)) * dx
    elif dimM==2:
        form = dot(w_2, vT_2[1]*chi_2[0] - vT_2[0]*chi_2[1]) * dx
    return form

def adj_gradp_form(chi_2,pT_3):
    form = inner(div(chi_2),pT_3) * dx
    return form

def curlw_form(dimM,chi_2,wT_1,Re):
    if dimM == 3:
        form = -1./Re*inner(chi_2, curl(wT_1)) * dx
    elif dimM == 2:
        form = -1./Re*dot(chi_2, rot2D(wT_1)) * dx
        # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    return form

def divu_form(chi_3, vT_2):
    form = inner(chi_3, div(vT_2)) * dx
    return form

def adj_curlu_form(dimM,chi_1, vT_2):
    if dimM == 3:
        form = inner(curl(chi_1), vT_2) * dx
    elif dimM == 2:
        form = dot(rot2D(chi_1), vT_2) * dx
    return form

def dirtrace_p_form(chi_2, p_0, n_vec):
    form = -p_0*dot(chi_2,n_vec) * ds
    return form

def tantrace_v_form(dimM,chi_1, v_1, n_vec):
    if dimM == 3:
        form = -dot(cross(chi_1, v_1), n_vec) * ds
    elif dimM == 2:
        form = chi_1*dot(as_vector((v_1[1], -v_1[0])), n_vec) * ds
    return form

def curl2D(v):
    return v[1].dx(0) - v[0].dx(1)

def rot2D(w):
    return as_vector((w.dx(1), -w.dx(0)))
