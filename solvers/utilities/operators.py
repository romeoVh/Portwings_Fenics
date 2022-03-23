from fenics import *


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
    if dimM==3:
        form = -1./Re*inner(curl(chi_1),w_2) * dx
    elif dimM==2:
        form = -1./Re*dot(curl2D(chi_1),w_2) * dx
    return form
    # return 0

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
    if dimM == 3:
        form = -1./Re*inner(chi_2, curl(wT_1)) * dx
    elif dimM == 2:
        form = -1./Re*dot(chi_2, rot2D(wT_1)) * dx
        # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    return form
    # return 0

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

