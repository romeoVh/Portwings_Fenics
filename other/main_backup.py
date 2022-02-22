from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def exact_solution(time_str = 't'):
    import sympy as sym
    # Spatial constants
    om_x = 1.0
    om_y = 1.0
    om_z = 1.0
    phi_x = 0.0
    phi_y = 0.0
    phi_z = 0.0
    # Time constants
    phi_t = 0.0
    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)

    # Mesh coordinates
    x,y,z = sym.symbols('x[0],x[1],x[2]')
    t = sym.symbols(time_str)

    # functions f and g
    f = 2.0 * sym.sin(om_t * t + phi_t) + 3.0 * sym.cos(om_t * t + phi_t)
    g = sym.cos(om_x * x + phi_x) * sym.sin(om_y * y + phi_y) * sym.sin(om_z * z + phi_z)
    # Derivatives of f and g
    _dt_f = sym.diff(f,t)
    _dx_g = -sym.diff(g, x)
    _dy_g = -sym.diff(g, y)
    _dz_g = -sym.diff(g, z)

    # Exact p and q expressions
    p_ex = g*_dt_f
    q_ex_1 = _dx_g * f
    q_ex_2 = _dy_g * f
    q_ex_3 = _dz_g * f

    # Substitute t=t_i
    #p_ex = p_ex.subs(t,t_i)
    #q_ex_1 = q_ex_1.subs(t,t_i)
    #q_ex_2 = q_ex_2.subs(t,t_i)
    #q_ex_3 = q_ex_3.subs(t,t_i)
    #print("------- At t = ",t_i)
    #print(p_ex)
    #print(q_ex_1)
    #print(q_ex_2)
    #print(q_ex_3)
    #print("-------------------")

    return p_ex,[q_ex_1,q_ex_2,q_ex_3]

def convert_sym_to_expr(t_i, p_ex,q_ex,degree = 6):
    # Convert from Sympy to Expression
    # THis method has an extra feature that allows having expressions with two time variables
    p_ex_code = convert_sym('p_ex', p_ex, False)
    _p_ex = Expression(p_ex_code, degree=degree, t=t_i[0],t_mid =t_i[1]) # Scalar valued expression

    q_ex_1_code = convert_sym('q_ex_1', q_ex[0], False)
    q_ex_2_code = convert_sym('q_ex_2', q_ex[1], False)
    q_ex_3_code = convert_sym('q_ex_3', q_ex[2], False)
    _q_ex = Expression((q_ex_1_code,q_ex_2_code,q_ex_3_code), degree=degree, t=t_i[0],t_mid =t_i[1]) # Vector valued expression

    return _p_ex,_q_ex

def convert_sym(name, fun,show_func=False):
    import sympy as sym
    fun_code = sym.printing.ccode(fun)
    if(show_func):
        print('Code of ',name,' is: ', fun_code)
    return fun_code

def  split_functions(e_32,e_10,show_dim = False):
    # Without deepcopy gives wrong dimensions and gives an eeror when assigning n+! to n in foor loop !
    p_3, q_2 = e_32.split(deepcopy=True)
    q_1, p_0 = e_10.split(deepcopy=True)

    if(show_dim):
        print("Dimensions before splitting: ", len(e_32.vector()),len(e_10.vector()))
        print("Dimensions after splitting: ", len(p_3.vector()),len(q_2.vector()), len(q_1.vector()),len(p_0.vector()))
    return p_3,q_2,q_1,p_0

def compute_L2_error(p_ex,q_ex,p_3_n,q_2_n,q_1_n,p_0_n):
    err_p_3 = errornorm(p_ex, p_3_n, norm_type="L2")
    err_q_2 = errornorm(q_ex, q_2_n, norm_type="L2")
    err_q_1 = errornorm(q_ex, q_1_n, norm_type="L2")
    err_p_0= errornorm(p_ex, p_0_n, norm_type="L2")
    return np.array([err_p_3,err_q_2,err_q_1,err_p_0])

def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the wave equation with the dual field method

            Parameters:
            n_el: number of elements for the discretization
            n_t: number of time instants
            deg: polynomial degree for finite
            Returns:
            some plots

           """
    def m_form32(v_3, p_3, v_2, q_2):
        m_form = inner(v_3, p_3) * dx + inner(v_2, q_2) * dx
        return m_form

    def m_form10(v_1, q_1, v_0, p_0):
        m_form = inner(v_1, q_1) * dx + inner(v_0, p_0) * dx
        return m_form

    def j_form32(v_3, p_3, v_2, q_2):
        j_form = -dot(v_3, div(q_2)) * dx + dot(div(v_2), p_3) * dx
        return j_form

    def j_form10(v_1, q_1, v_0, p_0):
        j_form = -dot(v_1, grad(p_0)) * dx + dot(grad(v_0), q_1) * dx
        return j_form

    def bdflow32(v_2, u_0,n_vec):
        b_form = -dot(v_2, n_vec) * u_0 * ds
        return b_form

    def bdflow10(v_0, u_2,n_vec):
        b_form = v_0 * dot(u_2, n_vec) * ds
        return b_form

    # Define Domain
    mesh = BoxMesh(Point(0,0,0), Point(1,0.5,0.5),n_el,n_el,n_el)
    mesh.init()
    n_ver = FacetNormal(mesh)
    dt = Constant(t_fin / n_t)
    t_vec = np.linspace(0, t_fin, n_t+1)

    print("Mesh cells,faces,edges,vertices: ",[mesh.num_cells(),mesh.num_faces(),mesh.num_edges(),mesh.num_vertices()])
    #plot(mesh,alpha=0.7)
    #plt.show()

    # Define mixed elements

    P_0 = FiniteElement("CG", tetrahedron, deg)
    P_1 = FiniteElement("N1curl", tetrahedron, deg)
    P_2 = FiniteElement("RT", tetrahedron, deg)
    P_3 = FiniteElement("DG", tetrahedron, deg - 1)

    P_32 = MixedElement([P_3, P_2])
    P_10 = MixedElement([P_1, P_0])

    # Define function spaces
    V_3 = FunctionSpace(mesh, P_3)
    V_2 = FunctionSpace(mesh, P_2)
    V_1 = FunctionSpace(mesh, P_1)
    V_0 = FunctionSpace(mesh, P_0)
    V_32 = FunctionSpace(mesh,P_32)
    V_10 = FunctionSpace(mesh, P_10)
    print("Function Space dimensions: ",[[V_3.dim(),V_2.dim()], [V_1.dim(),V_0.dim()]])

    # Define Test functions and split
    v_32 = TestFunction(V_32)
    v_10 = TestFunction(V_10)
    v_3, v_2 = split(v_32)
    v_1, v_0 = split(v_10)

    # Define Unknown Trial functions
    e_32 = TrialFunction(V_32)
    e_10 = TrialFunction(V_10)
    p_3, q_2 = split(e_32) # State variables
    q_1, p_0 = split(e_10) # Co-state variables

    # Define time variables
    t_c = Constant(0.0) # Current time step
    t_1_c = Constant(dt) # Next time step
    t_mid_c = Constant(float(dt)/2.0) # Mid time step

    # Define exact solution expressions at different time steps
    _p_ex, _q_ex = exact_solution()
    #print(_p_ex)
    p_ex, q_ex = convert_sym_to_expr([t_c,0], _p_ex,_q_ex)                  # Time step n
    p_ex_1, q_ex_1 = convert_sym_to_expr([t_1_c,0],_p_ex,_q_ex)             # Time step n+1
    p_ex_mid, q_ex_mid = convert_sym_to_expr([t_mid_c, 0], _p_ex, _q_ex)    # Time step n+1/2

    # Alternative approach for Time step n+1
    #_p_ex_1, _q_ex_1 = exact_solution('t_mid')
    #print(_p_ex_1)
    # p_ex_1, q_ex_1 = convert_sym_to_expr([0,t_1_c],_p_ex_1,_q_ex_1)

    # Alternative approach for Time step n+1/2
    #_p_ex_mid = (_p_ex/2.0 + _p_ex_1/2.0)
    #_q_ex_mid = [_q_ex[0]/2.0+_q_ex_1[0]/2.0,_q_ex[1]/2.0+_q_ex_1[1]/2.0,_q_ex[2]/2.0+_q_ex_1[2]/2.0]
    #print(_p_ex_mid)
    #p_ex_mid, q_ex_mid = convert_sym_to_expr([t_c,t_1_c],_p_ex_mid, _q_ex_mid)

    # Set initial condition at t=0
    e_32_init = Function(V_32, name="e_32 initial")
    e_10_init = Function(V_10, name="e_10 initial")

    p_3_init = interpolate(p_ex, V_3)
    q_2_init = interpolate(q_ex, V_2)
    q_1_init = interpolate(q_ex, V_1)
    p_0_init = interpolate(p_ex, V_0)

    fa_32 = FunctionAssigner(V_32, [V_3, V_2])
    fa_10 = FunctionAssigner(V_10, [V_1, V_0])

    fa_32.assign(e_32_init, [p_3_init, q_2_init])
    fa_10.assign(e_10_init, [q_1_init, p_0_init])

    _p_3_init, _q_2_init,_q_1_init, _p_0_init = split_functions(e_32_init,e_10_init)

    # Set boundary condition
    # Why is this using t_1_c and not t_c ?
    bc_D = bc_D_nat = bc_N = bc_N_nat = None
    if bd_cond == "D":
        bc_D = [DirichletBC(V_10.sub(1), p_ex_1, "on_boundary")]
        bc_N_nat = [DirichletBC(V_32.sub(1), q_ex_1, "on_boundary")]
    elif bd_cond == "N":
        bc_N = [DirichletBC(V_32.sub(1), q_ex_1, "on_boundary")]
        bc_D_nat = [DirichletBC(V_10.sub(1), p_ex_1, "on_boundary")]

    # Define variables for solutions of p and q at:
    # Time step n
    e_32_n = Function(V_32, name="e_32 n")
    e_10_n = Function(V_10, name="e_10 n")
    e_32_n.assign(e_32_init)
    e_10_n.assign(e_10_init)
    p_3_n, q_2_n,q_1_n, p_0_n = split_functions(e_32_n,e_10_n,False)

    # Time step n+1
    e_32_n1 = Function(V_32, name="e_32 n+1")
    e_10_n1 = Function(V_10, name="e_10 n+1")
    p_3_n1, q_2_n1,q_1_n1, p_0_n1 = split_functions(e_32_n1,e_10_n1,False)
    num_dof = np.sum(len(e_32_n1.vector())+len(e_10_n1.vector()))

    # Define Error Arrays
    errL2_vec = np.zeros((1 + n_t, 4))  # p_3,q_2,q_1,p_0

    # Compute Initial L2 Error # SANITY CHECK
    #errL2_vec[0] = compute_L2_error(p_ex,q_ex,p_3_init,q_2_init,q_1_init,p_0_init)
    #print("Initial Error: ",errL2_vec[0])
    #errL2_vec[0] = compute_L2_error(p_ex, q_ex, _p_3_init, _q_2_init, _q_1_init, _p_0_init)
    #print("Initial Error: ", errL2_vec[0])
    errL2_vec[0] = compute_L2_error(p_ex, q_ex, p_3_n, q_2_n, q_1_n, p_0_n)
    print("Initial Error: ",np.sum(errL2_vec[0]), errL2_vec[0])

    # Define total energy forms
    H_ex_n = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(q_ex, q_ex) * dx(domain=mesh))
    H_32_n = 0.5 * (inner(p_3_n, p_3_n) * dx + inner(q_2_n, q_2_n) * dx)
    H_10_n = 0.5 * (inner(p_0_n, p_0_n) * dx + inner(q_1_n, q_1_n) * dx)
    H_3210_n = 0.5 * (dot(p_0_n, p_3_n) * dx + dot(q_2_n, q_1_n) * dx)
    # Define total energy arrays
    H_vec = np.zeros((1 + n_t,4)) # [H_ex_n,H_32_n,H_10_n,H_3210_n]
    H_vec[0,:] = np.array([assemble(H_ex_n),assemble(H_32_n),assemble(H_10_n),assemble(H_3210_n)])
    print("Initial Energy: ", H_vec[0,:])
    # Time independent bilinear forms
    a_form10 = m_form10(v_1, q_1, v_0, p_0) - 0.5*dt*j_form10(v_1, q_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, q_2) - 0.5*dt*j_form32(v_3, p_3, v_2, q_2)
    A_10 = assemble(a_form10)
    A_32 = assemble(a_form32)

    print("Computation of the solution with # of elements: " + str(n_el) + ", # of time steps: " + str(n_t)+ ", # of DOFs: " + str(num_dof) + ", and deg: " + str(deg))
    if not (bc_D is None): print("Applying Dirichlet B.C")
    if not (bc_N is None): print("Applying Neumann B.C")
    print("==============")

    # Integrate in time
    ii = 1


    for t in tqdm(t_vec[1:]):
        #print("Time is now [t,t_c,t_mid_c,t_1_c]",t,float(t_c),float(t_mid_c),float(t_1_c)) # For Debug
        input_2 = -interpolate(q_ex_mid, V_2)
        input_0 = interpolate(p_ex_mid, V_0)
        #input_2 = 0.5 * (q_2_n + q_2_n1)
        #input_0 = 0.5 * (p_0_n + p_0_n1) # Alternative with larger error

        # Integration of 32 system (Dirichlet natural)
        b_form32 = m_form32(v_3, p_3_n, v_2, q_2_n) + dt * (0.5 * j_form32(v_3, p_3_n, v_2, q_2_n) + bdflow32(v_2, input_0,n_ver))
        b_32 = assemble(b_form32)
        if not(bc_N is None): [bc.apply(A_32, b_32) for bc in bc_N]
        solve(A_32, e_32_n1.vector(), b_32)
        # Integration of 10 system (Neumann natural)
        b_form10 = m_form10(v_1, q_1_n, v_0, p_0_n) + dt * (0.5 * j_form10(v_1, q_1_n, v_0, p_0_n) + bdflow10(v_0, input_2,n_ver))
        b_10 = assemble(b_form10)
        if not(bc_D is None): [bc.apply(A_10, b_10) for bc in bc_D]
        solve(A_10, e_10_n1.vector(), b_10)

        # Update previous solution
        e_32_n.assign(e_32_n1)
        e_10_n.assign(e_10_n1)
        p_3_n1, q_2_n1, q_1_n1, p_0_n1 = split_functions(e_32_n1, e_10_n1, False)
        #p_3_n, q_2_n, q_1_n, p_0_n = split_functions(e_32_n, e_10_n)
        p_3_n.assign(p_3_n1)
        q_2_n.assign(q_2_n1)
        q_1_n.assign(q_1_n1)
        p_0_n.assign(p_0_n1)

        # Advance time step
        t_c.assign(t)
        t_1_c.assign(t + float(dt))
        t_mid_c.assign(t+ float(dt)/2.0)

        # Compute L2 error at "new" time step n+1
        errL2_vec[ii] = compute_L2_error(p_ex, q_ex, p_3_n, q_2_n, q_1_n, p_0_n)
        #print("Error ",errL2_vec[ii])
        # Compute total energy
        H_vec[ii,:] = np.array([assemble(H_ex_n),assemble(H_32_n),assemble(H_10_n),assemble(H_3210_n)])
        #print("Energies ", H_vec[ii])
        ii = ii+1


    plt.bar(t_vec,np.sum(errL2_vec,axis=1),width = float(dt)/2)
    plt.figure()
    plt.plot(t_vec,H_vec[:,0:3])
    plt.legend(['H_ex','H_32','H_10','H_3210'])
    plt.show()


if __name__ == '__main__':
    print('--Besm Allah Al-Rahman Al-Rahim--')


    n_time = 50
    t_fin = 5

    dt = t_fin / n_time

    pol_deg = 1
    n_elem = 1
    #results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond="D")
    n_elem_arr = [2]
    pol_deg_arr = [1] # Why can't go above two ?
    [compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond="X") for n_elem in n_elem_arr for pol_deg in pol_deg_arr]




# Drafts
    #f_code = convert_sym('f',f,False)
    #g_code = convert_sym('g',g,True)

    #dt_f = om_t * (2 * sym.cos(om_t * t + phi_t) - 3 * sym.sin(om_t * t + phi_t))  # diff(dft_t, t)
    #dt_f_code = convert_sym('dt_f', dt_f, False)
    #_dt_f_code = convert_sym('dt_f Symbolic', _dt_f, False)

    #dgxyz_x = - om_x * sym.sin(om_x * x + phi_x) * sym.sin(om_y * y + phi_y) * sym.sin(om_z * z + phi_z)
    #dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
    #dgxyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)
    #grad_gxyz = as_vector([dgxyz_x,dgxyz_y,dgxyz_z])  # grad(gxyz)


    #dx_g_code = convert_sym('dx_g', dx_g, False)
    #_dx_g_code = convert_sym('dx_g Symbolic', _dx_g, False)
    #_dy_g_code = convert_sym('dy_g Symbolic', _dy_g, False)
    #_dz_g_code = convert_sym('dz_g Symbolic', _dz_g, False)
    #grad_g = sym.Matrix([_dx_g, _dy_g, _dz_g])  # grad(gxyz)


    #ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t) # ASK ?
    #dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    #q_ex = grad_g * f  # Is there a minus sign missing ?
    #q_ex = q_ex.subs(t, 0)
    #q_ex_code = convert_sym('q_ex', q_ex, True)
    #_q_ex = Expression(q_ex_code,degree = 3)
    #print(_q_ex)


    #q_ex = grad_gxyz * ft # Is there a minus sign missing ?


    #p_ex_1 = gxyz * dft_1
    #q_ex_1 = grad_gxyz * ft_1 # Is there a minus sign missing ?