from problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym


class ExactEuler2D(ProblemBase):
    "2D Taylor Green problem."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        self.mesh = UnitSquareMesh(self.n_el, self.n_el, "crossed")
        self.init_mesh()
        self.structured_time_grid()

        # Periodic Problem
        self.periodic = True
        # Solution exact
        self.exact = True


    def exact_solution(self, time_str='t'):
        from sympy import exp as Exp
        from sympy import sin as Sin
        from sympy import cos as Cos

        # Mesh coordinates
        x, y = sym.symbols('x[0],x[1]')
        t = sym.symbols(time_str)

        v_1 = 1-2*Cos(2*pi*(x-t))*Sin(2*pi*(y-t))
        v_2 = 1+2*Sin(2*pi*(x-t))*Cos(2*pi*(y-t))

        p = -Cos(4*pi*(x-t)) - Cos(4*pi*(y-t))

        w = 8*pi*Cos(2*pi*(x-t))*Cos(2*pi*(y-t))
        return [v_1,v_2], w, p

    def get_exact_sol_at_t(self, t_i):
        _v_ex, _w_ex, _p_ex = self.exact_solution()
        return self.convert_sym_to_expr(t_i, _v_ex, _w_ex, _p_ex)

    def initial_conditions(self, V_v, V_w, V_p):
        _v_ex, _w_ex, _p_ex = self.exact_solution()
        v_ex, w_ex, p_ex = self.convert_sym_to_expr(0.0, _v_ex, _w_ex, _p_ex)
        v_init = interpolate(v_ex, V_v)
        if V_w is not None:
            w_init = interpolate(w_ex, V_w)
        else:
            w_init = None
        p_init = interpolate(p_ex, V_p)
        return [v_init, w_init, p_init]

    def convert_sym_to_expr(self, t_i, _v_ex, _w_ex, _p_ex, degree=6, show_func=False):
        # Convert from Sympy to Expression for a given time instant or time-variable t_i

        v_ex_1_code = self.convert_sym('v_ex_1', _v_ex[0], show_func)
        v_ex_2_code = self.convert_sym('v_ex_2', _v_ex[1], show_func)
        v_ex = Expression((v_ex_1_code, v_ex_2_code), degree=degree, t=t_i)  # Vector valued expression

        w_ex_code = self.convert_sym('w_ex', _w_ex, show_func)
        w_ex = Expression(w_ex_code, degree=degree, t=t_i)  # Scalar valued expression

        p_ex_code = self.convert_sym('p_ex', _p_ex, show_func)
        p_ex = Expression(p_ex_code, degree=degree, t=t_i)  # Scalar valued expression

        return v_ex, w_ex, p_ex

    def init_outputs(self, t_c):
        # 6 outputs --> 3 exact states (velocity , vorticity and pressure)
        # and 3 exact integral quantities at time t (energy, enstrophy, helicity)
        u_ex_t, w_ex_t, p_ex_t = self.get_exact_sol_at_t(t_c)
        H_ex_t = 0.5 * (inner(u_ex_t, u_ex_t) * dx(domain=self.mesh))
        E_ex_t = 0.5 * (inner(w_ex_t, w_ex_t) * dx(domain=self.mesh))
        if self.dimM == 2:
            Ch_ex_t = None
        elif self.dimM == 2:
            Ch_ex_t = inner(u_ex_t, w_ex_t) * dx(domain=self.mesh)
        return [u_ex_t, w_ex_t, p_ex_t,H_ex_t,E_ex_t,Ch_ex_t]

    def calculate_outputs(self,exact_arr, u_t,w_t,p_t):
        err_u = errornorm(exact_arr[0], u_t, norm_type="L2")
        if w_t is not None:
            err_w = errornorm(exact_arr[1], w_t, norm_type="L2")
        else:
            err_w = 0.0 # Indicating that solver has no vorticity information
        err_p = errornorm(exact_arr[2], p_t, norm_type="L2")
        H_ex_t = assemble(exact_arr[3])
        E_ex_t = assemble(exact_arr[4])
        Ch_ex_t = 0.0#assemble(exact_arr[5])
        return np.array([err_u,err_w,err_p,H_ex_t,E_ex_t,Ch_ex_t])

    def boundary_conditions(self):

        # Create periodic boundary condition
        pbc = PeriodicBoundary()

        return pbc


    def __str__(self):
        return "PeriodicAnalyticalEuler2D"


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[1], 0)) and
            (not ((near(x[0], 1) and near(x[1], 0)) or
                  (near(x[0], 0) and near(x[1], 1)))) and on_boundary)

    def map(self, x, y):
        #### define mapping for a single point in the rectangle, such that 2 mappings are required
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
        #### define mapping for edges in the rectangle, such that 1 mappings is required
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
        else:
            y[0] = -1000
            y[1] = -1000
