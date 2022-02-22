
from problems.problem_base import *
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


# Problem definition
class BoxProblem(ProblemBase):
    "3D wave test problem with known analytical solution."

    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # Create mesh
        self.mesh = BoxMesh(Point(0, 0, 0), Point(1, 0.5, 0.5), self.n_el, self.n_el, self.n_el)
        self.init_mesh()
        self.structured_time_grid()
        # print("Space dimension is :", self.dimM)

    def exact_solution(self, time_str='t'):
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
        x, y, z = sym.symbols('x[0],x[1],x[2]')
        t = sym.symbols(time_str)

        # functions f and g
        f = 2.0 * sym.sin(om_t * t + phi_t) + 3.0 * sym.cos(om_t * t + phi_t)
        g = sym.cos(om_x * x + phi_x) * sym.sin(om_y * y + phi_y) * sym.sin(om_z * z + phi_z)
        # Derivatives of f and g
        _dt_f = sym.diff(f, t)
        _dx_g = sym.diff(g, x)
        _dy_g = sym.diff(g, y)
        _dz_g = sym.diff(g, z)

        # Exact p and q expressions
        p_ex = g * _dt_f
        q_ex_1 = -_dx_g * f
        q_ex_2 = -_dy_g * f
        q_ex_3 = -_dz_g * f

        return p_ex, [q_ex_1, q_ex_2, q_ex_3]

    def initial_conditions(self, V_p, V_q):
        _p_ex, _q_ex = self.exact_solution()
        p_ex, q_ex = self.convert_sym_to_expr(0.0, _p_ex, _q_ex)
        p_init = interpolate(p_ex, V_p)
        q_init = interpolate(q_ex, V_q)
        return [p_init,q_init]

    def boundary_conditions(self, V, Q, t):
        pass

    def get_exact_sol_at_t(self,t_i):
        _p_ex, _q_ex = self.exact_solution()
        return self.convert_sym_to_expr(t_i, _p_ex, _q_ex)

    def reference(self, t):
        pass

    def convert_sym_to_expr(self, t_i, p_ex, q_ex, degree=6, show_func=False):
        # Convert from Sympy to Expression for a given time instant or time-variable t_i
        p_ex_code = self.convert_sym('p_ex', p_ex, show_func)
        _p_ex = Expression(p_ex_code, degree=degree, t=t_i)  # Scalar valued expression

        q_ex_1_code = self.convert_sym('q_ex_1', q_ex[0], show_func)
        q_ex_2_code = self.convert_sym('q_ex_2', q_ex[1], show_func)
        q_ex_3_code = self.convert_sym('q_ex_3', q_ex[2], show_func)
        _q_ex = Expression((q_ex_1_code, q_ex_2_code, q_ex_3_code), degree=degree, t=t_i)  # Vector valued expression

        return _p_ex, _q_ex

    def __str__(self):
        return "Box Wave 3D"




