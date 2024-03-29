
from problems.problem_base import *
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import sympy as sym
# Problem definition
class BeltramiProblem(ProblemBase):
    "3D Beltrami test problem with known analytical solution."

    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # Create mesh
        # We start with a UnitCube and modify it to get the mesh we want: (-1, 1) x (-1, 1) x (-1, 1)
        self.mesh = UnitCubeMesh(self.n_el, self.n_el, self.n_el)
        scale = 2 * (self.mesh.coordinates() - 0.5)
        self.mesh.coordinates()[:, :] = scale
        self.init_mesh()
        self.structured_time_grid()

        # Set viscosity
        self.mu = 1.0
        # Set density
        self.rho = 1.0

    def exact_solution(self, time_str='t'):
        from sympy import exp as Exp
        from sympy import sin as Sin
        from sympy import cos as Cos

        a = pi/4.0
        d = pi/2.0

        # Mesh coordinates
        x, y, z = sym.symbols('x[0],x[1],x[2]')
        t = sym.symbols(time_str)

        v_1 = -a*(Exp(a*x)*Sin(a*y+d*z) + Exp(a*z)*Cos(a*x+d*y))*Exp(-d*d*t)
        v_2 = -a*(Exp(a*y)*Sin(a*z+d*x) + Exp(a*x)*Cos(a*y+d*z))*Exp(-d*d*t)
        v_3 = -a*(Exp(a*z)*Sin(a*x+d*y) + Exp(a*y)*Cos(a*z+d*x))*Exp(-d*d*t)

        p_bar = Sin(a*x+d*y)*Cos(a*z+d*x)*Exp(a*y+a*z) + Sin(a*y+d*z)*Cos(a*x+d*y)*Exp(a*x+a*z) \
              + Sin(a*z+d*x)*Cos(a*y+d*z)*Exp(a*x+a*y)
        p = -0.5*a*a*Exp(-2*d*d*t)*(Exp(2*a*x)+Exp(2*a*y)+Exp(2*a*z)+2*p_bar)

        w_1 =  sym.diff(v_3,y) - sym.diff(v_2,z)
        w_2 = sym.diff(v_1, z) - sym.diff(v_3, x)
        w_3 = sym.diff(v_2, x) - sym.diff(v_1, y)

        # Check divergence is zero and alignment of velocity and vorticity vector fields
        # div_v = sym.diff(v_1,x) + sym.diff(v_2,y) + sym.diff(v_3,z)
        # print("Exact divergence:", sym.simplify(div_v))
        # v = sym.Matrix([v_1,v_2,v_3])
        # w = sym.Matrix([w_1, w_2, w_3])
        # print("Exact allignment of vorticity and velocity:", sym.simplify(v.cross(w)))

        return [v_1,v_2,v_3], [w_1, w_2, w_3], p

    def initial_conditions(self, V_v, V_w, V_p):
        _v_ex, _w_ex, _p_ex = self.exact_solution()
        v_ex, w_ex, p_ex = self.convert_sym_to_expr(0.0, _v_ex, _w_ex, _p_ex)
        v_init = interpolate(v_ex, V_v)
        w_init = interpolate(w_ex, V_w)
        p_init = interpolate(p_ex, V_p)
        return [v_init,w_init,p_init]


    def boundary_conditions(self, V, Q, t):
        self.exact_u.t = t
        self.exact_p.t = t

        bc0 = DirichletBC(V, self.exact_u, DomainBoundary())

        bcu = [bc0]
        bcp = []

        return bcu, bcp

    def get_exact_sol_at_t(self, t_i):
        _v_ex, _w_ex, _p_ex = self.exact_solution()
        return self.convert_sym_to_expr(t_i, _v_ex, _w_ex, _p_ex)

    def reference(self, t):
        pass

    def convert_sym_to_expr(self, t_i, _v_ex, _w_ex, _p_ex, degree=6, show_func=False):
        # Convert from Sympy to Expression for a given time instant or time-variable t_i

        v_ex_1_code = self.convert_sym('v_ex_1', _v_ex[0], show_func)
        v_ex_2_code = self.convert_sym('v_ex_2', _v_ex[1], show_func)
        v_ex_3_code = self.convert_sym('v_ex_3', _v_ex[2], show_func)
        v_ex = Expression((v_ex_1_code, v_ex_2_code, v_ex_3_code), degree=degree, t=t_i)  # Vector valued expression

        w_ex_1_code = self.convert_sym('w_ex_1', _w_ex[0], show_func)
        w_ex_2_code = self.convert_sym('w_ex_2', _w_ex[1], show_func)
        w_ex_3_code = self.convert_sym('w_ex_3', _w_ex[2], show_func)
        w_ex = Expression((w_ex_1_code, w_ex_2_code, w_ex_3_code), degree=degree, t=t_i)  # Vector valued expression

        p_ex_code = self.convert_sym('p_ex', _p_ex, show_func)
        p_ex = Expression(p_ex_code, degree=degree, t=t_i)  # Scalar valued expression

        return v_ex, w_ex, p_ex

    def __str__(self):
        return "Beltrami 3D"
