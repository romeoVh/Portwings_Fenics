from dolfin import *
from mshr import *
import matplotlib.pyplot as plt

cx, cy, radius = 0.5, 0.5, 0.25
lx, ly = 1.0, 1.0

class circle(SubDomain):
    def inside(self, x, on_boundary):
        return pow(x[0] - cx, 2) + pow(x[1] - cy, 2) <= pow(radius, 2)

# Define 2D geometry
domain = Rectangle(Point(0.0, 0.0), Point(lx, ly))
domain.set_subdomain(1, Circle(Point(cx, cy), radius))


# Generate and plot mesh
mesh2d = generate_mesh(domain, 10)
plot(mesh2d, "2D mesh")

# Convert subdomains to mesh function for plotting
mf = MeshFunction("size_t", mesh2d, 2)
mf.set_all(0)
circle = circle()

for c in cells(mesh2d):
  if circle.inside(c.midpoint(), True):
    mf[c.index()] = 1


plot(mf, "Subdomains")
plt.show()
