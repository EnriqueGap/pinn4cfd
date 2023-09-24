import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class Flow():
	def __init__(self):
		"""
		class Flux
		Computes 2D ideal, irrotational flows using complex potentials and symbolic calculus using sympy
		"""
		self.X, self.Y = sym.symbols('x y',real=True)
		self.Z = self.X + sym.I*self.Y

	def uniformFlow(self, U:float = 1, alpha:float = 0.25*np.pi) -> sym.core:
		"""
		uniformFlow(U:float = 1, alpha:float = 0.25*np.pi) -> sympy.core
		
		Computes the complex potential of a uniform flow at angle to the x-axis

		Args
		U:float (Optional, 1) flow speed
		alpha:float (Optional, 0.25pi) angle formed by the flow and x-axis
		
		Returns:
		sympy.core: Complex potential Uze^(-i*alpha)
		"""
		return U * self.Z * sym.exp(-sym.I * alpha)

	def lineVortex(self, z = None, Gamma:float = -1, z0:sym.core = 0 + 0*sym.I) -> sym.core:
		"""
		lineVortex(self, z = None, Gamma:float = -1, z0:sympy.core = 0 + 0*sympy.I) -> sympy.core
		
		Computes the complex potential of a line vortex with singularity at z0

		Args
		z:sympy.core (Optional) used to define a conformal mapping as functions of self.Z
		Gamma:float (Optional, -1) flow circulation
		z0:sympy.core (Optional, 0) position of singularity (vortex center)
		
		Returns:
		sympy.core: Complex potential -i*Gamma*log(z-z0)/2pi
		"""
		if z is None:
			z = self.Z
		return -sym.I * Gamma * sym.log(z - z0)/(2*sym.pi)

	def stagnationPoint(self, U:float = 1, z0:sym.core = 0 + 0*sym.I) -> sym.core:
		"""
		stagnationPoint(self, U:float = 1, z0:sympy.core = 0 + 0*sympy.I) -> sympy.core
		
		Computes the complex potential of a flow with stagnation point at z0

		Args
		U:float (Optional, 1) flow speed
		z0:sympy.core (Optional, 0) position of stagnation point
		
		Returns:
		sympy.core: Complex potential 0.5*U(z-z0)^2
		"""
		return 0.5 * U * (self.Z - z0)**2

	def coaxialCircles(self, Gamma:float = -1, d: sym.core = 5 + 0*sym.I) -> sym.core:
		"""
		coaxialCircles(self, Gamma:float = 1, d: sympy.core = 5 + 0*sympy.I) -> sympy.core
		
		Computes the complex potential of two symetric line vortexes separated by d

		Args
		Gamma:float (Optional, -1) flow circulation
		d:sympy.core (Optional, 5+0*sympy.I) separation between vortex centers
		
		Returns:
		sympy.core: Complex potential -i*Gamma*log(z-d)/2pi + i*Gamma*log(z+d)/2pi
		"""
		image = self.lineVortex(Gamma = Gamma, z0 = d)
		vortex = self.lineVortex(Gamma = Gamma, z0 = -d)
		return image - vortex

	def circularCylinder(self, z:sym.core = None, U:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi, radius:float = 1) -> sym.core:
		"""
		circularCylinder(self, z:sympy.core = None, U:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi, radius:float = 1) -> sympy.core
		
		Computes the complex potential of a uniform flow passing a circular cylinder

		Args
		z:sympy.core (Optional) used to define a conformal mapping as functions of self.Z
		U:float (Optional, 1) flow speed
		Gamma:float (Optional, -4pi) flow circulation
		alpha:float (Optional, 0.25pi) angle formed by the flow and x-axis
		radius:float (Optional, 1) radius of the circular cylinder
		
		Returns:
		sympy.core: Complex potential U(ze^(-i*alpha) + e^(i*alpha)*a^2/z) - i*Gamma*log(z)/2pi
		"""
		if z is None:
			z = self.Z
		no_circulation = U * (z * sym.exp(-sym.I*alpha) + (sym.exp(sym.I * alpha) * radius**2)/z)
		return no_circulation + self.lineVortex(z=z, Gamma = Gamma)
		
	def elipticalCylinder(self, U:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi, radius:float = 1, c:float = 1) -> sym.core:
		"""
		elipticalCylinder(self, U:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi, radius:float = 1, c:float = 1) -> sympy.core
		
		Computes the complex potential of a uniform flow passing a eliptical cylinder using circularCylinder method and conformal mapping
		z=0.5*(Z+sqrt(Z^2 - 4c^2))
		
		Args
		U:float (Optional, 1) flow speed
		Gamma:float (Optional, -4pi) flow circulation
		alpha:float (Optional, 0.25pi) angle formed by the flow and x-axis
		radius:float (Optional, 1) radius of the circular cylinder
		c:float (Optional, 1) pending
		
		Returns:
		sympy.core: Complex potential
		"""
		z = 0.5 * (self.Z + (self.Z**2 - 4 * c**2)**0.5)
		return self.circularCylinder(z = z, U = U, Gamma = Gamma, alpha = alpha, radius = radius)

	def getFlux(self, expr:sym.core):
		"""
		getFlux(self, expr:sym.core)
		
		Compute the fluxes (u,v) given a complex potencial w(z)
		
		Args
		expr :: sympy.core
		
		Return
		u, v, psi :: Tuple(sympy.core)
		"""
		# z = ϕ + ıψ
		psi = sym.im(expr)
		# u =  ψ,y
		u = sym.diff(psi, self.Y)
		# v = - ψ,x
		v = -sym.diff(psi, self.X)
		return u, v, psi

	def convertSymbolic(self, expr:sym.core):
		"""
		convertSymbolic(self, expr:sym.core)
		
		Convert the symbolic expression into a sympy lambdify function

		Args
		expr :: sympy.core

		Return
		lambda function (x,y)
		"""
		return sym.lambdify([self.X,self.Y], expr)
