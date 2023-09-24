import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class Flux():
	def __init__(self):
		self.X, self.Y = sym.symbols('x y',real=True)
		self.Z = self.X + sym.I*self.Y

	def uniformFlow(self, alpha:float = 0.25*np.pi, U:float = 1) -> sym.core:
		return U * self.Z * sym.exp(-sym.I * alpha)

	def lineVortex(z:sym.core = self.Z, Gamma:float = -1, z0:sym.core = 0 + 0*sym.I) -> sym.core:
		return -sym.I * Gamma * sym.log(z - z0)/(2*sym.pi)

	def stagnationPoint(U:float = 1, z0:sym.core = 0 + 0*sym.I) -> sym.core:
		return 0.5 * U * (self.Z - z0)**2

	def coaxialCircles(Gamma:float = 1, d: sym.core = 5 + 0*sym.I) -> sym.core:
		image = lineVortex(Gamma = Gamma, z0 = d)
   		vortex = lineVortex(Gamma = Gamma, z0 = -d)
		return image - vortex

	def circularCylinder(z:sym.core = self.Z, U:float = 1, a:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi):
		no_circulation = U * (z * sym.exp(-sym.I*alpha) + (sym.exp(sym.I * alpha) * a**2)/z)
		return no_circulation + lineVortex(z=z, Gamma = Gamma)
		
	def elipticalCylinder(c:float = 1, U:float = 1, a:float = 1, Gamma:float = -4*np.pi, alpha:float = 0.25*np.pi):
		z = 0.5 * (Z + (Z**2 - 4 * c**2)**0.5)
		return circularCylinder(z = z, U = U, a = a, Gamma = Gamma, alpha = alpha)

	def getFlux(expr:sym.core):
		"""
		Compute the fluxes (u,v) given a complex potencial w(z)
		
		Args
		expr :: sympy.core
		
		Return
		u,v,psi :: Tuple(sympy.core)
		"""
		# z = ϕ + ıψ
		psi = sym.im(expr)
		# u =  ψ,y
		u = sym.diff(psi,self.Y)
		# v = - ψ,x
		v = -sym.diff(psi,self.X)
		return u, v, psi

	def convertSymbolic(expr:sym.core):
	"""
	Convert the symbolic expression into a lambda function
	
	Args
	expr :: sympy.core
	
	Return
	lambda function (x,y)
	"""
	return sym.lambdify([self.X,self.Y], expr)
