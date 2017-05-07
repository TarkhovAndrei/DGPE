'''
Copyright <2017> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology, https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes with subsequent publication of obtained results,
the GitHub repository shall be cited in all publications, according to the citation rule:
	"Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
	 source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE, 2017."

2) The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize


def harmonic_potential(j, ksi, j0):
	return ksi*((j-j0) ** 2)

def estimate_numerical_error(rho0, theta0, J, beta):
	T, RHO, THETA = run_dynamics(rho0, theta0, Hamiltonian)
	Jin, betain = J, beta
	error_J = 0
	error_beta = 0
	J, beta = reverse_hamiltonian(J, beta, error_J, error_beta)
	check_constants_of_motion(T, RHO, THETA)
	T_rev, RHO_rev, THETA_rev = run_dynamics(RHO[-1,:], THETA[-1,:], Hamiltonian)
	check_constants_of_motion(T_rev, RHO_rev, THETA_rev)
	J, beta = Jin, betain
	T_rev = T[-1] - T_rev

	idx = np.arange(RHO.shape[0])[::-1]
	rev_error = calc_traj_shift_matrix(RHO,THETA, RHO_rev[idx,:],THETA_rev[idx,:])
	plt.semilogy(T, rev_error,'r')
	plt.show()
	return rev_error[0]

def E_reset_perturbation(rho0, theta0, rho, theta, delta, E, FTOL, E_eps, J, beta, N_wells):
	rho0 = np.abs(np.array(rho0))
	theta0 = np.array(theta0)
	theta0 -= np.mean(theta0)
	rho = np.abs(np.array(rho))
	theta = np.array(theta)
	theta -= np.mean(theta)

	x0 = 1.0
	lam_E = 1.
	lam_reg = 1.
	multiplier = 1.
	dist_init = calc_traj_shift(rho0, theta0, rho, theta)
	#     return rho0 + delta * (rho - rho0)/dist_init, (theta0 + delta * (theta - theta0)/dist_init), 0

	if calc_traj_shift(rho0, theta0, rho, theta) < 1e-3:
		multiplier = 1e-3 / calc_traj_shift(rho0, theta0, rho, theta)
		rho = rho0 + multiplier * (rho - rho0)

	fun = lambda x: (lam_E * (energy(rho0 + x * (rho - rho0),theta0 + x * (theta - theta0),E, J, beta, N_wells) ** 2))#  +
	#                 lam_reg * ((calc_traj_shift(rho0 + x * (rho - rho0),theta0 + x * (theta - theta0), rho0, theta0) - delta)) ** 2))

	opt = minimize(fun, x0,
	               bounds=[(0,1)],
	               method='SLSQP', constraints=({'type': 'eq', 'fun': lambda z:  (calc_traj_shift(rho0 + z * (rho - rho0),theta0 + z * (theta - theta0), rho0, theta0) - delta)/delta}),
	               options={'ftol':FTOL})
	#     opt = minimize(fun, x0, bounds=[(0,1)],
	#                    method='L-BFGS-B', options={'ftol':FTOL})
	col = 0
	while ((col < 15) and ((opt.success == False)
	                       or (np.abs(energy(rho0 + opt.x * (rho - rho0),theta0 + opt.x * (theta - theta0),E, J, beta, N_wells)) > E_eps))):
		print " "
		print "Unsuccessful iteration, start again"
		np.random.seed()
		x0new = x0 * np.random.rand()
		#         opt = minimize(fun, np.random.rand(), bounds=[(0,1)],
		#                    method='L-BFGS-B', options={'ftol':FTOL})
		opt = minimize(fun, x0new,
		               bounds=[(0,10)],
		               method='SLSQP', constraints=({'type': 'eq', 'fun': lambda z:  (calc_traj_shift(rho0 + z * (rho - rho0),theta0 + z * (theta - theta0), rho0, theta0) - delta)/delta}),
		               options={'ftol':FTOL})
		col += 1
	print opt.success, np.abs(energy(rho0 + opt.x * (rho - rho0),theta0 + opt.x * (theta - theta0),E, J, beta, N_wells))
	print np.abs(calc_traj_shift(rho0 + opt.x * (rho - rho0),theta0 + opt.x * (theta - theta0), rho0, theta0) - delta) / delta

	if col == 15:
		print "Cannot find a trajectory"
		return np.abs(rho0 + opt.x * (rho - rho0)), (theta0 + opt.x * (theta - theta0)) - np.mean((theta0 + opt.x * (theta - theta0))), 1
	else:
		return np.abs(rho0 + opt.x * (rho - rho0)), (theta0 + opt.x * (theta - theta0)) - np.mean((theta0 + opt.x * (theta - theta0))), 0

