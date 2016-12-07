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

	def set_init(self, rho, theta, *args):
		self.RHO[0,:] = np.abs(rho)
		self.THETA[0,:] = self.phase_unwrap(theta)
		self.X[0,:], self.Y[0,:] = self.from_polar_to_XY(self.RHO[0,:], self.THETA[0,:])

	def set_init(self, rho, theta, rho1, theta1):
		DynamicsGenerator.set_init(self, rho, theta)
		self.RHO1[0,:] = np.abs(rho1)
		self.THETA1[0,:] = self.phase_unwrap(theta1)
		self.X1[0,:], self.Y1[0,:] = self.from_polar_to_XY(self.RHO1[0,:], self.THETA1[0,:])



	def calc_traj_shift_matrix_cartesian(self, RHO, THETA, RHO1, THETA1):
		return np.sqrt(np.sum((RHO * np.cos(THETA) - RHO1 * np.cos(THETA1)) ** 2 + (RHO * np.sin(THETA) - RHO1 * np.sin(THETA1)) ** 2, axis=1))


	def calc_traj_shift(self, rho0, theta0, rho1, theta1):
		rho0 = np.abs(rho0)
		rho1 = np.abs(rho1)
		theta0 -= np.mean(theta0)
		theta1 -= np.mean(theta1)
		return np.sqrt(np.sum((rho0 * np.cos(theta0) - rho1 * np.cos(theta1)) ** 2 + (rho0 * np.sin(theta0) - rho1 * np.sin(theta1)) ** 2))

	def constant_perturbation(self, rho0, theta0):
		np.random.seed(self.pert_seed)
		eps = 1e-1
		rho1 = rho0 + eps * np.random.randn(self.N_wells)
		theta1 = theta0 + eps * np.random.randn(self.N_wells)
		dist = self.calc_traj_shift(rho0, theta0, rho1, theta1)
		rho1 = rho0 + (rho1 - rho0) * self.PERT_EPS /dist
		theta1 = theta0 + (theta1 - theta0) * self.PERT_EPS /dist
		return rho1, theta1


	def reset_perturbation(self, rho0, theta0, rho1, theta1):
		dist = self.calc_traj_shift(rho0, theta0, rho1, theta1)
		rho1 = rho0 + (rho1 - rho0) * self.PERT_EPS /dist
		theta1 = theta0 + (theta1 - theta0) * self.PERT_EPS /dist
		return rho1, theta1

def calc_energy(self, rho, theta, E):
	E_new = -E
	for j in xrange(self.N_wells):
		E_new += (-self.J * (rho[j] * rho[self.NN(j-1)] * (np.cos(theta[j] - theta[self.NN(j-1)])) +
		                rho[j] * rho[self.NN(j+1)] * (np.cos(theta[j] - theta[self.NN(j+1)]))) +
		          self.beta/2. * np.abs(rho[j]**4)+
		           self.e_disorder[j] * np.abs(rho[j]**2))
	return E_new


def calc_number_of_particles(self, rho):
	return (np.sum(rho ** 2) - self.N_part)


def E_const_perturbation(self, rho0, theta0, delta):
	bnds = np.hstack((rho0, theta0))
	rho_err = 1.0
	theta_err = 1.0
	np.random.seed()
	rho_next = np.abs(rho0 + rho_err * np.random.randn(self.N_wells))
	theta_next = theta0 + theta_err * np.random.randn(self.N_wells)
	theta_next -= np.mean(theta_next)

	x0 = np.hstack((rho_next, theta_next))
	fun = lambda x: (((self.calc_energy(x[:self.N_wells],x[self.N_wells:],self.E_calibr))/self.E_calibr) ** 2 +
	                 (self.calc_number_of_particles(x[:self.N_wells])/self.N_part) ** 2)
	opt = minimize(fun, x0,
	               bounds=[(xi - 1.0 * delta, xi + 1.0 * delta) for xi in bnds],
	               options={'ftol':self.FTOL})

	col = 0
	while (col < 10) and ((opt.success == False) or (np.abs(self.calc_energy(opt.x[:self.N_wells],opt.x[self.N_wells:], self.E_calibr)) > self.E_eps) or
		                      (np.abs(self.calc_number_of_particles(opt.x[:self.N_wells])/self.N_part) > 0.01)):
		np.random.seed()
		x0new = x0 + 1.0 * np.random.randn(x0.shape[0])
		x0new[:self.N_wells] = np.abs(x0new[:self.N_wells])
		x0new[self.N_wells:] -= np.mean(x0new[self.N_wells:])
		opt = minimize(fun, x0new,
	               bounds=[(xi - 10.0 * delta, xi + 10.0 * delta) for xi in bnds],
	               options={'ftol':self.FTOL})
		col += 1
	rho1 = np.abs(opt.x[:self.N_wells])
	theta1 = opt.x[self.N_wells:]
	theta1 -= np.mean(theta1)
	if np.abs(self.calc_energy(rho1, theta1, self.E_calibr) / self.E_calibr) > 0.01:
		self.make_exception('Could not find a new initial on-shell state\n')
	if np.abs((self.calc_number_of_particles(rho1)) / self.N_part) > 0.01:
		self.make_exception('Could not find a new initial state with the same number of particles\n')
	if np.abs(self.calc_traj_shift(rho1,theta1, rho0, theta0) / delta) < 0.3:
		self.make_exception('Could not find a trajectory on such a distance\n')
		return rho1, theta1, 1
	if col == 10:
		self.make_exception('Exceeded number of attempts in E_const_perturbation\n')
		return rho1, theta1, 1
	else:
		return rho1, theta1, 0


def run_dynamics_new(rho0, theta0, Ham, FloatPrecision):
	# global FloatPrecision, n_steps, N_wells, J, beta, step
	psi0 = np.hstack((np.abs(rho0), theta0-np.mean(theta0)))
	T_ = np.zeros(n_steps, dtype=FloatPrecision)
	RHO_ = np.zeros((n_steps,rho0.shape[0]), dtype=FloatPrecision)
	THETA_ = np.zeros((n_steps, theta0.shape[0]), dtype=FloatPrecision)
	T_ = step * np.arange(n_steps)
	answer = odeint(Ham,psi0, T_, args=(J,beta,N_wells),Dfun=Jacobian,
	                printmessg=1,
	                h0=1e-7,
	                hmax=1e-5, hmin=1e-9,
	                mxordn=12, mxords=5,
	                rtol=1e-12, atol=1e-12)
	RHO_ = np.abs(answer[:,:N_wells])
	THETA_ = (answer[:,N_wells:].T - np.mean(answer[:,N_wells:], axis=1).T).T
	return T_, RHO_, THETA_

def plot_3D_dynamics(ax, x_dat, y_dat, z_dat, i, N_wells, color):
	global fontlabel, fontticks
	ax.set_xlabel(r'$Re(\psi)$',fontsize=fontlabel)
	ax.set_ylabel(r'$Im(\psi)$',fontsize=fontlabel)
	ax.set_zlabel(r'$Time$',fontsize=fontlabel)
	ax.set_title('Spin ' + str(i), fontsize=fontlabel)
	plt.setp(ax.get_xticklabels(), fontsize=fontticks)
	plt.setp(ax.get_yticklabels(), fontsize=fontticks)
	plt.setp(ax.get_zticklabels(), fontsize=fontticks)
	plt.xlim([-300,300])
	plt.setp(ax.get_zticklabels(), fontsize=fontticks)

	ax.plot(x_dat, y_dat, z_dat,color=color)

def plot_2D_dynamics(ax, x_dat, y_dat, i, N_wells, color):
	global fontlabel, fontticks
	ax.set_xlabel(r'$\theta$',fontsize=fontlabel)
	ax.set_ylabel(r'$|\psi|$',fontsize=fontlabel)
	ax.set_title('Spin ' + str(i), fontsize=fontlabel)
	plt.setp(ax.get_xticklabels(), fontsize=fontticks)
	plt.setp(ax.get_yticklabels(), fontsize=fontticks)
	ax.plot(x_dat, y_dat,color=color)


def run_eff_dynamics(p0, x0, Ham, FloatPrecision):
	global rho0, theta0
	P_ = np.zeros((n_steps,rho0.shape[0]))
	X_ = np.zeros((n_steps, theta0.shape[0]))
	T_ = np.zeros(n_steps)
	psi = np.vstack((p0,x0))
	for i in xrange(n_steps):
		t = i * step
		psi = rk4_step_exp(0, psi, N_wells, step, Hamiltonian_eff, J, beta, N_wells, FloatPrecision) #N_wells
		T_[i] = t
		P_[i, :] = psi[0, :]
		X_[i, :] = psi[1, :]
	return T_, P_, X_

def Hamiltonian_eff(t,psi,J, beta, N_wells):
	p0 = psi[0,:]
	x0 = psi[1,:]
	dp = np.zeros(psi.shape[1])
	dx = np.zeros(psi.shape[1])
	for i in xrange(dx.shape[0]):
		dx[i] += p0[i]
		dp[i] += 3 * N_part/N_wells * beta * (np.sin(x0[NN(i+1, N_wells)]-x0[i]) + np.sin(x0[NN(i-1, N_wells)]-x0[i]))
	#     ham += V_j * psi +
	ham = np.vstack((dp,dx))
	return ham


def run_lyapunov(rho0, theta0, Ham, RHO0, THETA0, current_iter, init_shift, stop_shift, FloatPrecision, E_calibr=0, stop_iter=10000):
	psi0 = np.hstack((np.abs(rho0), theta0 - np.mean(theta0)))
	T_ = np.zeros(n_steps, dtype=FloatPrecision)
	RHO_ = np.zeros((n_steps,rho0.shape[0]), dtype=FloatPrecision)
	THETA_ = np.zeros((n_steps, theta0.shape[0]), dtype=FloatPrecision)
	psi = psi0

	RHO_[current_iter, :] = np.abs(rho0)#RHO0[current_iter,:])#rho0
	THETA_[current_iter,:] = theta0 - np.mean(theta0)#THETA0[current_iter,:] - np.mean(THETA0[current_iter,:])#theta0

	i = current_iter + 1
	next_iter = i - 1

	while i < n_steps:
		t = i * step
		psi = rk4_step_exp(psi, t, N_wells, step, Ham, J, beta, N_wells, FloatPrecision) #N_wells
		T_[i] = t
		RHO_[i, :] = psi[:N_wells]
		THETA_[i, :] = psi[N_wells:]
		curr_shift = calc_traj_shift(RHO_[i, :], THETA_[i, :], RHO0[i, :], THETA0[i, :])
		if (curr_shift > stop_shift) or (i - current_iter > stop_iter):
			if i - current_iter < 10:
				print RHO_[i, :]
				print RHO0[i, :]
				print THETA0[i, :]
				print THETA_[i, :]
			print "Start resetting:"
			print curr_shift, stop_shift
			next_iter = i
			break
		i += 1
		if i == n_steps:
			print RHO_[i-1, :]
			print RHO0[i-1, :]
			print THETA0[i-1, :]
			print THETA_[i-1, :]
			print energy(RHO_[i-1, :], THETA_[i-1, :], E_calibr, J, beta, N_wells), energy(RHO0[i-1, :], THETA0[i-1, :], E_calibr, J, beta, N_wells)
			next_iter = n_steps - 1
			break
	return T_, RHO_, THETA_, next_iter

def filename(HOMEDIR, GROUP_NAMES, i, FILE_TYPE):
	return HOMEDIR + GROUP_NAMES + str(i) + FILE_TYPE

def energy(rho, theta, E_calibr, J, beta, N_wells):
	E_new = -E_calibr
	for j in xrange(rho.shape[0]):
		E_new += (-J * (rho[j] * rho[NN(j-1,N_wells)] * (np.cos(theta[j] - theta[NN(j-1,N_wells)])) +
		                rho[j] * rho[NN(j+1,N_wells)] * (np.cos(theta[j] - theta[NN(j+1,N_wells)]))) +
		          beta/2. * np.abs(rho[j]**4))
	return E_new

def number_of_particles(rho):
	return np.sum(rho ** 2)


def E_const_perturbation(rho0, theta0, delta, E, FTOL, E_eps, J, beta, N_wells):
	rho0 = np.abs(np.array(rho0))
	theta0 = np.array(theta0)
	theta0 -= np.mean(theta0)
	bnds = np.hstack((rho0, theta0))

	rho_err = 1.0 #* delta
	theta_err = 1.0 #delta

	theta_next = theta0 + theta_err * np.random.randn(N_wells)
	theta_next -= np.mean(theta_next)
	rho_next = np.abs(rho0 + rho_err * np.random.randn(N_wells))

	#     dist = calc_traj_shift(rho_next, theta_next, rho0, theta0)

	x0 = np.hstack((rho_next, theta_next))
	#     x0 = np.hstack((rho0, theta0))

	lam_E = 1.
	lam_reg = 1.

	fun = lambda x: (lam_E * ((energy(x[:N_wells],x[N_wells:],E, J, beta, N_wells))/E) ** 2)#+
	#                 lam_reg * ((calc_traj_shift(x[:N_wells],x[N_wells:], rho0, theta0) - delta)) ** 2)

	#     opt = minimize(fun, x0, bounds=[(xi - 1.0 * delta, xi + 1.0 * delta) for xi in bnds],
	#                    method='L-BFGS-B', options={'ftol':FTOL})
	opt = minimize(fun, x0,
	               bounds=[(xi - 1.0 * delta/np.sqrt(N_wells), xi + 1.0 * delta/ np.sqrt(N_wells)) for xi in bnds],
	               method='SLSQP', constraints=({'type': 'eq', 'fun': lambda x:  (calc_traj_shift(x[:N_wells],x[N_wells:], rho0, theta0) - delta)}),
	               options={'ftol':FTOL})
	col = 0
	while (col < 10) and ((opt.success == False) or (np.abs(energy(opt.x[:N_wells],opt.x[N_wells:], E, J, beta, N_wells)) > E_eps)):
		print "Unsuccessful iteration, start again"
		np.random.seed()
		#         opt = minimize(fun, x0 * (1.0 + 1.0 * np.random.randn(x0.shape[0])),
		#                        bounds=[(xi - 10.0*delta, xi + 10.0*delta) for xi in bnds],
		#                        method='L-BFGS-B', options={'ftol':FTOL})
		x0new = x0 + 1.0 * np.random.randn(x0.shape[0])
		x0new[:N_wells] = np.abs(x0new[:N_wells])
		x0new[N_wells:] -= np.mean(x0new[N_wells:])
		opt = minimize(fun, x0new,
		               bounds=[(xi - 1e+3 * delta, xi + 1e+3 * delta) for xi in bnds],
		               method='SLSQP', constraints=({'type': 'eq', 'fun': lambda z: (calc_traj_shift(z[:N_wells],z[N_wells:], rho0, theta0) - delta)}),
		               options={'ftol':FTOL})
		col += 1
	print opt.success, energy(opt.x[:N_wells],opt.x[N_wells:],E, J, beta, N_wells)
	print calc_traj_shift(opt.x[:N_wells],opt.x[N_wells:], rho0, theta0), delta

	if col == 10:
		print "Cannot find a trajectory"
		return np.abs(opt.x[:N_wells]), opt.x[N_wells:] - np.mean(opt.x[N_wells:]), 1
	else:
		return np.abs(opt.x[:N_wells]), opt.x[N_wells:] - np.mean(opt.x[N_wells:]), 0




# def calc_traj_shift_cartesian(rho0, theta0, rho1, theta1):
def calc_traj_shift(rho0, theta0, rho1, theta1):
	rho0 = np.abs(rho0)
	rho1 = np.abs(rho1)
	theta0 -= np.mean(theta0)
	theta1 -= np.mean(theta1)
	X0 = rho0 * np.cos(theta0)
	Y0 = rho0 * np.sin(theta0)
	X1 = rho1 * np.cos(theta1)
	Y1 = rho1 * np.sin(theta1)
	return np.sqrt(np.sum((X0 - X1) ** 2 + (Y0 - Y1) ** 2))

def calc_traj_shift_polar(rho0, theta0, rho1, theta1):
	theta0_n = theta0 - np.mean(theta0)
	theta1_n = theta1 - np.mean(theta1)
	rho0_n = np.abs(rho0)
	rho1_n = np.abs(rho1)
	return np.sqrt(np.sum((rho0_n - rho1_n) ** 2 + (theta0_n - theta1_n) ** 2))

def calc_traj_shift_matrix(RHO0, THETA0, RHO1, THETA1):
	theta0_n = (THETA0.T - np.mean(THETA0, axis=1).T).T
	theta1_n = (THETA1.T - np.mean(THETA1, axis=1).T).T
	rho0_n = np.abs(RHO0)
	rho1_n = np.abs(RHO1)
	return np.sqrt(np.sum((rho0_n - rho1_n) ** 2 + (theta0_n - theta1_n) ** 2, axis=1))

def calc_traj_shift_matrix_cartesian(RHO0, THETA0, RHO1, THETA1):
	RHO0 = np.abs(RHO0)
	RHO1 = np.abs(RHO1)
	THETA0 = (THETA0.T - np.mean(THETA0, axis=1).T).T
	THETA1 = (THETA1.T - np.mean(THETA1, axis=1).T).T

	X0 = RHO0 * np.cos(THETA0)
	Y0 = RHO0 * np.sin(THETA0)
	X1 = RHO1 * np.cos(THETA1)
	Y1 = RHO1 * np.sin(THETA1)
	return np.sqrt(np.sum((X0 - X1) ** 2 + (Y0 - Y1) ** 2, axis=1))

def run_eff_dynamics(p0, x0, rho0, theta0, Ham, FloatPrecision):
	P_ = np.zeros((n_steps,rho0.shape[0]))
	X_ = np.zeros((n_steps, theta0.shape[0]))
	T_ = np.zeros(n_steps)
	psi = np.vstack((p0,x0))
	for i in xrange(n_steps):
		t = i * step
		psi = rk4_step_exp(0, psi, N_wells, step, Ham, J, beta, N_wells, FloatPrecision) #N_wells
		T_[i] = t
		P_[i, :] = psi[0, :]
		X_[i, :] = psi[1, :]
	return T_, P_, X_


def run_lyapunov_new(rho0, theta0, Ham, RHO0, THETA0, current_iter, init_shift, stop_shift, FloatPrecision, stop_iter=10000):
	psi0 = np.hstack((np.abs(rho0), theta0 - np.mean(theta0)))
	T_ = np.zeros(n_steps, dtype=FloatPrecision)
	RHO_ = np.zeros((n_steps,rho0.shape[0]), dtype=FloatPrecision)
	THETA_ = np.zeros((n_steps, theta0.shape[0]), dtype=FloatPrecision)

	RHO_[current_iter, :] = np.abs(rho0)#RHO0[current_iter,:])#rho0
	THETA_[current_iter,:] = theta0 - np.mean(theta0)#THETA0[current_iter,:] - np.mean(THETA0[current_iter,:])#theta0

	next_iter = current_iter + stop_iter

	if current_iter + stop_iter > n_steps:
		next_iter = n_steps

	t = np.arange(current_iter, next_iter) * step
	answer = odeint(Ham,psi0, t, args=(J,beta,N_wells),Dfun=Jacobian,
	                printmessg=1,
	                h0=1e-7,
	                hmax=1e-5, hmin=1e-9,
	                mxordn=12, mxords=8,
	                rtol=1e-12, atol=1e-12)
	RHO_[current_iter:next_iter] = answer[:,:N_wells]
	THETA_[current_iter:next_iter] = answer[:,N_wells:]
	THETA_ = (THETA_.T - np.mean(THETA_, axis=1).T).T
	T_[current_iter:next_iter] = t
	return T_, RHO_, THETA_, next_iter




def rk2_step_exp(y0, t0, N_wells, h, dydt_exp, *args):

	y0[:N_wells] = np.abs(y0[:N_wells])
	y0[N_wells:] -= np.mean(y0[N_wells:])

	k1 = h * dydt_exp(y0, t0, *args)
	t2 = t0 + h
	y2 = y0 + k1
	k2 = h * dydt_exp(y2, t2, *args)
	yi = y0 + (k1 + k2)/2.0

	yi[:N_wells] = np.abs(yi[:N_wells])
	yi[N_wells:] -= np.mean(yi[N_wells:])

	return yi


def Hamiltonian_eff(t,psi,J, beta, N_wells):
	global N_part
	p0 = psi[0,:]
	x0 = psi[1,:]
	dp = np.zeros(psi.shape[1])
	dx = np.zeros(psi.shape[1])
	for i in xrange(dx.shape[0]):
		dx[i] += p0[i]
		dp[i] += 3 * N_part/N_wells * beta * (np.sin(x0[NN(i+1, N_wells)]-x0[i]) + np.sin(x0[NN(i-1, N_wells)]-x0[i]))
	#     ham += V_j * psi +
	ham = np.vstack((dp,dx))
	return ham

def harmonic_potential(j, ksi, j0):
	return ksi*((j-j0) ** 2)

def run_dynamics_new(rho0, theta0, Ham, FloatPrecision):
	global n_steps, step, J, beta, N_wells
	psi0 = np.hstack((np.abs(rho0), theta0-np.mean(theta0)))
	T_ = np.zeros(n_steps, dtype=FloatPrecision)
	RHO_ = np.zeros((n_steps,rho0.shape[0]), dtype=FloatPrecision)
	THETA_ = np.zeros((n_steps, theta0.shape[0]), dtype=FloatPrecision)
	T_ = step * np.arange(n_steps)
	answer = odeint(Ham,psi0, T_, args=(J,beta,N_wells, FloatPrecision),Dfun=Jacobian,
	                printmessg=1,
	                h0=1e-7,
	                hmax=1e-5, hmin=1e-9,
	                mxordn=12, mxords=5,
	                rtol=1e-12, atol=1e-12)
	RHO_ = np.abs(answer[:,:N_wells])
	THETA_ = (answer[:,N_wells:].T - np.mean(answer[:,N_wells:], axis=1).T).T
	return T_, RHO_, THETA_

