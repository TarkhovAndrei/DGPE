import numpy as np
from sklearn.linear_model import LinearRegression
from .two_trajs_generator import TwoTrajsGenerator

class LyapunovGenerator(TwoTrajsGenerator):
	def __init__(self, **kwargs):
		TwoTrajsGenerator.__init__(self, **kwargs)
		self.Lyapunov_EPS = kwargs.get('Lyapunov_EPS', 1e-3)
		self.instability_stops = []
		self.reset_steps_duration = kwargs.get('reset_steps_duration', 10000)
		self.distance_check = []

	def reset_perturbation_XY(self, x0, y0, x1, y1):
		dst = self.calc_traj_shift_XY(x0, y0, x1, y1)
		if dst < 1e-5:
			x1 = x0 + (x1 - x0) * 1e-3/dst
			y1 = y0 + (y1 - y0) * 1e-3/dst
		dst = self.calc_traj_shift_XY(x0, y0, x1, y1)
		x1 = x0 + (x1 - x0) * self.PERT_EPS /dst
		y1 = y0 + (y1 - y0) * self.PERT_EPS /dst
		return x1, y1

	def run_dynamics(self):
		x1, y1 = self.constant_perturbation_XY(self.X[:,:,:,0], self.Y[:,:,:,0])
		self.set_init_XY(self.X[:,:,:,0], self.Y[:,:,:,0], x1, y1)

		self.instability_stops = [0]
		self.distance_check = [self.calc_traj_shift_XY(self.X[:,:,:,0], self.Y[:,:,:,0], self.X1[:,:,:,0], self.Y1[:,:,:,0])]
		for i in xrange(1, self.n_steps):
			if (np.any(self.RHO[:,:,:,i-1] ** 2 < self.threshold_XY_to_polar) or (np.any(self.RHO1[:,:,:,i-1] ** 2 < self.threshold_XY_to_polar))):
				psi = self.rk4_step_exp_XY(np.hstack((self.X[:,:,:,i-1].flatten(), self.Y[:,:,:,i-1].flatten())))
				psi1 = self.rk4_step_exp_XY(np.hstack((self.X1[:,:,:,i-1].flatten(), self.Y1[:,:,:,i-1].flatten())))
				self.X[:,:,:,i] = psi[:self.N_wells].reshape(self.N_tuple)
				self.Y[:,:,:,i] = psi[self.N_wells:].reshape(self.N_tuple)
				self.RHO[:,:,:,i], self.THETA[:,:,:,i] = self.from_XY_to_polar(self.X[:,:,:,i], self.Y[:,:,:,i])
				self.X1[:,:,:,i] = psi1[:self.N_wells].reshape(self.N_tuple)
				self.Y1[:,:,:,i] = psi1[self.N_wells:].reshape(self.N_tuple)
				self.RHO1[:,:,:,i], self.THETA1[:,:,:,i] = self.from_XY_to_polar(self.X1[:,:,:,i], self.Y1[:,:,:,i])
				# self.X1[:,:,:,i], self.Y1[:,:,:,i] = self.from_polar_to_XY(self.RHO1[:,:,:,i], self.THETA1[:,:,:,i])
			else:
				psi = self.rk4_step_exp(np.hstack((self.RHO[:,:,:,i-1].flatten(), self.THETA[:,:,:,i-1].flatten())))
				psi1 = self.rk4_step_exp(np.hstack((self.RHO1[:,:,:,i-1].flatten(), self.THETA1[:,:,:,i-1].flatten())))
				self.RHO[:,:,:,i] = psi[:self.N_wells].reshape(self.N_tuple)
				self.THETA[:,:,:,i] = psi[self.N_wells:].reshape(self.N_tuple)
				self.X[:,:,:,i], self.Y[:,:,:,i] = self.from_polar_to_XY(self.RHO[:,:,:,i], self.THETA[:,:,:,i])
				self.RHO1[:,:,:,i] = psi1[:self.N_wells].reshape(self.N_tuple)
				self.THETA1[:,:,:,i] = psi1[self.N_wells:].reshape(self.N_tuple)
				self.X1[:,:,:,i], self.Y1[:,:,:,i] = self.from_polar_to_XY(self.RHO1[:,:,:,i], self.THETA1[:,:,:,i])
			dist = self.calc_traj_shift_XY(self.X[:,:,:,i], self.Y[:,:,:,i], self.X1[:,:,:,i], self.Y1[:,:,:,i])
			self.distance_check.append(dist)
			# if (dist > self.Lyapunov_EPS) or (i - self.instability_stops[-1] > self.reset_steps_duration):
			if (i - self.instability_stops[-1] > self.reset_steps_duration):
				self.X1[:,:,:,i], self.Y1[:,:,:,i] = self.reset_perturbation_XY(self.X[:,:,:,i], self.Y[:,:,:,i], self.X1[:,:,:,i], self.Y1[:,:,:,i])
				self.RHO1[:,:,:,i], self.THETA1[:,:,:,i] = self.from_XY_to_polar(self.X1[:,:,:,i], self.Y1[:,:,:,i])
				self.instability_stops.append(i)
		if self.instability_stops[-1] != self.n_steps:
			self.instability_stops.append(self.n_steps)
		self.set_constants_of_motion()
		self.distance = self.calc_traj_shift_matrix_cartesian_XY(self.X, self.Y, self.X1, self.Y1)
		if (np.abs(np.max(np.abs(self.energy - self.E_calibr)) / self.E_calibr) > 0.01) or (np.abs(np.max(np.abs(self.energy1 - self.E_calibr)) / self.E_calibr) > 0.01):
			self.make_exception('Energy is not conserved during the dynamics\n')
		if (np.abs(np.max(np.abs(self.number_of_particles - self.N_part)) / self.N_part) > 0.01) or (np.abs(np.max(np.abs(self.number_of_particles1 - self.N_part)) / self.N_part) > 0.01):
			self.make_exception('Number of particles is not conserved during the dynamics\n')
		if (np.max(self.distance_check) > 1.):
			self.make_exception('Discontinuity during the calculations (distance b/w trajectories > 1)\n')
		self.calculate_lambdas()

	def calculate_lambdas(self):
		self.lambdas = []
		self.lambdas_no_regr = []
		clf = LinearRegression()
		for ist in xrange(np.array(self.instability_stops).shape[0]-1):
			fr = self.instability_stops[ist] + 1
			to = self.instability_stops[ist+1] - 1
			try:
				clf.fit(self.T[fr:to].reshape(to-fr,1), np.log(self.distance[fr:to] + 1e-15).reshape(to-fr,1))
				self.lambdas.append(clf.coef_[0][0])
			except:
				print 'Bad Lyapunov lambda'
				self.lambdas.append(0.)
			self.lambdas_no_regr.append((np.log(self.distance[to] + 1e-15) - np.log(self.distance[fr]+ 1e-15)) / (self.T[to] - self.T[fr]))
