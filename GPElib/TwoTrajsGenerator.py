import numpy as np
from .DynamicsGenerator import DynamicsGenerator

class TwoTrajsGenerator(DynamicsGenerator):
	def __init__(self, **kwargs):
		DynamicsGenerator.__init__(self, **kwargs)
		self.RHO1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.THETA1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.X1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.Y1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.energy1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.participation_rate1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.effective_nonlinearity1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)

		self.number_of_particles1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.distance = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.lambdas = []
		self.lambdas_no_regr = []

	def calc_traj_shift_matrix_cartesian_XY(self, X, Y, X1, Y1):
		return np.sqrt(np.sum((X - X1) ** 2 + (Y - Y1) ** 2, axis=1))

	def set_init_XY(self, x, y, x1, y1):
		DynamicsGenerator.set_init_XY(self, x,y)
		self.X1[0,:] = x1
		self.Y1[0,:] = y1
		self.RHO1[0,:], self.THETA1[0,:] = self.from_XY_to_polar(self.X1[0,:], self.Y1[0,:])

	def set_constants_of_motion(self):
		DynamicsGenerator.set_constants_of_motion(self)
		self.energy1, self.number_of_particles1, self.angular_momentum1 = self.calc_constants_of_motion(self.RHO1, self.THETA1, self.X1, self.Y1)
		self.participation_rate1 = np.sum(self.RHO1 ** 4, axis=1) / (np.sum(self.RHO1 ** 2, axis=1) ** 2)
		self.effective_nonlinearity1 = self.beta * self.participation_rate1 / self.N_wells