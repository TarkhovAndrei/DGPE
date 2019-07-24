'''
Copyright <2017> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes followed by publication of obtained results,
the citation of the present code shall be provided according to the rule:

    "Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
    source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE
    was used to obtain the presented results, 2017."

2) The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
import matplotlib.pyplot as plt
from .dynamics_generator import DynamicsGenerator

class TwoTrajsGenerator(DynamicsGenerator):
	def __init__(self, **kwargs):
		DynamicsGenerator.__init__(self, **kwargs)
		self.RHO1 = np.zeros(self.N_tuple + (self.n_steps,), dtype=self.FloatPrecision)
		self.THETA1 = np.zeros(self.N_tuple + (self.n_steps,), dtype=self.FloatPrecision)
		self.X1 = np.zeros(self.N_tuple + (self.n_steps,), dtype=self.FloatPrecision)
		self.Y1 = np.zeros(self.N_tuple + (self.n_steps,), dtype=self.FloatPrecision)
		self.energy1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.participation_rate1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.effective_nonlinearity1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.histograms1 = {}
		self.rho_histograms1 = {}

		self.number_of_particles1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.distance = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.lambdas = []
		self.lambdas_no_regr = []

	def calc_traj_shift_matrix_cartesian_XY(self, X, Y, X1, Y1):
		return np.sqrt(np.sum((X - X1) ** 2 + (Y - Y1) ** 2, axis=(0,1,2)))

	def set_init_XY(self, x, y, x1, y1):
		DynamicsGenerator.set_init_XY(self, x,y)
		self.X1[:,:,:,0] = x1.reshape(self.N_tuple)
		self.Y1[:,:,:,0] = y1.reshape(self.N_tuple)
		self.RHO1[:,:,:,0], self.THETA1[:,:,:,0] = self.from_XY_to_polar(self.X1[:,:,:,0], self.Y1[:,:,:,0])

	def set_constants_of_motion(self):
		DynamicsGenerator.set_constants_of_motion(self)
		self.energy1, self.number_of_particles1, self.angular_momentum1 = self.calc_constants_of_motion(self.RHO1, self.THETA1, self.X1, self.Y1)
		self.participation_rate1 = np.sum(self.RHO1 ** 4, axis=(0,1,2)) / (np.sum(self.RHO1 ** 2, axis=(0,1,2)) ** 2)
		self.effective_nonlinearity1 = self.beta * self.participation_rate1 / self.N_wells
		for i in self.wells_indices:
			self.histograms1[i] = np.histogram2d(np.float64(self.X1[i]), np.float64(self.Y1[i]), bins=100)
			self.rho_histograms1[i] = np.histogram(np.float64(self.RHO1[i] ** 2), bins=100)
