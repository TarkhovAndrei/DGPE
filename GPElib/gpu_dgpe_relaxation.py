import torch
import torchdiffeq
from torch.autograd import Variable


class DGPE_ODE(torch.nn.Module):

    def __init__(self, device):
        super(DGPE_ODE, self).__init__()

        self.a = torch.nn.Parameter(torch.tensor(0.2).to(device))
        self.b = torch.nn.Parameter(torch.tensor(3.0).to(device))

		self.torch_FloatPrecision = kwargs.get('torch_FloatPrecision', torch.float64)

		self.torch_gpu_id = kwargs.get('gpu_id', 0)
		self.torch_device = torch.device('cuda:' + str(self.torch_gpu_id)
								   if torch.cuda.is_available() else 'cpu')

		# self.tf_J = tf.placeholder(self.tf_FloatPrecision, name='J')
		# self.tf_anisotropy = tf.placeholder(self.tf_FloatPrecision, name='anisotropy')

		self.torch_J = torch.as_tensor(np.zeros(self.N_wells) + self.J, dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_anisotropy = torch.as_tensor(np.zeros(self.N_wells) + self.anisotropy,
											 dtype=self.torch_FloatPrecision, device=self.torch_device)

		self.torch_gamma = torch.as_tensor(np.zeros(self.N_wells) + self.gamma, dtype=self.torch_FloatPrecision,
										device=self.torch_device)

		# self.torch_N_wells = torch.tensor(self.N_wells, tf.int64)

		self.torch_nn_idx_1 = torch.as_tensor(self.nn_idx_1, dtype=torch.int64, device=self.torch_device)
		self.torch_nn_idx_2 = torch.as_tensor(self.nn_idx_2, dtype=torch.int64, device=self.torch_device)
		self.torch_nn_idy_1 = torch.as_tensor(self.nn_idy_1, dtype=torch.int64, device=self.torch_device)
		self.torch_nn_idy_2 = torch.as_tensor(self.nn_idy_2, dtype=torch.int64, device=self.torch_device)
		self.torch_nn_idz_1 = torch.as_tensor(self.nn_idz_1, dtype=torch.int64, device=self.torch_device)
		self.torch_nn_idz_2 = torch.as_tensor(self.nn_idz_2, dtype=torch.int64, device=self.torch_device)

		self.torch_first_half = torch.as_tensor(np.arange(self.N_wells), dtype=torch.int64, device=self.torch_device)
		self.torch_second_half = torch.as_tensor(np.arange(self.N_wells, 2 * self.N_wells),
											  dtype=torch.int64, device=self.torch_device)

		# self.torch_psi = Variable(self.psi, device=self.torch_device)
		# self.torch_x = Variable(self.psi[:self.N_wells], device=self.torch_device)
		# self.torch_y = Variable(self.psi[self.N_wells:], device=self.torch_device)

		self.torch_psi = torch.tensor(self.psi, dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_x = torch.tensor(self.psi[:self.N_wells], dtype=self.torch_FloatPrecision,
									device=self.torch_device)
		self.torch_y = torch.tensor(self.psi[self.N_wells:], dtype=self.torch_FloatPrecision,
									device=self.torch_device)

		self.torch_dpsi = torch.tensor(self.dpsi.shape, dtype=self.torch_FloatPrecision, device=self.torch_device)

		# self.torch_dpsi = Variable(np.zeros(self.dpsi.shape), device=self.torch_device)
		self.torch_E_new = torch.tensor(np.zeros(self.N_wells), dtype=self.torch_FloatPrecision, device=self.torch_device)
		# self.torch_E_new = Variable(np.zeros(self.N_wells), device=self.torch_device)

		# self.torch_xL = Variable(np.zeros(self.N_wells), device=self.torch_device)
		# self.torch_yL = Variable(np.zeros(self.N_wells), device=self.torch_device)

		self.torch_xL = torch.tensor(np.zeros(self.N_wells), dtype=self.torch_FloatPrecision,
									 device=self.torch_device)
		self.torch_yL = torch.tensor(np.zeros(self.N_wells), dtype=self.torch_FloatPrecision,
									 device=self.torch_device)

		self.torch_zero = torch.as_tensor(np.zeros(self.N_wells), dtype=self.torch_FloatPrecision, device=self.torch_device)
		# self.tf_zero = tf.placeholder(self.tf_FloatPrecision, name='zero')

		self.torch_h_dis_x_flat = torch.as_tensor(self.h_dis_x_flat, dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_h_dis_y_flat = torch.as_tensor(self.h_dis_y_flat, dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_beta_disorder_array_flattened = torch.as_tensor(self.beta_disorder_array_flattened,
																dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_beta = torch.as_tensor(self.beta_flat, dtype=self.torch_FloatPrecision, device=self.torch_device)

		if self.gpu_integrator == 'torch':
			self.torch_e_disorder = torch.tensor(self.e_disorder_flat, dtype=self.torch_FloatPrecision,
												 device=self.torch_device)

	def forward(self, t, y):
        return self.a + (y - (self.a * t + self.b))**5

    def y_exact(self, t):
        return self.a * t + self.b

	def torch_HamiltonianXY_fast(self, ts, y0):
		self.torch_psi = y0
		# self.torch_x = torch.gather(y0, 0, self.torch_first_half)
		# self.torch_y = torch.gather(y0, 0, self.torch_second_half)
		self.torch_x = y0[:self.N_wells]
		self.torch_y = y0[self.N_wells:]

		self.torch_dpsi = torch.cat([self.torch_e_disorder * self.torch_y, -self.torch_e_disorder * self.torch_x], dim=0)

		self.torch_xL = (self.torch_J * (
				torch.gather(self.torch_x, 0, self.torch_nn_idx_1) +
				torch.gather(self.torch_x, 0, self.torch_nn_idx_2) +
				torch.gather(self.torch_x, 0, self.torch_nn_idy_1) +
				torch.gather(self.torch_x, 0, self.torch_nn_idy_2) +
				self.torch_anisotropy * (torch.gather(self.torch_x, 0, self.torch_nn_idz_1) +
										 torch.gather(self.torch_x, 0, self.torch_nn_idz_2)
										 )
		))

		self.torch_yL = (self.torch_J * (
				torch.gather(self.torch_y, 0, self.torch_nn_idx_1) +
				torch.gather(self.torch_y, 0, self.torch_nn_idx_2) +
				torch.gather(self.torch_y, 0, self.torch_nn_idy_1) +
				torch.gather(self.torch_y, 0, self.torch_nn_idy_2) +
				self.torch_anisotropy * (torch.gather(self.torch_y, 0, self.torch_nn_idz_1) +
										 torch.gather(self.torch_y, 0, self.torch_nn_idz_2)
										 )
		))

		self.torch_dpsi.add_(torch.cat([-self.torch_yL, self.torch_xL], dim=0))

		self.torch_dpsi.add_(torch.cat([self.torch_h_dis_y_flat, -self.torch_h_dis_x_flat], dim=0))

		self.torch_dpsi.add_(torch.cat([self.torch_beta *
										(torch.pow(self.torch_y, 2) + torch.pow(self.torch_x, 2)) * self.torch_y,
										- self.torch_beta *
										(torch.pow(self.torch_y, 2) + torch.pow(self.torch_x, 2)) * self.torch_x], dim=0))

		return self.torch_dpsi


	def torch_Hamiltonian_with_Relaxation_XY_fast(self, ts, y0):
		self.torch_psi = y0
		self.torch_x = y0[:self.N_wells]
		self.torch_y = y0[self.N_wells:]

		# self.torch_x.data.zero_().add_(torch.gather(y0, 0, self.torch_first_half))
		# self.torch_y.data.zero_().add_(torch.gather(y0, 0, self.torch_second_half))

		self.torch_xL = (self.torch_J.mul(
			torch.gather(self.torch_x, 0, self.torch_nn_idx_1).add(
				torch.gather(self.torch_x, 0, self.torch_nn_idx_2)).add(
				torch.gather(self.torch_x, 0, self.torch_nn_idy_1)).add(
				torch.gather(self.torch_x, 0, self.torch_nn_idy_2)).add(
				self.torch_anisotropy.mul(torch.gather(self.torch_x, 0, self.torch_nn_idz_1).add(
					torch.gather(self.torch_x, 0, self.torch_nn_idz_2))
				))
		))

		self.torch_yL = (self.torch_J.mul(
			torch.gather(self.torch_y, 0, self.torch_nn_idx_1).add(
				torch.gather(self.torch_y, 0, self.torch_nn_idx_2)).add(
				torch.gather(self.torch_y, 0, self.torch_nn_idy_1)).add(
				torch.gather(self.torch_y, 0, self.torch_nn_idy_2)).add(
				self.torch_anisotropy.mul(torch.gather(self.torch_y, 0, self.torch_nn_idz_1).add(
					torch.gather(self.torch_y, 0, self.torch_nn_idz_2))
				))
		))

		# self.torch_dpsi[self.torch_first_half].zero_().add_(self.torch_gamma * self.torch_y * (
		# 		self.torch_xL * self.torch_y - self.torch_yL * self.torch_x))
		# self.torch_dpsi[self.torch_second_half].zero_().add_(-self.torch_gamma * self.torch_x * (
		# 							 self.torch_xL * self.torch_y - self.torch_yL * self.torch_x))

		self.torch_dpsi = torch.cat([self.torch_gamma.mul(self.torch_y).mul(
			torch.sub(self.torch_xL.mul(self.torch_y), self.torch_yL.mul(self.torch_x))),
			-self.torch_gamma.mul(self.torch_x).mul(
				torch.sub(self.torch_xL.mul(self.torch_y), self.torch_yL.mul(self.torch_x)))], dim=0)

		self.torch_dpsi.mul_(self.torch_get_gamma_reduction())

		self.torch_dpsi.add_(
			torch.cat([self.torch_e_disorder.mul(self.torch_y), -self.torch_e_disorder.mul(self.torch_x)], dim=0))

		# self.torch_dpsi[self.torch_first_half].add_(self.torch_e_disorder * self.torch_y)
		# self.torch_dpsi[self.torch_second_half].add_(-self.torch_e_disorder * self.torch_x)

		self.torch_dpsi.add_(torch.cat([-self.torch_yL, self.torch_xL], dim=0))
		# self.torch_dpsi[self.torch_first_half].add_(-self.torch_yL)
		# self.torch_dpsi[self.torch_second_half].add_(self.torch_xL)

		self.torch_dpsi.add_(torch.cat([self.torch_h_dis_y_flat, -self.torch_h_dis_x_flat], dim=0))

		# self.torch_dpsi[self.torch_first_half].add_(self.torch_h_dis_y_flat)
		# self.torch_dpsi[self.torch_second_half].add_(-self.torch_h_dis_x_flat)

		self.torch_dpsi.add_(torch.cat([
			self.torch_beta.mul(
				(torch.pow(self.torch_y, 2).add(torch.pow(self.torch_x, 2))).mul(self.torch_y)),
			- self.torch_beta.mul(
				(torch.pow(self.torch_y, 2).add(torch.pow(self.torch_x, 2))).mul(self.torch_x))], dim=0))

		# self.torch_dpsi[self.torch_first_half].add_(self.torch_beta *
		# 		   (torch.pow(self.torch_y, 2) + torch.pow(self.torch_x, 2)) * self.torch_y)
		# self.torch_dpsi[self.torch_second_half].add_(- self.torch_beta *
		# 		   (torch.pow(self.torch_y, 2) + torch.pow(self.torch_x, 2)) * self.torch_x)

		return self.torch_dpsi


	def torch_calc_energy_XY(self):
		# tf_E_new = tf.Variable(self.tf_zero, dtype=self.tf_FloatPrecision, trainable=True, initializer=tf.zeros_initializer)
		self.torch_E_new = self.torch_beta * 0.5 * (
			torch.pow(torch.pow(self.torch_x, 2.) + torch.pow(self.torch_y, 2.), 2.))
		self.torch_E_new.add_(self.torch_e_disorder * (torch.pow(self.torch_x, 2) + torch.pow(self.torch_y, 2)))
		self.torch_E_new.add_(-self.torch_J * (self.torch_x * (
				torch.gather(self.torch_x, 0, self.torch_nn_idx_1) +
				torch.gather(self.torch_x, 0, self.torch_nn_idx_2) +
				torch.gather(self.torch_x, 0, self.torch_nn_idy_1) +
				torch.gather(self.torch_x, 0, self.torch_nn_idy_2) +
				self.torch_anisotropy * (torch.gather(self.torch_x, 0, self.torch_nn_idz_1) +
										 torch.gather(self.torch_x, 0, self.torch_nn_idz_2)
										 )) +
											   self.torch_y * (
													   torch.gather(self.torch_y, 0, self.torch_nn_idx_1) +
													   torch.gather(self.torch_y, 0, self.torch_nn_idx_2) +
													   torch.gather(self.torch_y, 0, self.torch_nn_idy_1) +
													   torch.gather(self.torch_y, 0, self.torch_nn_idy_2) +
													   self.torch_anisotropy * (
																   torch.gather(self.torch_y, 0, self.torch_nn_idz_1) +
																   torch.gather(self.torch_y, 0, self.torch_nn_idz_2)
																   )
											   )
											   ))
		self.torch_E_new.add_(self.torch_h_dis_x_flat * self.torch_x + self.torch_h_dis_y_flat * self.torch_y)
		return self.torch_E_new


	if self.gpu_integrator == 'torch':
		# self.torch_E_new = torch.tensor(np.zeros(self.N_wells), dtype=self.torch_FloatPrecision,
		# 								device=self.torch_device)
		self.torch_gamma_reduction = torch.tensor(np.zeros(self.N_wells) + self.gamma_reduction,
												  dtype=self.torch_FloatPrecision, device=self.torch_device)
		self.torch_E_desired = torch.tensor(np.zeros(self.N_wells) + self.E_desired,
											dtype=self.torch_FloatPrecision, device=self.torch_device)
	# self.torch_temperature_dependent_rate = torch.constant(True, torch.bool)
