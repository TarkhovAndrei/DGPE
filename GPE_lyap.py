import numpy as np
from GPElib.lyapunov_generator import LyapunovGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed):
	inst.generate_init('random', traj_seed, 100.)
	# delta = (2. * np.sqrt(1.0 * inst.N_part/inst.N_wells)) * np.random.rand()
	delta = (2. * np.sqrt(1.0 * inst.N_part)) * np.random.rand()
	x0, y0, err = inst.E_const_perturbation_XY(inst.X[:,:,:,0], inst.Y[:,:,:,0], delta)
	x1, y1 = inst.constant_perturbation_XY(x0,y0)
	inst.set_init_XY(x0,y0,x1,y1)
	return err

print len(sys.argv), sys.argv

if len(sys.argv) > 4:
	seed_from = int(sys.argv[1])
	seed_to = int(sys.argv[2])
	my_id = int(sys.argv[3])
	unique_id = sys.argv[4]
else:
	seed_from = 0
	seed_to = 1
	needed_trajs = np.arange(seed_from, seed_to)
	my_id = 0
	unique_id = 'ID_not_stated'

needed_trajs = np.arange(seed_from, seed_to)
perturb_seeds = np.arange(123,124)#(2381,2382)#(100, 110)#(106,108)#(97,98)#(97, 100)#(15, 18) #[53, 12, 20, 87]

time = 400 * 80.
# time = 1.
# time = 100. * 15
# step = 0.00015625
step = 0.001
N_wells = 10.
W = 0.

lyap = LyapunovGenerator(N_part_per_well=100,
                         W=W, disorder_seed=53,
                         # N_wells=(10,1,1), dimensionality=1, threshold_XY_to_polar=0.25,
                         # N_wells=(10,10,1), dimensionality=2, threshold_XY_to_polar=0.25,
                         N_wells=(4,4,4), dimensionality=3, threshold_XY_to_polar=0.25,
                         reset_steps_duration=3000,
                         # reset_steps_duration=150,
                         time=time, step=step)

grname = 'GPE_lyap_' + unique_id
vis = Visualisation(is_local=0, GROUP_NAMES=grname)

print "Noise ", W
print "Characteristic, full, step times, n_steps"
print lyap.tau_char, lyap.time, lyap.step, lyap.n_steps

num_good = 0

lmbdas = []
lmbdas_no_regr = []
chosen_trajs = []
effective_nonlinearity = []
energies = []
numb_of_part = []

for i_traj, traj_seed in enumerate(needed_trajs):
	if num_good > needed_trajs.shape[0] - 1:
		print 'We really have enough trajs, exit!'
		break
	for j_traj, pert_seed in enumerate(perturb_seeds):
		if num_good > needed_trajs.shape[0] - 1:
			print 'We really have enough trajs, exit!'
			break
		np.random.seed(traj_seed)
		lyap.traj_seed = traj_seed
		lyap.pert_seed = pert_seed
		err = init_instability(lyap, traj_seed)
		if err == 1:
			print 'Bad trajectory! ', i_traj
		lyap.run_dynamics()
		lmbdas.append(lyap.lambdas)
		lmbdas_no_regr.append(lyap.lambdas_no_regr)
		chosen_trajs.append((traj_seed, pert_seed))
		effective_nonlinearity.append(lyap.effective_nonlinearity)
		energies.append(lyap.energy)
		numb_of_part.append(lyap.number_of_particles)
		print lyap.lambdas
		print lyap.lambdas_no_regr
		num_good += 1
		plt.semilogy(lyap.T, lyap.distance)
		np.savez(vis.filename(my_id) + '_traj_' + str(i_traj),
		         step=lyap.step, time=lyap.time,
		         traj_seed=lyap.traj_seed,
		         pert_seed=lyap.pert_seed,
		         disorder_seed=lyap.disorder_seed,
		         disorder=lyap.e_disorder,
		         n_steps=lyap.n_steps,
		         wells_indices=lyap.wells_indices,
		         beta=lyap.beta, W=lyap.W,
		         J=lyap.J, N_tuple=lyap.N_tuple,
		         energy=lyap.energy, number_of_particles=lyap.number_of_particles,
		         eff_nonl=lyap.effective_nonlinearity,
		         error_code=lyap.error_code, checksum=lyap.consistency_checksum,
		         distance=lyap.distance,
		         x=lyap.X, y=lyap.Y, x1=lyap.X1, y1=lyap.Y1,
		         lambdas=lyap.lambdas, lambdas_no_regr=lyap.lambdas_no_regr,
		         hist2d=lyap.histograms, hist1d=lyap.rho_histograms,
		         hist2d1=lyap.histograms1, hist1d1=lyap.rho_histograms1)

plt.savefig(vis.HOMEDIR + 'pics/Lyap_' + unique_id + '_' + str(my_id)+'.png', format='png', dpi=100)

print "Error code: ", lyap.error_code
print "\n\nChecksum: ", lyap.consistency_checksum

np.savez(vis.filename(my_id),
         lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
         eff_nonl=effective_nonlinearity,
         numb_of_part=numb_of_part, energies=energies,
         chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
         my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
         checksum=lyap.consistency_checksum, error_code=lyap.error_code,
         distance=lyap.distance)
