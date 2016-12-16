import numpy as np
from GPElib.instability_generator import InstabilityGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed):
	inst.generate_init('random', traj_seed, 10.)
	delta = (2. * np.sqrt(1.0 * inst.N_part/inst.N_wells)) * np.random.rand()
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
	seed_to = 3
	my_id = 0
	unique_id = 'ID_not_stated'

needed_trajs = np.arange(seed_from, seed_to)

# time = 100.
time = 100.
# step = 0.00015625
step = 0.01
N_wells = 64
W = 0.

inst = InstabilityGenerator(N_part_per_well=100,
                             # N_wells=(10,10), dimensionality=2,
                             N_wells=(4,4,4), dimensionality=3,
                             disorder_seed=53, time=time, step=step,
                             perturb_hamiltonian=False,
                             error_J=0, error_beta=0, error_disorder=0)

grname = 'Instability_' + unique_id
vis = Visualisation(is_local=1, GROUP_NAMES=grname)

print "Characteristic, full, step times, n_steps"
print inst.tau_char, inst.time, inst.step, inst.n_steps

answers = []
lambdas = []
lambdas_no_regr = []
polarisation = []
polarisation1 = []
dist = []
energy = []

all_x = {}
all_y = {}
all_x1 = {}
all_y1 = {}

for ii in needed_trajs:
	inst.traj_seed = ii
	np.random.seed(ii)
	err = init_instability(inst, ii)
	if err == 1:
		print 'Bad trajectory! ', ii
	inst.run_dynamics()
	answers.append(inst.distance)
	lambdas.append(inst.lambdas[0])
	lambdas_no_regr.append(inst.lambdas_no_regr[0])
	polarisation.append(inst.polarisation)
	polarisation1.append(inst.polarisation1)
	dist.append(inst.distance)
	energy.append(inst.energy)
	all_x[ii] = inst.X
	all_y[ii] = inst.Y
	all_x1[ii] = inst.X1
	all_y1[ii] = inst.Y1

	# np.savez(vis.filename(my_id) + '_traj_' + str(ii),
	# 	         step=inst.step, time=inst.time, n_steps=inst.n_steps,
	# 	         error_code=inst.error_code, checksum=inst.consistency_checksum,
	# 	         distance=inst.distance,
	# 	         x=inst.X, y=inst.Y, x1=inst.X1, y1=inst.Y1)

T = np.linspace(0, inst.time, inst.n_steps)
for ii in xrange(needed_trajs.shape[0]):
	plt.semilogy(T, dist[ii])

plt.savefig(vis.HOMEDIR + 'pics/Inst_' + unique_id + '_' + str(my_id)+'.png', format='png', dpi=100)

np.savez(vis.filename(my_id),
         x=all_x, y=all_y, x1=all_x1, y1=all_y1,
         data=answers, lambdas=lambdas, lambdas_no_regr=lambdas_no_regr,
         polar=polarisation, polar1=polarisation1,
         distance=dist, energy=energy,
         step=inst.step, time=inst.time, n_steps=inst.n_steps,
         well_indices=inst.wells_indices,
         my_info=[seed_from, seed_to, my_id],
         needed_trajs=needed_trajs,
         description='Loschmidt echo for 10000 replicates, trying to calculate F(t)')