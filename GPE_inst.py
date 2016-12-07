import numpy as np
from GPElib.dynamics import InstabilityGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed):
	inst.generate_init('random', traj_seed, 10.)
	delta = (2. * np.sqrt(1.0 * inst.N_part/inst.N_wells)) * np.random.rand()
	x0, y0, err = inst.E_const_perturbation_XY(inst.X[0,:], inst.Y[0,:], delta)
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
	seed_to = 5
	my_id = 0
	unique_id = 'ID_not_stated'

needed_trajs = np.arange(seed_from, seed_to)

# time = 100.
time = 100.
# step = 0.00015625
step = 0.01
N_wells = 100
W = 4.
np.random.seed(78)
e_disorder = -W  + 2 * W * np.random.rand(N_wells)
inst = InstabilityGenerator(N_part_per_well=100, N_wells=N_wells, disorder=e_disorder, time=time, step=step,
                            perturb_hamiltonian=False,
                            error_J=0, error_beta=0, error_disorder=0)
grname = 'Instability_' + unique_id
vis = Visualisation(is_local=0, GROUP_NAMES=grname)

print "Characteristic, full, step times, n_steps"
print inst.tau_char, inst.time, inst.step, inst.n_steps

answers = []
lambdas = []
lambdas_no_regr = []
polarisation = []
polarisation1 = []
dist = []
energy = []

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

T = np.linspace(0, inst.time, inst.n_steps)
for ii in xrange(needed_trajs.shape[0]):
	plt.semilogy(T, dist[ii])

plt.savefig(vis.HOMEDIR + 'pics/Inst_' + unique_id + '_' + str(my_id)+'.png', format='png', dpi=100)

np.savez(vis.filename(my_id),
         data=answers, lambdas=lambdas, lambdas_no_regr=lambdas_no_regr,
         polar=polarisation, polar1=polarisation1,
         distance=dist, energy=energy,
         step=inst.step, time=inst.time, n_steps=inst.n_steps,
         my_info=[seed_from, seed_to, my_id],
         needed_trajs=needed_trajs,
         description='Loschmidt echo for 10000 replicates, trying to calculate F(t)')