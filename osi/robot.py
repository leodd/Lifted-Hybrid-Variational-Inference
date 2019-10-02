import utils
import pickle
import os.path
from pprint import pprint

utils.set_path(('..', '../gibbs'))
from Demo.Data.HMLN.GeneratorRobotMapping import generate_rel_graph, load_data
from RelationalGraph import *
from MLNPotential import *
from Potential import QuadraticPotential, TablePotential, HybridQuadraticPotential
from EPBPLogVersion import EPBP
from OneShot import OneShot, LiftedOneShot
from NPVI import NPVI, LiftedNPVI
from CompressedGraphSorted import CompressedGraphSorted
import numpy as np
import time
from copy import copy

seed = 0
utils.set_seed(seed)
from mixture_beliefs import joint_map
from utils import eval_joint_assignment_energy
# from hybrid_gaussian_mrf import HybridGaussianSampler
# from hybrid_gaussian_mrf import convert_to_bn, block_gibbs_sample, get_crv_marg, get_drv_marg, \
#     get_rv_marg_map_from_bn_params
# import sampling_utils

from utils import kl_continuous_logpdf

rel_g = generate_rel_graph()
g, rvs_dict = rel_g.ground_graph()

query = dict()
for key, rv in rvs_dict.items():
    if key[0] == 'PartOf' or key[0] == 'SegType':
        query[key] = rv

data = load_data('../Demo/Data/HMLN/robot-map')
for key, rv in rvs_dict.items():
    if key not in data and key not in query and not rv.domain.continuous:
        data[key] = 0  # closed world assumption

# data = dict()

g, rvs_dict = rel_g.add_evidence(data)
print(len(rvs_dict) - len(data))
print(len(g.factors))

obs_rvs = [v for v in g.rvs if v.value is not None]
evidence = {rv: rv.value for rv in obs_rvs}
cond_g = utils.get_conditional_mrf(g.factors_list, g.rvs, evidence)  # this will also condition log_potential_funs
cond_g.init_nb()  # this will make cond_g rvs' .nb attributes consistent (baseline didn't care so it was OK)

print('cond number of rvs', len(cond_g.rvs))
print('cond num drvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'd']))
print('cond num crvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'c']))

K = 1
T = 3
lr = 0.5
its = 500
fix_mix_its = int(its * 0.5)
logging_itv = 50
utils.set_log_potential_funs(cond_g.factors_list, skip_existing=True)


def double_check_energy(factors, assignment):
    # a version of /utils.py/log_likelihood to check against my implementation
    res = 0
    for f in factors:
        parameters = [assignment[rv] for rv in f.nb]
        value = f.potential.get(parameters)
        if value == 0:
            return -np.Inf
        res += log(value)

    return res


algo_name = 'OSI'
if algo_name in ('OSI', 'NPVI'):
    if algo_name == 'OSI':
        vi = OneShot(g=cond_g, K=K, T=T, seed=seed)
    else:
        vi = NPVI(g=cond_g, K=K, T=T, isotropic_cov=False, seed=seed)

    start_time = time.process_time()
    start_wall_time = time.time()
    res = vi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
    cpu_time = time.process_time() - start_time
    wall_time = time.time() - start_wall_time
    obj = res['record']['obj'][-1]

    map_config = joint_map(cond_g.rvs_list, cond_g.Vd, cond_g.Vc, cond_g.Vd_idx, cond_g.Vc_idx, vi.params)
    map_energy = eval_joint_assignment_energy(cond_g.factors_list,
                                              {rv: map_config[i] for (i, rv) in enumerate(cond_g.rvs_list)})

    print(map_energy)
    double_check = double_check_energy(cond_g.factors_list,
                                       {rv: map_config[i] for (i, rv) in enumerate(cond_g.rvs_list)})
    print(map_energy - double_check)  # should be basically 0
