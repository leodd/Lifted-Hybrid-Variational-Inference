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

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('algo', type=str)  # any of OSI, LOSI, NPVI, LNPVI
parser.add_argument('K', type=int)
parser.add_argument('-n', '--num_tests', type=int, default=5)
args = parser.parse_args()
# algo = args.algo
K = args.K
num_tests = args.num_tests
print('#### run setup ####')
pprint(vars(args))

record_fields = ['cpu_time',
                 'wall_time',
                 'obj',  # this is BFE/-ELBO for variational methods, -logZ for exact baseline
                 # 'mmap_err',  # |argmax p(xi) - argmax q(xi)|, avg over all nodes i
                 # 'kl_err',  # kl(p(xi)||q(xi)), avg over all nodes i
                 'map_energy'
                 ]
# algo_names = ['baseline', 'EPBP', 'OSI', 'LOSI']
# algo_names = ['baseline', 'NPVI', 'OSI', ]
# algo_names = ['baseline', 'EPBP', 'NPVI', 'LNPVI', 'OSI', 'LOSI']
# algo_names = ['baseline', 'EPBP', 'OSI', 'LOSI', 'NPVI', 'LNPVI']
# algo_names = ['baseline', algo, ]
# algo_names = ['baseline', 'EPBP']
# algo_names = ['EPBP']
# assert algo_names[0] == 'baseline'
algo_names = ['OSI', 'NPVI']
# for each algorithm, we keep a record, which is a dict mapping a record_field to a list (which will eventually be
# averaged over)
records = {algo_name: {record_field: [] for record_field in record_fields} for algo_name in algo_names}

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
utils.set_log_potential_funs(cond_g.factors_list, skip_existing=True)

print('cond number of rvs', len(cond_g.rvs))
print('cond num drvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'd']))
print('cond num crvs', len([rv for rv in cond_g.rvs if rv.domain_type[0] == 'c']))

for test_num in range(num_tests):
    test_seed = seed + test_num

    for algo_num, algo_name in enumerate(algo_names):
        print('####')
        print('test_num', test_num)
        print('running', algo_name)
        np.random.seed(test_seed + algo_num)

        # temp storage
        # mmap = np.zeros(len(query_rvs)) - 123
        # margs = [None] * len(query_rvs)
        # marg_kls = np.zeros(len(query_rvs)) - 123
        map_energy = -123
        obj = -1
        cpu_time = wall_time = -1  # don't care

        if algo_name in ('OSI', 'LOSI', 'NPVI', 'LNPVI'):
            # K = 1
            T = 3
            lr = 0.5
            its = 500
            fix_mix_its = int(its * 0.5)
            logging_itv = 100

            if algo_name in ('OSI', 'NPVI'):
                if algo_name == 'OSI':
                    vi = OneShot(g=cond_g, K=K, T=T, seed=test_seed)
                else:
                    vi = NPVI(g=cond_g, K=K, T=T, isotropic_cov=False, seed=test_seed)

                start_time = time.process_time()
                start_wall_time = time.time()
                res = vi.run(lr=lr, its=its, fix_mix_its=fix_mix_its, logging_itv=logging_itv)
                cpu_time = time.process_time() - start_time
                wall_time = time.time() - start_wall_time
                obj = res['record']['obj'][-1]

                # joint MAP
                map_config = joint_map(cond_g.rvs_list, cond_g.Vd, cond_g.Vc, cond_g.Vd_idx, cond_g.Vc_idx, vi.params)
                map_energy = eval_joint_assignment_energy(cond_g.factors_list,
                                                          {rv: map_config[i] for (i, rv) in enumerate(cond_g.rvs_list)})

                print(map_energy)

        algo_record = dict(cpu_time=cpu_time, wall_time=wall_time, obj=obj,  # mmap_err=mmap_err, kl_err=kl_err,
                           map_energy=map_energy)
        for key, value in algo_record.items():
            records[algo_name][key].append(value)

from collections import OrderedDict

avg_records = OrderedDict()
for algo_name in algo_names:
    record = records[algo_name]
    avg_record = OrderedDict()
    for record_field in record_fields:
        avg_record[record_field] = (np.mean(record[record_field]), np.std(record[record_field]))
    avg_records[algo_name] = avg_record

for key, value in avg_records.items():
    print(key + ':')
    pprint(dict(value))
# import json
# output = json.dumps(avg_records, indent=0, sort_keys=True)
print()
print()
