from utils import log_likelihood
from Demo.Data.HMLN.GeneratorRobotMapping import generate_rel_graph, load_data
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from EPBPLogVersion import EPBP
from HybridMaxWalkSAT import HybridMaxWalkSAT as HMWS


rel_g = generate_rel_graph()
g, rvs_dict = rel_g.ground_graph()

query = dict()
for key, rv in rvs_dict.items():
    if key[0] == 'PartOf' or key[0] == 'SegType':
        query[key] = rv

data = load_data('Demo/Data/HMLN/robot-map')
for key, rv in rvs_dict.items():
    if key not in data and key not in query and not rv.domain.continuous:
        data[key] = 0  # closed world assumption

# data = dict()

g, rvs_dict = rel_g.add_evidence(data)
print(len(rvs_dict) - len(data))
print(len(g.factors))

# infer = HMWS(g)
# infer.run(max_tries=2, max_flips=10000, epsilon=0.9, noise_std=1)

infer = C2FVI(g, num_mixtures=2, num_quadrature_points=3)
infer.run(100, lr=0.7)

# map_res = infer.rvs_map(rvs_dict.values())
# for key, rv in rvs_dict.items():
#     print(key, map_res[rv])

map_res = dict()
for key, rv in rvs_dict.items():
    map_res[rv] = infer.map(rv)
    print(key, map_res[rv])

print(log_likelihood(g, map_res))
