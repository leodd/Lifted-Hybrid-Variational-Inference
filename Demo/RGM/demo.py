from Demo.Data.RGM.Generator import generate_rel_graph
from utils import log_likelihood
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP
from HybridMaxWalkSAT import HybridMaxWalkSAT as HMWS


rel_g = generate_rel_graph()
rel_g.ground_graph()

data = {
    ('recession', 'all'): 25
}

g, rvs_dict = rel_g.add_evidence(data)

infer = HMWS(g)
infer.run(max_tries=1, max_flips=10000, epsilon=0.0, noise_std=0.5)

# infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(200, lr=0.2)

# infer = GaBP(g)
# infer.run(20)

map_res = dict()
for key, rv in rvs_dict.items():
    map_res[rv] = infer.map(rv)
    print(key, map_res[rv])

print(log_likelihood(g, map_res))
