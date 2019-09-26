from Demo.Data.HMLN.GeneratorRobotMapping import generate_rel_graph, load_data
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from EPBPLogVersion import EPBP


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

g, rvs_dict = rel_g.add_evidence(data)

infer = VI(g, num_mixtures=3, num_quadrature_points=3)
infer.run(200, lr=0.2)

# map_res = infer.rvs_map(rvs_dict.values())
# for key, rv in rvs_dict.items():
#     print(key, map_res[rv])

for key, rv in rvs_dict.items():
    print(key, infer.map(rv))
