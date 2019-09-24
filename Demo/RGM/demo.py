from Demo.Data.RGM.Generator import generate_rel_graph
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI
from GaBP import GaBP


rel_g = generate_rel_graph()
rel_g.ground_graph()

data = {
    ('recession', 'all'): 50
}

g, rvs_dict = rel_g.add_evidence(data)

infer = VI(g, num_mixtures=1, num_quadrature_points=3)
infer.run(50, lr=0.2)

# infer = GaBP(g)
# infer.run(20)

for key, rv in rvs_dict.items():
    print(key, infer.rvs_map([rv]))
