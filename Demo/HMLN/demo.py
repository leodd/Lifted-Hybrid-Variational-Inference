from Demo.Data.HMLN.GeneratorRobotMapping import generate_rel_graph
from VarInference import VarInference as VI
from LiftedVarInference import VarInference as LVI
from C2FVarInference import VarInference as C2FVI


rel_g = generate_rel_graph()
g, rvs_dict = rel_g.ground_graph()

infer = VI(g, num_mixtures=1, num_quadrature_points=3)
infer.run(200, lr=0.2)
