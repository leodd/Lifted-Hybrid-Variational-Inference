from RelationalGraph import *
from Potential import GaussianPotential
from GaBP import GaBP
import numpy as np
import json


def generate_rel_graph():
    instance_category = []
    instance_bank = []
    for i in range(100):
        instance_category.append(f'c{i}')
    for i in range(5):
        instance_bank.append(f'b{i}')

    d = Domain((-50, 50), continuous=True, integral_points=linspace(-30, 30, 100))

    p1 = GaussianPotential([0., 0.], [[10., -7.], [-7., 10.]])
    p2 = GaussianPotential([0., 0.], [[10., 5.], [5., 10.]])
    p3 = GaussianPotential([0., 0.], [[10., 7.], [7., 10.]])

    lv_recession = LV(('all',))
    lv_category = LV(instance_category)
    lv_bank = LV(instance_bank)

    atom_recession = Atom(d, logical_variables=(lv_recession,), name='recession')
    atom_market = Atom(d, logical_variables=(lv_category,), name='market')
    atom_loss = Atom(d, logical_variables=(lv_category, lv_bank), name='loss')
    atom_revenue = Atom(d, logical_variables=(lv_bank,), name='revenue')

    f1 = ParamF(p1, nb=(atom_recession, atom_market))
    f2 = ParamF(p2, nb=(atom_market, atom_loss))
    f3 = ParamF(p3, nb=(atom_loss, atom_revenue))

    rel_g = RelationalGraph()
    rel_g.atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
    rel_g.param_factors = (f1, f2, f3)
    rel_g.init_nb()

    return rel_g


def generate_data(f, rel_g, evidence_ratio):
    data = dict()
    key_list = rel_g.key_list()

    idx_evidence = np.random.choice(len(key_list), int(len(key_list) * evidence_ratio), replace=False)
    for i in idx_evidence:
        key = str(key_list[i])
        data[key] = np.random.uniform(-30, 30)

    with open(f, 'w+') as file:
        file.write(json.dumps(data))


def load_data(f):
    with open(f, 'r') as file:
        s = file.read()
        temp = json.loads(s)

    data = dict()
    for k, v in temp.items():
        data[eval(k)] = v

    return data


if __name__ == "__main__":
    evidence_ratio = 0.01
    rel_g = generate_rel_graph()
    for i in range(5):
        f = str(evidence_ratio) + '_' + str(i)
        generate_data(f, rel_g, evidence_ratio)
