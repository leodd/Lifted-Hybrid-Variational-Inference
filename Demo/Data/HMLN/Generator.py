from RelationalGraph import *
from MLNPotential import *
import numpy as np
import json


num_paper = 100
num_topic = 10

Paper = []
for i in range(num_paper):
    Paper.append(f'paper{i}')
Topic = []
for i in range(num_topic):
    Topic.append(f'topic{i}')

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(-15, 15, 20))

lv_paper = LV(Paper)
lv_paper_2 = LV(Paper)
lv_topic = LV(Topic)

atom_paper_popularity = Atom(domain_real, logical_variables=(lv_paper,), name='popularity')
atom_topic_popularity = Atom(domain_real, logical_variables=(lv_topic,), name='popularity')
atom_paperIn = Atom(domain_bool, logical_variables=(lv_paper, lv_topic), name='paperIn')
atom_paperIn_2 = Atom(domain_bool, logical_variables=(lv_paper_2, lv_topic), name='paperIn')
atom_cites = Atom(domain_bool, logical_variables=(lv_paper, lv_paper_2), name='cites')

f1 = ParamF(
    MLNPotential(lambda x: imp_op(x[0] * x[1], x[2]), w=1),
    nb=(atom_cites, atom_paperIn, atom_paperIn_2),
    constrain=lambda sub: sub[lv_paper] != sub[lv_paper_2]
)
f2 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=0.01),
    nb=(atom_paperIn, atom_paper_popularity, atom_topic_popularity)
)


def generate_rel_graph():
    rel_g = RelationalGraph()
    rel_g.atoms = (atom_cites, atom_paperIn, atom_paper_popularity, atom_topic_popularity)
    rel_g.param_factors = (f1, f2)
    rel_g.init_nb()

    return rel_g


def generate_data(f):
    data = dict()

    X_ = np.random.choice(num_paper, int(num_paper * 0.2), replace=False)
    for x_ in X_:
        data[str(('popularity', f'paper{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)

    X_ = np.random.choice(num_topic, int(num_topic * 0.5), replace=False)
    for x_ in X_:
        data[str(('popularity', f'topic{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)

    X_ = np.random.choice(num_paper, int(num_paper * 0.2), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_topic, np.random.randint(3), replace=False)
        for y_ in Y_:
            data[str(('paperIn', f'paper{x_}', f'topic{y_}'))] = int(np.random.choice([0, 1]))

    X_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
        for y_ in Y_:
            data[str(('cites', f'paper{x_}', f'paper{y_}'))] = int(np.random.choice([0, 0, 0, 1]))

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
    rel_g = generate_rel_graph()
    for i in range(5):
        f = str(i)
        generate_data(f)
