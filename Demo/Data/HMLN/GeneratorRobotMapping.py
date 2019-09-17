from RelationalGraph import *
from MLNPotential import *
import numpy as np
import json


Seg = []
Type = ['W', 'D', 'O']
Line = []

for i in range(1, 38):
    Seg.append(f'A1_{i}')

for i in range(1, 3):
    Seg.append(f'LA{i}')

domain_bool = Domain((0, 1))
domain_length = Domain((0, 1), continuous=True, integral_points=linspace(0, 1, 20))
domain_depth = Domain((0, 0.5), continuous=True, integral_points=linspace(0, 1, 20))

lv_seg = LV(Seg)
lv_type = LV(Type)
lv_line = LV(Line)

atom_PartOf = Atom(domain_bool, logical_variables=(lv_seg, lv_line), name='PartOf')
atom_SegType = Atom(domain_bool, logical_variables=(lv_seg, lv_type), name='SegType')
atom_Aligned = Atom(domain_bool, logical_variables=(lv_seg, lv_seg), name='Aligned')
atom_Length = Atom(domain_length, logical_variables=(lv_seg,), name='Length')
atom_Depth = Atom(domain_depth, logical_variables=(lv_seg,), name='Depth')

f1 = ParamF(
    MLNPotential(lambda x: 1 if x[0] + x[1] + x[2] > 0 else 0, w=np.Inf),
    nb=['SegType(s,$W)', 'SegType(s,$D)', 'SegType(s,$O)']
)
f2 = ParamF(
    MLNPotential(lambda x: 1 if neg_op(x[0]) + neg_op(x[1]) + x[2] + neg_op(x[3]) + neg_op(x[4]) else 0, w=1.591),
    nb=['SegType(s1,$W)', 'SegType(s2,$W)', 'PartOf(s1,l)', 'PartOf(s2,l)', 'Aligned(s2,s1)']
)
f3 = ParamF(
    MLNPotential(lambda x: x[0], w=0.81),
    nb=['SegType(s,$W)']
)
f4 = ParamF(
    MLNPotential(lambda x: x[0], w=-0.737),
    nb=['SegType(s,$D)']
)
f5 = ParamF(
    MLNPotential(lambda x: x[0], w=-0.077),
    nb=['SegType(s,$O)']
)
f6 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.1), w=3.228),
    nb=['SegType(s,$D)', 'Length(s)']
)
f7 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.02), w=2.668),
    nb=['SegType(s,$D)', 'Depth(s)']
)
f8 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.1), w=3.228),
    nb=['SegType(s,$D)', 'Length(s)']
)
f9 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.02), w=2.668),
    nb=['SegType(s,$D)', 'Depth(s)']
)


def generate_rel_graph():
    rel_g = RelationalGraph()
    rel_g.atoms = (atom_PartOf, atom_SegType, atom_Aligned, atom_Length, atom_Depth)
    rel_g.param_factors = (f1, f2, f3, f4, f5, f6, f7, f8, f9)

    return rel_g


# def generate_data(f):
#     data = dict()
#
#     X_ = np.random.choice(num_paper, int(num_paper * 0.2), replace=False)
#     for x_ in X_:
#         data[str(('popularity', f'paper{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)
#
#     X_ = np.random.choice(num_topic, int(num_topic * 0.5), replace=False)
#     for x_ in X_:
#         data[str(('popularity', f'topic{x_}'))] = np.clip(np.random.normal(0, 3), -10, 10)
#
#     X_ = np.random.choice(num_paper, int(num_paper * 0.2), replace=False)
#     for x_ in X_:
#         Y_ = np.random.choice(num_topic, np.random.randint(3), replace=False)
#         for y_ in Y_:
#             data[str(('paperIn', f'paper{x_}', f'topic{y_}'))] = int(np.random.choice([0, 1]))
#
#     X_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
#     for x_ in X_:
#         Y_ = np.random.choice(num_paper, int(num_paper * 1), replace=False)
#         for y_ in Y_:
#             data[str(('cites', f'paper{x_}', f'paper{y_}'))] = int(np.random.choice([0, 0, 0, 1]))
#
#     with open(f, 'w+') as file:
#         file.write(json.dumps(data))


def load_data(f):
    with open(f, 'r') as file:
        s = file.read()
        temp = json.loads(s)

    data = dict()
    for k, v in temp.items():
        data[eval(k)] = v

    return data


# if __name__ == "__main__":
#     rel_g = generate_rel_graph()
#     for i in range(5):
#         f = str(i)
#         generate_data(f)
