import utils

seed = 0
utils.set_path()
utils.set_seed(seed=seed)

from RelationalGraph import *
from MLNPotential import *

instance = [
    'Joey',
    'Rachel',
    'Tim',
]

data = {
    ('Friend', 'Joey', 'Rachel'): 1,
    ('Friend', 'Joey', 'Tim'): 1,
    ('Friend', 'Rachel', 'Joey'): 1,
    ('Friend', 'Rachel', 'Tim'): 0,
    ('Friend', 'Tim', 'Joey'): 1,
    ('Friend', 'Tim', 'Rachel'): 0,
    ('Smoke', 'Tim'): 7,
}

domain_bool = Domain((0, 1))
domain_cont = Domain((0, 10), continuous=True)

lv_x = LV(instance)
lv_y = LV(instance)

# formulas:
# Friend(X, Y) => [Smoke(X) = Smoke(Y)]
# Smoke(X) => Cancer(X)

atom_friend = Atom(domain=domain_bool, logical_variables=[lv_x, lv_y], name='Friend')
atom_smoke_x = Atom(domain=domain_cont, logical_variables=[lv_x], name='Smoke')
atom_smoke_y = Atom(domain=domain_cont, logical_variables=[lv_y], name='Smoke')
atom_cancer = Atom(domain=domain_bool, logical_variables=[lv_x], name='Cancer')

f1 = ParamF(
    MLNPotential(lambda atom: imp_op(atom[0], eq_op(atom[1], atom[2]))),
    nb=[atom_friend, atom_smoke_x, atom_smoke_y],
    constrain=lambda sub: sub[lv_x] > sub[lv_y]
)
f2 = ParamF(
    MLNPotential(lambda atom: imp_op(atom[0], atom[1])),
    nb=[atom_smoke_x, atom_cancer]
)

rel_g = RelationalGraph()
rel_g.atoms = (atom_friend, atom_smoke_x, atom_cancer)
rel_g.param_factors = (f1, f2)
rel_g.init_nb()

rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

from OneShot import OneShot

grad_check = True
if not grad_check:
    K = 3
    T = 20
    lr = 1e-2
    its = 600
    osi = OneShot(g=g, K=K, T=T, seed=seed)
    res = osi.run(lr=lr, its=its)
    record = res['record']
    del res['record']
    print(res)
    for key, rv in rvs_table.items():
        if rv.value is None:  # only test non-evidence nodes
            print(rv, key, osi.map(rv))

    import matplotlib.pyplot as plt

    for key in record:
        plt.plot(record[key], label=key)
    plt.legend(loc='best')
    save_name = __file__.split('.py')[0]
    plt.savefig('%s.png' % save_name)

    # EPBP inference
    from EPBPLogVersion import EPBP

    bp = EPBP(g, n=50, proposal_approximation='simple')
    bp.run(30, log_enable=True)

    for key, rv in rvs_table.items():
        if rv.value is None:  # only test non-evidence nodes
            print(key, bp.map(rv))

else:
    K = 3
    lr = 1e-2
    its = 1
    import tensorflow as tf
    for T in [10, 20, 50, 100]:
        print('grad check, T =', T)
        utils.set_seed(seed)
        osi = OneShot(g=g, K=K, T=T, seed=seed)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        res = osi.run(lr=lr, optimizer=optimizer, its=its, grad_check=True)
        print()
