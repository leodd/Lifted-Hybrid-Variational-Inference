import utils

seed = 0
utils.set_seed(seed=seed)
utils.set_path()

from RelationalGraph import *
from MLNPotential import *

instance = [
    'Joey',
    'Tim',
]

data = {
    ('Smoke', 'Tim'): -2,
}

domain_cont = Domain((-3, 3), continuous=True)

lv_x = LV(instance)
lv_y = LV(instance)

atom_smoke_x = Atom(domain=domain_cont, logical_variables=[lv_x], name='Smoke')
atom_smoke_y = Atom(domain=domain_cont, logical_variables=[lv_y], name='Smoke')

f1 = ParamF(
    MLNPotential(lambda atom: eq_op(atom[0], atom[1])),
    nb=[atom_smoke_x, atom_smoke_y],
    constrain=lambda sub: sub[lv_x] > sub[lv_y]
)
f2 = ParamF(
    MLNPotential(lambda atom: -0.1 * atom[0] ** 2),  # just to help ensure normalizability
    nb=[atom_smoke_x]
)

rel_g = RelationalGraph()
rel_g.atoms = (atom_smoke_x, atom_smoke_y)
rel_g.param_factors = (f1, f2)
rel_g.init_nb()

rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

from OneShot import OneShot

grad_check = False
if not grad_check:
    K = 5
    T = 20
    lr = 1e-1
    its = 500
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

    from mixture_beliefs import calc_marg_log_prob

    plt.figure()
    rvs = g.rvs_list
    xx, yy = np.meshgrid(np.linspace(rvs[0].values[0], rvs[0].values[1]),
                         np.linspace(rvs[1].values[0], rvs[1].values[1]))
    zs = np.exp(calc_marg_log_prob(g, X=np.vstack([xx.ravel(), yy.ravel()]).T, obs_rvs=rvs, params=res))
    zz = zs.reshape(xx.shape)
    num_levels = 10
    plt.contourf(xx, yy, zz, num_levels)
    plt.colorbar()
    plt.title('final mixture belief; bfe = %g' % record['bfe'][-1])
    plt.xlabel('$x_%d$' % rvs[0].id)
    plt.ylabel('$x_%d$' % rvs[1].id)
    save_name += '_beliefs'
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
