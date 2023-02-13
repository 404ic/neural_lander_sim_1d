import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
sys.path.append('../../../')

# Import do_mpc package:
import do_mpc

def quadrotor_mpc(fts, params, n_steps, h0, v0, m=1.47, g=9.81):
    # Initialize environment
    model_type = 'continuous' 
    model = do_mpc.model.Model(model_type)
    fts = [ft.replace('x0','height').replace('x1','dheight').replace('^','**').replace(" ", "*") for ft in fts]
    terms = [str(c)+"*"+ft for c, ft in zip(params, fts)]
    eqn = ' + '.join(terms)
    
    height = model.set_variable('_x', 'height')
    dheight = model.set_variable('_x', 'dheight')
    u = model.set_variable('_u', 'force')
    ddheight = model.set_variable('_z', 'ddheight')

    model.set_rhs('height', dheight)
    model.set_rhs('dheight', ddheight)

    newton = (m*ddheight + m*g - u - eval(eqn))

    model.set_alg('newton', newton)
    E_kin = 1 / 2 * m * dheight**2
    E_pot = m * g * height
    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)
    model.setup()

    # Initialize controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 50, # No. iterations in MPC optimization inner loop
        'n_robust': 0,
        'open_loop': 0,
        't_step': 1e-2,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)

    mterm = model.aux['E_kin'] + model.aux['E_pot']
    lterm = model.aux['E_kin'] + model.aux['E_pot']
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(force=0.1)

    # Inequalitiies (don't hit ground)
    mpc.bounds['lower','_u','force'] = 0
    mpc.bounds['upper','_u','force'] = 50
    mpc.bounds['lower','_x','height'] = 0 # maybe change this to some small epislon

    mpc.setup()

    # Initialiize simulator
    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.01
    }
    simulator.set_param(**params_simulator)
    simulator.setup()
    simulator.x0['height'] = h0
    simulator.x0['dheight'] = v0

    x0 = simulator.x0.cat.full()

    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    # Run MPC optimization loop
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
    
    return mpc.data['_x'], mpc.data['_u']