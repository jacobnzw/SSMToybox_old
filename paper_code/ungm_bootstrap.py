from models.ungm import UNGM


# setup an SSM
ssm = UNGM(x0_cov=5)

# generate ground-truth state and measurement trajectories
num_steps, num_mc = 500, 100
x, y = ssm.simulate(steps=num_steps, mc_sims=num_mc)

# TODO: try and finish this demo
# repurposed Simo's MATLAB code, might wanna do it in Python later though

