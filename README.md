Companion repo for the paper "Scalable Online Exploration via Coverability"

Commands to recreate our experiment

Mountaincar:

python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --num_rollouts=100 --episodes=1000 --epochs=20 --exp_runs=10 --exp_name=mountaincar_test --replicate=0 --save_models --measurements='el' --reg_eps=1e-4

Pendulum:

python collect_baseline.py --env="Pendulum-v0" --T=200 --train_steps=400 --num_rollouts=100 --episodes=1000 --epochs=20 --exp_runs=10 --exp_name=pendulum_test --replicate=0 --save_models --measurements='el' --reg_eps=1e-4

To plot: 

Mountaincar:
python plotting.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --num_rollouts=100 --episodes=1000 --epochs=20 --exp_runs=10 --exp_name=mountaincar_test --replicate=0 --save_models --measurements='el' --reg_eps=1e-4

and

Pendulum:
python plotting.py --env="Pendulum-v0" --T=200 --train_steps=400 --num_rollouts=100 --episodes=1000 --epochs=20 --exp_runs=10 --exp_name=pendulum_test --replicate=0 --save_models --measurements='el' --reg_eps=1e-4

Things to install:

cd into the gym-fork folder and run ``pip install -e .''

Flags:

- Change exp_name for the name of the experiment (this is where data will get stored)

- Change exp_runs for the number of experimental runs you will do (e.g. 5 or 10)

- At each run change the replicate number from {0,1,...,exp_runs-1}

Things to change in the code:

- sys.path in Line 4 (at time of writing) of collect_baseline.py. 