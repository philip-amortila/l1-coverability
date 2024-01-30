Commands to run:

big version:

python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --num_rollouts=1000 --episodes=15000 --epochs=51 --exp_runs=5 --exp_name=mountaincar_test --replicate=0 --save_models --measurements='el'

smaller version:

python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --num_rollouts=100 --episodes=10000 --epochs=31 --exp_runs=5 --exp_name=mountaincar_noleak_smalleps_smallinit --replicate=0 --save_models --measurements='e' 

Things to change in the flags:

- Change exp_name for the name of the experiment (this is where data will get stored)

- Change exp_runs for the number of experimental runs you will do (e.g. 5 or 10)

- At each run change the replicate number from 0 to exp_runs-1

Things to change in the code:

- sys.path in Line 10 (at time of writing) of collect_baseline.py. 

- FIG_DIR in Line 23 (at time of etc.) of plotting.py

- MODEL_DIR in Line 654 (etc.) of collect_baseline.py 