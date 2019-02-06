# skill-based-exploration-drl
code for reproducing results in paper on skill based exploration

## To train an DDPG agent with e-normal exploration 

$ python HER/examples/run.py --env-id=<env name>  --eval-env-id=<test env name> --log-dir=<give logging dir path> --nb-rollout-steps=800 --nb-epochs=200

## To train parameterized action space agent with hierarchical hindsight experience replay

$python HER/examples/prun.py --env-id=<env name> --eval-env-id=<test env name>  --log-dir=<give logging dir path>  --commit-for=10 --nb-rollout-steps=80 --nb-epochs=200 --skillset=set14 --her

## To train the DDPG with skill library(will need to modify skill library paths in skills/set<set number>.py  )

$python HER/examples/succlookaheadrun.py --env-id=<env name> --eval-env-id=<test env name>  --log-dir=<give logging dir path> --commit-for=10 --nb-rollout-steps=800 --nb-epochs=200 --skillset=<skill set>


## Some example commands used to train picknmove env
$ python HER/examples/run.py --env-id=picknmove-v2  --eval-env-id=picknmovet-v2 --log-dir=$HOME/new_RL3/corl_paper_results/clusters-v1/picknmove-v2/run1 --nb-rollout-steps=800 --nb-epochs=200

$ python HER/examples/succlookaheadrun.py --env-id=picknmove-v2 --eval-env-id=picknmovet-v2  --log-dir=$HOME/new_RL3/corl_paper_results/clusters-v1/picknmoveflat-v2/run1 --actor-lr=0.01  --critic-lr=0.01 --commit-for=10 --nb-rollout-steps=800 --nb-epochs=200 --skillset=set14

$ $python HER/examples/prun.py --env-id=picknmove-v2 --eval-env-id=picknmovet-v2  --log-dir=$HOME/new_RL3/corl_paper_results/clusters-v1/picknmovehie-v2/run1  --commit-for=10 --nb-rollout-steps=80 --nb-epochs=200 --skillset=set14

## Acknowledgments
Our code is built over the high quality baselines DDPG implementation.
We would like to thanks all the contributors to OpenAI gym, baselines and mujoco_py

@misc{baselines,
  author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
  title = {OpenAI Baselines},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openai/baselines}},
}
