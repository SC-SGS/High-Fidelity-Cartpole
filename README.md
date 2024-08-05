# High-Fidelity-Cartpole

This gymnasium compatible environment implements a realistic Cartpole simulation of 
the Quanser IP-02 Inverted Pendulum hardware.

## Usage

Place the `env_config.gin` inside your working directory.

Call `pip install -e /relative_path_to_cartpole_realistic_dir` inside your working director.

To use the environment within python, you only need three lines of code:

```
import cartpole_realistic
import tf_agents
env = tf_agents.environments.suite_gym.load('cartpole-realistic')

```

## Citation (BibTeX)

```
@inproceedings{bantel2024high,
  title={High-Fidelity Simulation of a Cartpole for Sim-to-Real Deep Reinforcement Learning},
  author={Bantel, Linus and Domanski, Peter and Pfl{\"u}ger, Dirk},
  booktitle={2024 4th Interdisciplinary Conference on Electrics and Computer (INTCEC)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}

```
