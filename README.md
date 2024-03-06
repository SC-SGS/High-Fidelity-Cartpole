# High-Fidelity-Cartpole

This gymnasium compatible environment implements a realistic Cartpole simulation of 
the Quanser IP-02 Inverted Cartpole hardware.

## Usage

Place the `env_config.gin` inside your working directory.

Call `pip install -e /relative_path_to_cartpole_realistic_dir` inside your working director.

To use the environment within python, you only need three lines of code:

```
import cartpole_realistic
import tf_agents
env = tf_agents.environments.suite_gym.load('cartpole-realistic')

```
