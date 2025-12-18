# Hierarchical Navigation for ANYmal-C on Rough Terrain

## Overview
This project implements a **Hierarchical Reinforcement Learning (HRL)** framework for the ANYmal-C quadruped in **NVIDIA Isaac Lab**. It moves beyond standard flat-ground navigation by training a high-level planner to navigate complex, rough terrains populated with obstacles, using a pre-trained robust locomotion policy.

<video src="media/rl-video-step-10000.mp4" width="640" height="480" controls></video>
<video src="media/Navigation_Flat_single.mp4" width="640" height="480" controls></video>

## Key Achievements

* **Hierarchical Architecture:** Decoupled control into a High-Level Policy (Navigation/Path Planning) and a Low-Level Policy (Robust Rough-Terrain Locomotion).
* **Rough Terrain Adaptation:** Successfully transferred the navigation task from a flat plane to complex **height-field terrains**, utilizing a locomotion checkpoint specifically trained for uneven ground.
* **Natural Gait Engineering:** Designed custom reward functions to enforce realistic movement:
    * **`face_target`:** Eliminates unnatural "strafing" by forcing the robot to orient towards the goal while moving.
    * **Actuation Penalties:** Minimizes energy usage and "jerkiness" for smoother sim-to-real transfer.
* **Obstacle Avoidance:** Augmented the environment with static obstacles, training the agent to plan trajectories that avoid collisions while reaching dynamic targets.

## Technical Details

### Architecture
* **Low-Level:** Pre-trained velocity-tracking policy (Robust to noise and terrain irregularities).
* **High-Level:** Custom PPO policy outputting velocity commands ($v_x, v_y, \omega_z$) based on exteroceptive terrain data and goal relative position.

### Custom Rewards
Implemented in `rewards.py` to align heading with the target vector:
```python
def face_target(env, command_name):
    # Penalizes the angular difference between robot forward vector and target vector
    command = env.command_manager.get_command(command_name)
    target_angle = torch.atan2(command[:, 1], command[:, 0])
    return torch.abs(target_angle)
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Anymal_Navigation

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```