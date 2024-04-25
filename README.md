# Implementation of a Deep Reinforcement Learning Algorithm in the VizDoom Environment

## Author: [Najib El khadir](https://github.com/NajibXY)
## French Version [README-fr](https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/README-fr.md)

## 1. Motivations

During my second year of CS Master Degree specializing in Artificial Intelligence, I had the opportunity to learn about a Python library providing training environments for learning agents: VizDoom. Therefore, as part of my experiments in April 2024, I decided to implement a deep reinforcement learning algorithm and test it on VizDoom environments.

</br> </br>

## 2. Technologies Used
![](https://skillicons.dev/icons?i=python,pytorch,anaconda)
- Python 3.12, PyTorch, VizDoom, Conda (for my personal environment)

## 3. References
- [VizDoom Scenarios and Environments](https://vizdoom.farama.org/environments/default/)
- [Basic example from the FARAMA Foundation on using VizDoom](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/learning_pytorch.py)
- [What is Deep Reinforcement Learning](https://www.v7labs.com/blog/deep-reinforcement-learning-guide)

## 4. [DQL.py](https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/dql.py) File

- The implementation is, for the moment, entirely done in the file [DQL.py](https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/dql.py).
</br></br>
- This script is structured as follows:
  
### Environment Parameters
- Definition of VizDoom configurations to load.
- Since the environment is perceived by the agent as a set of pixels, a resolution used in VizDoom and a repetition window need to be defined.

### Replay Memory
- A Replay Memory class is implemented.
- This class is used to store recent transitions (source state, action, resulting state, reward).
- This class also tracks terminal actions (ending the current episode).

### QNN Class
- This class is the core of the deep learning algorithm.
- It creates a convolutional neural network with 1 input layer, 1 output layer, and 2 main layers.
- It enables the agent to select the best current action at a given step, perform a step, and adjust its behavior based on the Replay Memory and the reward.

### Utility Functions
- There are also functions for processing the current image of the VizDoom window, initializing the simulation, and knowing its state.
- A function is also dedicated to running demo episodes once the training is finished. This requires changing the appropriate parameters in the main loop of the script.
- The FLAGS part of the main function also allows tuning training parameters (number of episodes, number of iterations per episode, batch size, learning rate, etc.). Feel free to experiment with this part.

## 5. Agent Training

### Setting Up Your Environment

- You'll need to set up a Python environment (in my case conda), preferably 3.12 to avoid library compatibility issues.
- To install dependencies with conda, you can simply run:
  > conda create --name `<your_env_name>` --file requirements.txt
- Or if you're using pip:
  > pip install -r requirements.txt
- Then, you can execute the script.

### Running the Python Script

> python .\dql.py
- This will start training the agent on the `basic.cfg` environment.
- The training results are stored in a `saved_model_doom.pth` file.
- Once training is finished, you can run tests on your trained agent by setting the FLAGS "skip_training" and "load_model" to `True` in the main function of the script.
</br></br>
- After several tests with different settings, here are some "satisfactory" results that were obtained.
  
### Examples 

+ Example of a trained agent after 20 episodes of 2000 iterations on a `basic.cfg` environment 

  <img src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/basic_conf_fullscreen.gif" width="400">
  </br>
    - The agent has indeed learned to move from right to left and react to its environment at the right moment by shooting the target. 
  </br>
  
+ Example of a trained agent after 20 episodes of 2000 iterations on a `defend_the_center.cfg` environment 

  <img src="https://github.com/NajibXY/ViZDoom-Deep-Q-Learning-implementation/blob/master/assets/defend_the_center.gif" width="350">
  </br>
  - Here, the agent had more difficulty finding a reliably human strategy.
  </br>  
  - Nevertheless, we can observe that the inferred behavior is rather consistent: defending the center by turning and shooting (spam strategy!?)

</br></br>

## 5. Possible Improvements

- Tuning the algorithm to better adapt it to different environments.
- Continue experiments on other environments.
- Experiment with other deep reinforcement learning models.
- Develop a GUI or a CLI to allow entering parameters and the learning method, the training environment, etc.
- [...]
