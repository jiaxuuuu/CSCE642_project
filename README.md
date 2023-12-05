# Project Title

Final Project File for CSCE-642 Deep Reinforcement Learning
Feasibility Study of Inverse Reinforcement Learning for Interpreting Construction Worker Wayfinding Behavior

## Authors
Minguk Kim, Ph.D. Student, Dept. of Construction Science, Texas A&M University, College Station, TX 77843 USA, UIN: 934007250, EMAIL: mingukkim@tamu.edu\
Jia Xu, Ph.D. Student, Dept. of Construction Science, Texas A&M University, College Station, TX 77843 USA, UIN: 334002354, EMAIL: jiaxu@tamu.edu

## Human record collection
Please note that all the relevant files mentioned in this section are located in the `Experiment` folder. The Unity package named `ConstructionSite.unitypackage` is used in this section.

### Unity3D project set up
1. **Unity Editor**: Unity Editor installed (`2022.3.7f1`) was used in this project.
2. **Create a new HDPR projet**: create a new HDPR project.
3. **Download and Import Package**: import `ConstructionSite.unitypackage` into your Unity project.
4. **Open Scene**: Locate and open the scene named `OutDoorScene`.

### Data recording
In this project, the player is represented as a cube, and the target they must reach is depicted as a sphere. The objective is to control the cube using keyboard inputs to navigate towards the target sphere.
1. **How to play**: Launch the game and press the play button to begin.
2. **Controlling the Player**: Use the keyboard to control the cube's movement. There are 8 discrete actions corresponding to specific keys. Each key press moves the cube by a fixed stride of 0.3 meters.

![Alt text](image.png)

3. **Recording and Data Export**: When the cube reaches close to the target sphere, the game automatically stops, and the trajectory data of that particular movement is recorded.The trajectory data is saved in the project's asset folder. The files are named following the convention "ExportedData_trajectorynumber.csv".Alongside the data, a picture capturing the trail of the trajectory is also saved, named as "AssetsExportedData_trajectorynumber.png".
4. **Moveinto Next trajectory**: Press the "Next Try" button located at the bottom-left corner of the canvas to reset the player to its initial position. This action prepares the game for the next attempt. Upon starting the next try, the game will record the data of the new trajectory, similar to previous attempts.

## IRL implemention for parameter estimation
### Installation

Instructions on how to install any dependencies required for the project.
Please note that all the relevant files mentioned in this section are located in the `IRLforCSCE642` folder.

```bash
# Example installation steps
pip install -r requirements.txt
```

### Usage

Here's how you can run the project.

#### Step 1: Preprocessing

- **File**: `step_1_preprocessing_agg.ipynb`, `step_1_preprocessing_con.ipynb`
- **Description**: In this step, we use the recorded files to create the necessary files before performing MIRL. See the code-specific comments for details. If you want to put in new records and analyse them, you can do something similar to 'Record/Aggresive/Data/ExportedData_0.csv', then drop in the new files and follow these steps. As of now, execute the files below separately.

To run this step, execute the following:

```bash
jupyter notebook step_1_preprocessing_agg.ipynb
jupyter notebook step_1_preprocessing_con.ipynb
```

#### Step 2: Modular Inverse Reinforcement Learning
- **File**: `step_2_inverserl.ipynb`
- **Description**: This notebook contains information or code derived from [this GitHub repository](https://github.com/corgiTrax/Modular-Reinforcement-Learning-Driving/blob/master/inverseRL.py). Once you execute the file, you can estimate reward and gamma value for each module(subtask) and visualized results including bar graph and heatmap. Please refer to code-specific comments for details.

To run this step, execute the following:

```bash
jupyter notebook step_2_inverserl.ipynb
```

### Modules

#### Data Processing
- **File**: `dataprocessing.py`
- **Description**: This script includes imports for numpy and pandas, indicating its use in data processing tasks.

#### Modular IRL
- **File**: `modularirl.py`
- **Description**: The script includes scipy optimization and math operations, suggesting its role in computational tasks related to Inverse Reinforcement Learning.

### Visualization

Scripts for visualizing the results.

#### Heatmap Generation
- **File**: `heatmap.py`
- **Description**: The script includes pandas, numpy, matplotlib, and scipy, indicating its functionality for generating heatmaps.

#### Histogram Visualization
- **File**: `histovis.py`
- **Description**: This script uses matplotlib for plotting, likely to generate histogram visualizations.

#### Dataframe Visualization
- **File**: `dataframevis.py`
- **Description**: Utilizes matplotlib, seaborn, numpy, and scipy for dataframe visualization, possibly providing statistical plot representations.


## IRL agent trajectories collection
Please note that all the relevant files mentioned in this section are located in the `Experiment/Communication` folder. 

The Q-value function of the agent is calculated based on the average reward and discount factor obtained from human data using IRL. These parameters guide the agent's decision-making process for action selection. Utilizing a socket connection, the agent interacts with the Unity3D environment. This setup allows for the recording of the agent's trajectory in a manner analogous to the collection of human trajectory data.

1. **Change parameters accordingly**: Change the discount factors and rewards of each module according to the result of IRL in `ConnectTest.py`. Based on the test episode wanted, change the parameter of `num_episodes`.
2. **How to play**: Launch the unity game and press the play button to begin. Then, run the `ConnectTest.py` to connect the unity environment with the python algorithm.
3. **Communication**：The agent selects actions based on the current state's Q-value, utilizing a SoftMax action selection function. Once an action is chosen, it is communicated to the Unity environment. The agent then moves according to this selected action within Unity, allowing for the recording of its trajectory. 
4. **Data collection**: The trajectory data for the specified number of episodes (num_episode) will be recorded. Each set of trajectory data and its corresponding trail image will be saved in the same format and location as described in the human data recording section. 

## Contributing
Instructions for contributing to the project. (If applicable.)

## License
Copyright <2023> <COPYRIGHT Minguk Kim and Jia Xu>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.