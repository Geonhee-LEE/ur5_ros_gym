# Reinforcement-Learning Framework
A reinforcement-Learning Framework for testing learning agents on a UR5 manipulator. The framework consists of 2 ROS packages which are [rl_gazebo_sim](rl_gazebo_sim), [rl_moveit_config](rl_moveit_config). Besides the two ROS packages there is the folder [rl-gym](rl-gym) with contains an OpenAI Gym environment as well as python scripts for runnin reinforcement learning. The rl-gym folder is an implementation example on how to use the framework with OpenAI Gym

The framework is designed and prepared for screwing tasks meanning the simulated UR5 is fitted with a screw tool and the world is fitted with a block with a screw.

## Content
[About](#about)  
[Installation](#installation)  
[Launch example aplication](#launch-example-aplication)  

## About
This is repository is by as a 6th semester project at Aalborg University.

## Installation
Install ROS Kinetic following [this guide](http://wiki.ros.org/kinetic/Installation/Ubuntu) from the ROS wiki.  
Install ROS dependencies
```bash
sudo apt-get  install  python-rosinstall  python-rosinstall-generator  python-wstool  build-essential
sudo apt-get  install ros-kinetic -moveit
sudo apt-get  install ros-kinetic-ros-control ros-kinetic-ros-controllers  ros-kinetic-gazebo-ros ros-kinetic-gazebo-ros-control 
 ros-kinetic-controller-manager
```

Install OpenAIGym.
```
pip  install  gym ==0.10.5
```

## Launch example aplication
In order to launch the framework three things must be launched. First the Gazebo simulation.
```bash
roslaunch rl_gazebo_sim rl_world.launch
```

Then launch the controller
```bash
roslaunch rl_moveit_config moveit_planning_execution.launch
```

Now the framework is ready to be used. In order to launch the example aplication run the following python script.
```
python  ~/catkin_ws/src/rl_project/rl_gym/simple_loop.py
```
You shold now be running a simple Q-learning algorithm 
