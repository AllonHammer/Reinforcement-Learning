# Lunar Lander Agent
This project implements a reinforcement learning agent to interact 
with the Lunar Lander environment. As the name suggests, the task 
is to land a spaceship (agent) on the moon. The environemnt obeys basic 
physics (and gravity). 


####Functions of the environment
1. get_velocity - returns an array representing the x, y velocity of the lander. Both the x and y velocity are in the range [0,60]
2. get_angle - returns a scalar representing the angle of the lander. The angle is in the range [0,359]
3. get_position - returns an array representing the x, y position of the lander. Both the x and y position of the agent are in the range [0,100]
4. get_landing_zone - returns an array representing the x, y position of the landing zone. Both the x, y coordinates are in the range [1,100]
5. get_fuel - returns a scalar representing the remaining amount of fuel. Fuel starts at 100 and is in range [0,100]

####Actions with the environment
The spaceship can interact with the environment 
using one of its three engine thrusters :
1. main-  acceleration in the negative Y- axis
2. left-  acceleration in the positive X- axis
3. right- acceleration in the negative X- axis

####The lander will crash if
1. it touches the ground when y_velocity < -3 (the downward velocity is greater than three). 
2. it touches the ground when x_velocity < -10 or 10 < x_velocity (horizontal speed is greater than 10).
3. it touches the ground with  5 < angle < 355 (angle differs from vertical by more than 5 degrees).
4. it runs out of fuel 
5. it touches the ground when x_position ∉ landing_zone (it lands outside the landing zone)

![Screenshot](lunar_landar.png)


#### Reward
This is an episodic task. Performance is measured as the avg reward per episode across the experiment.
This experiment is repeat several times (each time with a different random seed) and the results are averaged.
1. A big positive reward will be given for landing successfully
2. A big negative reawrd will be given for crashing
3. A small negative reward will be given for fuel consumption

## Quick Setup


#### Prerequisites

    * Python >= 3.5

> **NOTICE:** [`pyenv`](https://github.com/pyenv/pyenv) is recommended for python installation.

#### Setting up the project

Set up virtualenv

```sh
$ make virtualenv
$ source venv/bin/activate
```

(You can wipe out virtualenv by running;)

```sh
$ make cleanenv
```

Once project is set, export its PYTHONPATH.

```sh
$ pip install -r requirements.txt
```



#### Usage


```sh
$ python main.py 
 37%|███████████████▎                         | 112/300 [02:17<07:40,  2.45s/it]

```


