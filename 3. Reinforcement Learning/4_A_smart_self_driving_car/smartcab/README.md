# A Smart self driving Car
A smart car that will be able to learn the  traffic rules of the roads, by driving itself through the road.

#Description: 
In this project, we use the Reinforcement Learning (Q-Learning ), to train our smart car how to dirve. 
Once done, the car will be able to learn the traffic light rules, intersection rules, based on the other vehicles movement.

Our smart cab will operate in an idealised grid like city, with  roads going North-South and East-West. 
Other vehicles may be present on the roads, but no pedestrians. There is a traffic light at each 
intersection that can be in one of two states: North-South open or East-West open.

US right-of-way rules applies for other car's navigation i.e On a green light, you can turn left only if there is no oncoming traffic at 
the intersection coming straight. On a red light, you can turn right if there is no oncoming traffic 
turning left or traffic from the left going straight. Our car shall be able to learn this rule, once trained.

## Getting Started Guide:

### Step 1: Install

This project requires Python 2.7 with the pygame library installed:

https://www.pygame.org/wiki/GettingStarted

### Step 2: Run
Make sure you are in the top-level project directory `smartcab/` (that contains this README). Then run:

```python smartcab/agent.py```

OR:

```python -m smartcab.agent```
