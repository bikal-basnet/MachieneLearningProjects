import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy
#as .random import choice

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None
        self.color = 'red'

    def update(self, t):
        # Gather inputs
        #self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.next_waypoint = numpy.random.choice(['forward', 'left','right',None])

        curent_env_as_observed_by_our_car = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO 1.0 : Update state
        # Our current state is this, now in the next to do section TODO 2.0
        # we define which way we will be going, then in the next section 3.0
        # we will take that step as defined in TODO 2.0 and then reach a new state
        self.state = {'location': self.env.agent_states[self]['location'],
                      'heading': self.env.agent_states[self]['heading']}

        print("curent_env_as_observed_by_our_car is :: \n {}. \n Next action is  :: '{}'. Deadline is  :: '{}'. Time is {}"\
              "\n State prior to taking action is :: {}"
              .format(curent_env_as_observed_by_our_car, self.next_waypoint, deadline, t,self.state))

        # TODO 2.0 : Select action according to your policy
        action = self.next_waypoint

        reward = self.env.act(self, action)


        if (self.env.done == True):
            print('Reached Destination')
        # Section 3.0 : Execute action and get reward
        # after we execute a certain action from the state, we also get into a new state.



        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward for taking the action = {}"\
            .format(deadline, curent_env_as_observed_by_our_car, action, reward)  # [debug]







def run():
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
