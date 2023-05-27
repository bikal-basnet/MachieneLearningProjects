# References:
# 1. https://www-s.acm.illinois.edu/sigart/docs/QLearning.pdf (pg 6)

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy
import collections
#as .random import choice
import time
from pprint import pprint



class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.all_actions = ['left','right','forward', None]
        self.q_value_state_action_dict = dict()
        #self.alpha = 0.6  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
        #self.discount = 0.5  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
        #self.epsilon = 0.2   # more epsilon value more random action choice, less epsilon value then more best action choosen but at the expense of more computation

        #self.exploration_of_unknown_state_actions = 5 # reward  for exploring is more than the moving into the optimal solution i.le left ,right or forward in  any state of light or incoming traffic
        self.reached_destination_count = 0
        self.wait_seconds_prior_to_next_run = 0
        self.run_iteration = 0
        self.total_reward = 0
        #self.urgency = 'low'
        self.initial_q_0_value = 0
        self.agent_has_reached_destination_in_cur_iteration = 0

    def get_agents_final_status(self):
        print('/*************************** get status call ENDS : Destination Reached Status ::', self.reached_destination_count,
              '/ (last 20 of ',self.run_iteration,') . Total Reward : ', self.total_reward, ' run_iteration', self.run_iteration,
              '****************************************/')
        return {'self.reached_destination_count' : self.reached_destination_count,
                'self.total_reward':self.total_reward,
                'self.run_iteration':self.run_iteration
                }

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.agent_has_reached_destination_in_cur_iteration = 0
        self.state = None
        #self.next_waypoint = None
        self.color = 'red'
        self.previous_state = ''
        self.action = ''
        print('')
        print('Prior to resetting learned q value is ' )
        print(self.q_value_state_action_dict)
        self.run_iteration = self.run_iteration+1
        self.original_deadline = self.env.get_deadline(self)
        self.total_reward = 0
        time.sleep(self.wait_seconds_prior_to_next_run)

        #self.epsilon = 3.0 / self.run_iteration
        # if self.epsilon < self.original_deadline * 0.25:
        #     self.epsilon = 0.8
        # elif self.epsilon < self.original_deadline * 0.75:
        #     self.epsilon = 0.1
        # else:
        #     self.epsilon = 0.001



    def update(self, t):
        # Gather inputs
        # Tells us which action to take, if so what will we learn then from the q learning.
        curent_env_as_observed_by_our_car = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO 1.0 : Update state
        # WE will create the generic state, independent of the location and the heading, that can be used across all,,
        # else if we consider the location and the heading then we will not be able to learn the generic rules, but willlearn individiual instatnce speciif rules
        # 'location': self.env.agent_states[self]['location'],
        # 'heading': self.env.agent_states[self]['heading'],
        # converting the state to immutable objects to be used as dictionary key
        state_tuple = collections.namedtuple('state_tuple','light oncoming left right')
        self.state = state_tuple(
                      light = curent_env_as_observed_by_our_car['light']
                      ,oncoming = curent_env_as_observed_by_our_car['oncoming']
                      ,left = curent_env_as_observed_by_our_car['left']
                      ,right= curent_env_as_observed_by_our_car['right']
                      )

        # TODO: Learn policy based on state, action, reward
        action = self.get_best_action_via_qlearning(self.state)

        # Ref (1): Observe the current state s
        self.previous_state = self.state

        # Ref (1) : Select an action a and execute it
        reward = self.env.act(self, action)  # Ref 1: Receive immediate reward r

        self.total_reward = self.total_reward +  reward
        #Ref 1 : Observe the new state s'
        self.state = self.state

        # Ref 1 : Update the table entry for Q(s,a)    # T(s,a,r,s')
        self.update_q_table_entry(self.previous_state, action, reward, self.state)


        if ( self.run_iteration >= 80   and (self.env.done == True or self.run_iteration == 100  or self.run_iteration == 50) ):
            if self.agent_has_reached_destination_in_cur_iteration == 0:
                self.reached_destination_count = self.reached_destination_count + 1;
                self.agent_has_reached_destination_in_cur_iteration = 1
            print('/*************************** ENDS : Destination Reached Status ::',self.reached_destination_count,'/ (last 20 of ',self.run_iteration,') . Total Reward : ',self.total_reward,' run_iteration',self.run_iteration,'****************************************/')



        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward for taking the action = {}"\
            .format(deadline, curent_env_as_observed_by_our_car, action, reward)  # [debug]
        print('')



    # Ref (1)
    # For each state -action pair(s,a), initialise the table entry Q(s,a) to zero
    def get_q_value_for_state_action_pair(self, state, action):
        if (state, action) not in self.q_value_state_action_dict:
            return self.initial_q_0_value # Setting Q0 value to higher than the destination reward, so that we enforce the agent to learn all the rules first
        else:
            return self.q_value_state_action_dict[(state, action)]

    def get_best_action_via_qlearning(self, state):
        return self._get_best_q_state(state)['best_action']

    def get_q_value_for_best_action_at(self, state):
        return self._get_best_q_state(state)['best_q_value']


    # returns q-state{ 'best_action'  : 'xxxxx'   'best_q_value': 'xxx'# }
    def _get_best_q_state(self, state):
        # Now for each action compute q function, over time t and take whichever action gives us more reward
        # With probability Epsilon, choose random action i.e more Epsilon, then we explore more and with 1- epsilon, we choose the best action
        # Using Exploration - Exploitation here
        if (numpy.random.choice(['best_action', 'random_action'], p=[1 - self.epsilon, self.epsilon]) == 'random_action'):
            return {'best_action': numpy.random.choice(self.all_actions), 'best_q_value': 0}
        else:
            # iterate over all actions, get Q- value for all action
            # then choose the  action with the highest q- value
            best_action = None
            best_q_value = -99;
            # try all l, r, forward, and none actions in that state
            for possible_indv_action in self.all_actions:
                # get q value for state l,r, f and none
                tmp_q_value = self.get_q_value_for_state_action_pair(state, possible_indv_action)
                # choose the best action and best action among the l,r,forward and none
                if tmp_q_value > best_q_value:
                    best_q_value = tmp_q_value
                    best_action = possible_indv_action
            return {'best_action': best_action, 'best_q_value': best_q_value}


    #                      T(         s           ,  a   ,   r   ,     s'   )
    def update_q_table_entry(self, previous_state, action, reward, new_state):
        # state = (1,1), and action  = L, current_reward =  reward on turnin left from state
        # we create new Q dictionary that represents all states and its rewards i.e
        # default dictionary REf: https://discussions.udacity.com/t/how-to-get-best-action-from-q-table/172983
        # defaultdict(int,
        #             {(('green', None, None, None, 'left'), 'forward'): 2,
        #              (('green', None, None, None, 'left'), 'right'): 4,
        #              (('red', None, None, None, 'left'), 'forward'): -10})

        # when we take action a on state s then we reach the next state and then we want to get the maxQvalue in that next state and go on
        # REF : http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
        self.q_value_state_action_dict[(previous_state, action)] \
                        = self.get_q_value_for_state_action_pair(previous_state, action) + \
                            self.alpha * ( reward + self.discount * self.get_q_value_for_best_action_at(new_state) - self.get_q_value_for_state_action_pair(previous_state,action)   )
        print('')
        print('/***** Updating Q dictionary :: if prev state is ',previous_state,' & action is ',action,' then q value is ', self.q_value_state_action_dict[(previous_state, action)])
        pprint(self.q_value_state_action_dict)

def run():
    """Run the agent for a finite number of trials."""
    dest_count_reached = 0
    alpha = 0.7  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
    discount = 0.2  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
    epsilon = 0.0  # more epsilon value more random action choice, less epsilon value then more best action choosen but at the expense of more computation

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    agent = QLearningAgent
    agent.alpha = alpha  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
    agent.discount = discount  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
    agent.epsilon = epsilon
    a = e.create_agent(agent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    #agent_final_status.append( [alpha,  discount,epsilon, a.get_agents_final_status()['self.reached_destination_count'], a.get_agents_final_status()['self.total_reward']] )
    dest_count_reached =  a.get_agents_final_status()['self.reached_destination_count']
    print('')
    print('dest_count_reached ::',dest_count_reached,'.  at alpha ::',alpha,'.  with discount :: ',discount,' and epsilon', epsilon)



def run_for_best_configuration_extraction():
    """Run the agent for a finite number of trials."""
    best_dest_count_reached = 0
    best_epsilon = 0
    best_alpha = 0
    best_discount = 0
    agent_final_status = []

    # for alpha in [x * 0.1 + 0.1 for x in range(4,8)] :
    #     for discount in [ y * 0.2 for y in range(0,5) ]:
    #         for epsilon in [0.0]:
    for alpha in [0.7] :
        for discount in [0.2]:
             for epsilon in [x * 0.1 for x in range(0,10)]:

                #alpha = 0.6  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
                #discount = 0.5  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
                #epsilon = 0.1+epsilon  # more epsilon value more random action choice, less epsilon value then more best action choosen but at the expense of more computation

                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                agent = QLearningAgent
                agent.alpha = alpha  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
                agent.discount = discount  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
                agent.epsilon = epsilon
                a = e.create_agent(agent)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

                # Now simulate it
                sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
                sim.run(n_trials=100)  # press Esc or close pygame window to quit


                agent_final_status.append( [alpha,  discount,epsilon, a.get_agents_final_status()['self.reached_destination_count'], a.get_agents_final_status()['self.total_reward']] )

                if a.get_agents_final_status()['self.reached_destination_count'] > best_dest_count_reached:
                   best_conf = a.get_agents_final_status()
                   best_dest_count_reached =  a.get_agents_final_status()['self.reached_destination_count']
                   best_alpha = alpha
                   best_discount = discount
            #print a.reached_destination_count
    print('')
    pprint(best_conf)
    print('best_dest_count_reached ::',best_dest_count_reached,'. best_alpha ::',best_alpha,'. best_discount :: ',best_discount)
    print(agent_final_status)

if __name__ == '__main__':
    #run_for_best_configuration_extraction()
    run()

