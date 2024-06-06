import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
import random

style.use("ggplot")

SIZE = 5 # grid size of the PD-World
EPISODES = 3000 # 3000 because each agent take one step so that 3000 * 3 = 9000 steps
MOVE_PENALTY = 1 # -1 for each move
PICKUP_REWARD = 13 
DROPOFF_REWARD = 13
LEARNING_RATE = 0.3 # alpha
DISCOUNT = 0.5   # gamma
SEED = 8 # change this before each experiment
# agents and their key value in the dictionary
RED_AGENT = 1  # (3,3)
BLUE_AGENT = 2  # (3,5)
BLACK_AGENT = 3  # (1,3)
PICKUP_CELL = 4 # pickup cells: (1,5), (2,4), (5,2)
DROPOFF_CELL = 5 # dropoff cells: (1,1), (3,1), (4,5)

start_q_table = None # can later replace with existing q_table (file path)

# 1,2, and 3 are agent color
PD_world_dic = {1: (0, 0, 255), # red agent
     2: (255, 0, 0), # blue agent
     3: (0, 0, 0),  # black agent
     4: (0, 0, 128), # pickup cell (green)
     5: (0, 255, 0)} # dropoff cell (magenta)


class Agent:
    # initialize the agent position, pickup cell, and dropoff cell
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.blocks = 0 # initalize blocks held by agent

    # print out the position of the agents
    def __str__(self):
        return f"{self.x}, {self.y}, {self.blocks}"
    
    # agent's move method
    def move(self, x, y, other_agents):
        new_x = self.x + x
        new_y = self.y + y

        # Check if the new position is within bounds
        if 0 <= new_x < SIZE and 0 <= new_y < SIZE:
            # Check if the new position is not occupied by another agent
            if all(new_x != agent.x or new_y != agent.y for agent in other_agents):
                self.x = new_x
                self.y = new_y

    # check if agent can move or not
    def can_move(self, dx, dy, pickup_cells, dropoff_cells, other_agents):
        new_x = self.x + dx
        new_y = self.y + dy

        # Check if the new position is within bounds
        if not (0 <= new_x < SIZE and 0 <= new_y < SIZE):
            return False

        # Check if the new position is not occupied by another agent
        if any(new_x == agent.x and new_y == agent.y for agent in other_agents):
            return False

        # Check if the new position is a pickup cell or dropoff cell
        for pickup_cell in pickup_cells:
            if (new_x, new_y) == (pickup_cell.x, pickup_cell.y):
                if self.blocks < 5:
                    return True
                else:
                    return any((new_x, new_y) != (cell.x, cell.y) and self.blocks < 5 for cell in pickup_cells)
        for dropoff_cell in dropoff_cells:
            if (new_x, new_y) == (dropoff_cell.x, dropoff_cell.y):
                if self.blocks > 0:
                    return True
                else:
                    return any((new_x, new_y) != (cell.x, cell.y) and self.blocks > 0 for cell in dropoff_cells)

        return True

    # the method handle agent movement and award the agent depend on what move they make
    def handle_agent_movement(self, dx, dy, pickup_cells, dropoff_cells, other_agents):
        reward = 0
        if self.can_move(dx, dy, pickup_cells, dropoff_cells, other_agents):
            self.move(dx, dy, other_agents)
            # reward the agent if they pickup or dropoff the block
            for pickup_cell, dropoff_cell in zip(pickup_cells, dropoff_cells):
                if (self.x, self.y) == (pickup_cell.x, pickup_cell.y):
                    if pickup_cell.blocks > 0 and self.blocks == 0:
                        self.blocks += 1
                        pickup_cell.blocks -= 1
                        reward += PICKUP_REWARD
                elif (self.x, self.y) == (dropoff_cell.x, dropoff_cell.y):
                    if dropoff_cell.blocks < 5 and self.blocks == 1:
                        self.blocks -= 1
                        dropoff_cell.blocks += 1
                        reward += DROPOFF_REWARD
            # else they will penalize for moving
            reward -= MOVE_PENALTY
        return reward

    def action(self, choice):
        if choice == 'n':  # Move up (North)
            return -1, 0
        elif choice == 's':  # Move down (South)
            return 1, 0
        elif choice == 'w':  # Move left (West)
            return 0, -1
        elif choice == 'e':  # Move right (East)
            return 0, 1
        
    # implement the 3 policies
    def choose_action(self, paction, q_table):
        random.seed(SEED)
        if paction == "PRANDOM":
            return self.action(np.random.choice(['n', 's', 'w', 'e']))
        elif paction == "PEXPLOIT":
            if np.random.uniform(0, 1) < 0.8:
                action_type = max(q_table[(self.x, self.y)], key=q_table[(self.x, self.y)].get)
                return self.action(action_type)
            else:
                valid_actions = ['n', 's', 'w', 'e']
                q_values = q_table[(self.x, self.y)]
                max_q = np.max(list(q_values.values()))
                non_max = [action for action in valid_actions if q_values[action] != max_q]
                if non_max:
                    action_type = np.random.choice(non_max)
                    return self.action(action_type)
                else:
                    return self.action(np.random.choice(valid_actions))
        elif paction == "PGREEDY":
            valid_actions = ['n', 's', 'w', 'e']
            q_values = q_table[(self.x, self.y)]
            if q_values:
                return self.action(max(q_values, key=q_values.get))
            else:
                return self.action(np.random.choice(valid_actions))


class Cells:
    def __init__(self, x, y, initial_blocks=0, capacity=5):
        self.x = x
        self.y = y
        self.blocks = initial_blocks
        self.capacity = capacity

    def __str__(self):
        return f"{self.x}, {self.y}, {self.blocks}"


# if we don't have a q table then make one else use the existing one
if start_q_table is None:
    q_table = {}
    for i in range(SIZE):
        for j in range(SIZE):
            for action_type in ['n', 's', 'w', 'e']:
                q_table[(i, j)] = {action_type: 0 for action_type in ['n', 's', 'w', 'e']}
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# this funnction is use to update the q table for Q learning
def update_q_table(q_table, current_state, action, reward, max_future_q):
    if (action[0], action[1]) == (-1, 0):
        action_type = 'n'
    if (action[0], action[1]) == (1, 0):
        action_type = 's'
    if (action[0], action[1]) == (0, -1):
        action_type = 'w'
    if (action[0], action[1]) == (0, 1):
        action_type = 'e'

    current_q = q_table[current_state][action_type]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[current_state][action_type] = new_q

# this function is use to update the q table for SARSA
def update_sarsa_table(q_table, current_state, action, reward, max_future_q):
    if (action[0], action[1]) == (-1, 0):
        action_type = 'n'
    if (action[0], action[1]) == (1, 0):
        action_type = 's'
    if (action[0], action[1]) == (0, -1):
        action_type = 'w'
    if (action[0], action[1]) == (0, 1):
        action_type = 'e'

    current_q = q_table[current_state][action_type]
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
    q_table[current_state][action_type] = new_q

# this function is use to visualize the agent movement in the pd world
def display_pd_world(red, blue, black, pickup1, pickup2, pickup3, dropoff1, dropoff2, dropoff3, size):
    pd_world = np.zeros((size, size, 3), dtype=np.uint8)
    pd_world.fill(255)  # Fill with white background

    pd_world[red.x][red.y] = PD_world_dic[1] # PD_world_disc store the color that associate with each agent
    pd_world[blue.x][blue.y] = PD_world_dic[2]
    pd_world[black.x][black.y] = PD_world_dic[3]
    pd_world[pickup1.x][pickup1.y] = PD_world_dic[4]
    pd_world[pickup2.x][pickup2.y] = PD_world_dic[4]
    pd_world[pickup3.x][pickup3.y] = PD_world_dic[4]
    pd_world[dropoff1.x][dropoff1.y] = PD_world_dic[5]
    pd_world[dropoff2.x][dropoff2.y] = PD_world_dic[5]
    pd_world[dropoff3.x][dropoff3.y] = PD_world_dic[5]

    # Resize for better display
    pd_world_resized = cv2.resize(pd_world, (400, 400), interpolation=cv2.INTER_NEAREST)

    # Display the image
    cv2.imshow("PD World", pd_world_resized)
    # change the number depend on how fast you want the pd world to refresh
    cv2.waitKey(10)
    cv2.destroyAllWindows()

# main function to run the test
def pd_world(q_table, agents, pickup_cells, dropoff_cells):
    steps_taken = 0
    total_reward_arr = []
    total_reward = 0
    terminate = False
    print("Initial 500 steps")
    for _ in range(EPISODES):
        if terminate:
            break
        # this is use to check if current agent next move is not in the position of the other agents
        other_agents = [agent for agent in agents if agent != _]
        for agent in agents:
            # run PRANDOM for the first 500 steps
            if steps_taken != 500:
                # calculate the remaining blocks that the agents hold and check if pickup is empty and dropoff is full
                blocks_remaining = sum(agent.blocks for agent in agents)
                pickup_empty = all(cell.blocks == 0 for cell in pickup_cells)
                dropoff_full = all(cell.blocks == 5 for cell in dropoff_cells)
                # set terminate flag to true if the agents complete in the first 500 steps
                if blocks_remaining == 0 and pickup_empty and dropoff_full:
                    print("All pickup cells empty and dropoff cells full")
                    terminate = True
                    break
                action = agent.choose_action("PRANDOM", q_table)
                # x and y coordinate for the action (n,s,w,e)
                dx, dy = action[0], action[1]
                # agent current position
                curr_state = (agent.x, agent.y)
                # handle_agent_movement will also return the reward after the agent moved
                reward = agent.handle_agent_movement(dx, dy, pickup_cells, dropoff_cells, other_agents)
                display_pd_world(red, blue, black, pickup1, pickup2, pickup3, dropoff1, dropoff2, dropoff3, SIZE)
                total_reward += reward
                # get agent position after moved
                next_state = (agent.x, agent.y)
                # if next_state is not in q_table then give it the action (n,s,w,e) with values of 0s
                if next_state not in q_table:
                    q_table[next_state] = {action_type: 0 for action_type in ['n', 's', 'w', 'e']}
                # calculate the next state max q value
                max_future_q = np.max(list(q_table[next_state].values()))
                # change update_q_table to update_sarsa_table depend on the test
                update_q_table(q_table, curr_state, action, reward, max_future_q)
                # increment steps taken
                steps_taken += 1
            # run the remaining 8500 steps
            else:
                print("\nContinued for 8500 steps")
                blocks_remaining = sum(agent.blocks for agent in agents)
                pickup_empty = all(cell.blocks == 0 for cell in pickup_cells)
                dropoff_full = all(cell.blocks == 5 for cell in dropoff_cells)
                
                if blocks_remaining == 0 and pickup_empty and dropoff_full:
                    print("All pickup cells empty and dropoff cells full")
                    terminate = True
                    break
                action = agent.choose_action("PEXPLOIT", q_table)
                dx, dy = action[0], action[1]
                curr_state = (agent.x, agent.y)
                reward = agent.handle_agent_movement(dx, dy, pickup_cells, dropoff_cells, other_agents)
                display_pd_world(red, blue, black, pickup1, pickup2, pickup3, dropoff1, dropoff2, dropoff3, SIZE)
                total_reward += reward
                next_state = (agent.x, agent.y)
                if next_state not in q_table:
                    q_table[next_state] = {action_type: 0 for action_type in ['n', 's', 'w', 'e']}
                max_future_q = np.max(list(q_table[next_state].values()))
                update_q_table(q_table, curr_state, action, reward, max_future_q)
                steps_taken += 1
        total_reward_arr.append(total_reward)

    # dump the q_table for later usage
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)

    # print the reward vs episodes graph
    print("Total Reward: ", total_reward)
    print("Total steps: ", steps_taken)
    plt.plot(range(1, len(total_reward_arr) + 1), total_reward_arr)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs. Episodes')
    plt.show()

# initialize the agents, pickup cells, and dropoff cells positions
red = Agent(2, 2)
blue = Agent(4, 2)
black = Agent(0, 2)
pickup1 = Cells(0, 4, initial_blocks=5)
pickup2 = Cells(1, 3, initial_blocks=5)
pickup3 = Cells(4, 1, initial_blocks=5)
dropoff1 = Cells(0, 0)
dropoff2 = Cells(2, 0)
dropoff3 = Cells(3, 4)

# create list for each components
agents = [red, blue, black]
pickup_cells = [pickup1, pickup2, pickup3]
dropoff_cells = [dropoff1, dropoff2, dropoff3]

# call pd_world for the training
pd_world(q_table, agents, pickup_cells, dropoff_cells)

# Print out the results of all the Q-tables
# for state, actions in q_table.items():
#     print(f"State: {state}")
#     for action, value in actions.items():
#         print(f"Action: {action}, Value: {value}")