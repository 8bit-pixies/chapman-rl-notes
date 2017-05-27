from __future__ import print_function
import numpy as np
from PIL import Image
import random as rand
import imageio
import glob
import os
import shutil
import matplotlib.pyplot as plt

class Maze():
    def __init__(self, maze_location, verbose=False):
        inf = open(maze_location)
        data = np.array([list(map(float,s.strip().split(','))) for s in inf.readlines()])
        self.originalmap = data.copy()
        self.data = data.copy()
        self.verbose=verbose

        self.startpos = self.getrobotpos() #find where the robot starts
        self.goalpos = self.getgoalpos() #find where the goal is
        
        self.state = self.getrobotpos()
        # n.b. state for DQN should be theoretically
        # the raw self.data map with information stripped out        
        self.robopos = self.getrobotpos()
        
        # replay for animation
        self.replay = [self.data[:]]
    
    # print out the map
    def printmap(self):
        print("--------------------")
        for row in range(0, self.data.shape[0]):
            for col in range(0, self.data.shape[1]):
                if self.data[row,col] == 0:
                    print(" ", end=" "),
                if self.data[row,col] == 1:
                    print("O", end=" "),
                if self.data[row,col] == 2:
                    print("*", end=" "),
                if self.data[row,col] == 3:
                    print("X", end=" "),
                if self.data[row,col] == 4:
                    print(".", end=" "),
                if self.data[row,col] == 5: # previous position..
                    print("#", end=" "),
            print("")
        print("--------------------")
    
    def show_game(self):
        """show the game as an image"""
        # map to better value...
        img = map_data(self.data)
        plt.imshow(img)

    def reset(self):
        self.data = self.originalmap.copy()
        self.startpos = self.getrobotpos() #find where the robot starts
        self.goalpos = self.getgoalpos() #find where the goal is
        
        self.state = self.getrobotpos()
        # n.b. state for DQN should be theoretically
        # the raw self.data map with information stripped out        
        self.robopos = self.getrobotpos()
        self.replay = [self.data[:]]
        
    
    # find where the robot is in the map
    def getrobotpos(self):
        R = -999
        C = -999
        for row in range(0, self.originalmap.shape[0]):
            for col in range(0, self.originalmap.shape[1]):
                if self.originalmap[row,col] == 2:
                    C = col
                    R = row
        if (R+C)<0:
            print("warning: start location not defined")
        return R, C

    # find where the goal is in the map
    def getgoalpos(self):
        R = -999
        C = -999
        for row in range(0, self.originalmap.shape[0]):
            for col in range(0, self.originalmap.shape[1]):
                if self.originalmap[row,col] == 3:
                    C = col
                    R = row
        if (R+C)<0:
            print("warning: goal location not defined")
        return (R, C)

    # convert the location to a single integer
    def _state_dqn(self):
        """takes data, and strips away all information
        besides goal, current position and walls"""
        data = self.data.copy()
        data[data >= 4] = 0
        data[data == 1] = -1
        data[data == 2] = 1
        data[data == 3] = 2
        return data[np.newaxis, :]
    
    # move the robot according to the action and the map
    def movebot(self, a, type='qlearner'):
        """moves robot but does not update state..."""
        testr, testc = self.robopos

        # update the test location
        if a == 0: #north
            testr = testr - 1
        elif a == 1: #east
            testc = testc + 1
        elif a == 2: #south
            testr = testr + 1
        elif a == 3: #west
            testc = testc - 1

        # see if it is legal. if not, revert
        if testr < 0: # off the map
            testr, testc = self.state
        elif testr >= self.data.shape[0]: # off the map
            testr, testc = self.state
        elif testc < 0: # off the map
            testr, testc = self.state
        elif testc >= self.data.shape[1]: # off the map
            testr, testc = self.state
        elif self.data[testr, testc] == 1: # it is an obstacle
            testr, testc = self.state
        
        # set state for qlearner
        self.state = (testr, testc)
        
        # get reward...
        if (testr, testc) == self.goalpos:
            r = 1
            
        else:
            r = -1
        
        # update state
        self.data[self.data == 5] = 4
        self.data[self.robopos] = 4
        self.data[(testr, testc)] = 2
        self.robopos = (testr, testc)
        
        self.replay.append(self.data.copy())
        return self.get_state(type), r, (testr, testc) == self.goalpos
    
    def get_state(self, type='qlearner'):
        if type=='qlearner':
            return self.state[0]*10 + self.state[1]
        else:
            return self._state_dqn()

def map_image(data, colour_mapping={0:(0,0,0), 
                                    1:(128, 128, 128), 
                                    2:(64, 64, 255), 
                                    3:(255, 0, 0), 
                                    4:(128, 128, 255), 
                                    5:(0, 0, 255)}):
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    
    for idx, mapping in list(colour_mapping.items()):
        mask = (red == idx) & (green == idx) & (blue == idx)
        data[:, :, :3][mask] = list(mapping)
    return data

def map_data(data):
    img = Image.fromarray(data).convert('RGB')
    img_col = Image.fromarray(map_image(np.array(img))).resize((200,200)).convert('RGB')
    return img_col

def animate(list_array, path=None, animation=None):
    """shows and saves animation, it will be down sampled for large
    number of frames..."""
    if path is None:
        path = "output_maze"

    if animation is None:
        animation = "output_maze_animated.gif"

    try:
        shutil.rmtree('{}/'.format(path))
        os.mkdir("{}".format(path))
    except:
        os.mkdir("{}".format(path))

    for idx, im in enumerate(list_array):
        img = map_data(im)
        img.save("{}/{}.png".format(path, idx))

    # generate animation
    frames = []
    imgs = sorted(glob.glob("{}/*.png".format(path)))
    
    total_frames = len(list_array)
    if total_frames < 100:
        downsample = 1
    else:
        downsample = int(total_frames/100.0)

    for idx, img in enumerate(imgs):
        if idx % downsample == 0 or idx == total_frames -1:
            frames.append(imageio.imread(img.replace("\\", "/")))
    
    # make the last one twice as long
    frames.append(imageio.imread(img.replace("\\", "/")))
    duration = list(np.ones(len(frames))*0.1)
    #print(duration)
    kwargs = {'duration':duration}
    imageio.mimsave(animation, frames, 'GIF', **kwargs)