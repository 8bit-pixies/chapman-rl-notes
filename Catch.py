import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import glob
import os
import shutil
import random as rand
import random
import functools
import matplotlib.pyplot as plt

class Catch():
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.replay_states = []
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out
        
        # for animation
        self.replay_states.append(self._draw_state())

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 2  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas
    
    def show_game(self):
        img = map_data(self._draw_state())
        plt.imshow(img)
    
    def update_and_show(self, action):
        self._update_state(action)
        self.show_game()
        
    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas[np.newaxis, :]

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.replay_states = []
        self.state = np.asarray([0, n, m])[np.newaxis]
        self.replay_states.append(self._draw_state())

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
    """shows and saves animation
    not embedded in class to use with other history
    """
    if path is None:
        path = "output"

    if animation is None:
        animation = "output_animated.gif"

    try:
        shutil.rmtree('{}/'.format(path))
        os.mkdir("{}".format(path))
    except:
        os.mkdir("{}".format(path))

    for idx, im in enumerate(list_array):
        img = map_data(im)
        img.convert('RGB').save("{}/{}.png".format(path, idx))

    # generate animation
    frames = []
    imgs = sorted(glob.glob("{}/*.png".format(path)))

    for img in imgs:
        frames.append(imageio.imread(img.replace("\\", "/")))

    duration = list(np.ones(len(frames))*0.1)
    kwargs = {'duration':duration}
    imageio.mimsave(animation, frames, 'GIF', **kwargs)

