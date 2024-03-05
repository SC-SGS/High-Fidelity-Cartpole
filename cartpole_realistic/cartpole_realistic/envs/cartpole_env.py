"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Extended for more realism by Linus Bantel
"""
import math
from math import pi, cos, sin, trunc, e
from typing import Optional, Union
import gin

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces, logger
from gym.utils import seeding

@gin.configurable
class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 force_steps = [0,100,150,200,250],
                 masscart = 0.94,
                 masspole = 0.1,
                 length = 0.1778,
                 force_mag = 3.0,
                 internal_tau = 0.02,
                 tau = 0.02,
                 x_threshold = 0.3,
                 pole_resistance = 0.99,
                 cart_resistance = 0.99,
                 force_curve = [0,1,0],
                 evaluation = False,
                 swingup = False,
                 discrete_state = False,
                 action_in_state = True):

        #set "evaluation" in the code to obtain a second environment for evaluation, 
        #for conistent resets

        #physical parameters
        self.force_steps = force_steps
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag_mean = force_mag
        self.force_mag = None
        self.internal_tau = internal_tau # seconds for euler step
        self.tau = tau  # seconds between state updates, multiple of internal tau
        self.mean_pole_resistance = e**(pole_resistance*internal_tau)
        self.mean_cart_resistance = cart_resistance
        self.cart_resistance = None
        self.pole_resistance = None
        self.force_curve = force_curve

        #different experimental setups
        self.evaluation = evaluation
        self.swingup = swingup
        self.action_in_state = action_in_state
        self.discrete_state = discrete_state

        self.states = []


        # Angle at which to fail the classical task
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = x_threshold

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                pi,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )
        
        self.action_space = spaces.Discrete(len(self.force_steps) * 2 - 1)


        #action within state
        if self.action_in_state:
            action = np.ones(len(force_steps)*2+1)
            self.observation_space = spaces.Box(np.concatenate((-high, action*0)), np.concatenate((high, action)), dtype=np.float32)
            print('action_in_state = True')
        else:
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.num_steps = None

        self.steps_beyond_done = None

        self.last_action = None
        self.last_reward = 0




    
    def update_state(self, state, force, tau):
        #function implements the actual differential equation and integration

        x, x_dot, theta, theta_dot = state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf

        #sign function can lead to weird results depending on the time-step-width
        #possible replacement is a scaled and parameterized sigmoid

        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
            - self.cart_resistance*sign(x_dot)
        ) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass



        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot*self.pole_resistance + tau * thetaacc
        theta = theta + tau * theta_dot


        theta = angle_normalize(theta)


        return (x, x_dot, theta, theta_dot)



    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."


        action = action - len(self.force_steps) + 1



        if action > 0:
            dir = 1
        elif action < 0:
            dir = -1
        else:
            dir = 0


        #motor values to use (0-255)
        force = self.force_steps[abs(action)]


        #force curve of the cartpole
        force = self.force_curve[0]*force**2 - self.force_curve[1]*force +  self.force_curve[2]
        force *= dir
        

        x, x_dot, theta, theta_dot = self.state

        if self.discrete_state: 
            old_disc_theta = discretize_angle(theta)


        #internal finer stepping of the simulation
        for i in range(int(self.tau/self.internal_tau)):
            self.state = self.update_state(self.state, force, self.internal_tau)



        x, x_dot, theta, theta_dot = self.state

        #calculating the derivatives of the discrete state for the output
        if self.discrete_state:
            new_disc_theta = discretize_angle(theta)

            if new_disc_theta > pi/2 and old_disc_theta < -pi/2:
                old_disc_theta += 2*pi 
            elif new_disc_theta < -pi/2 and old_disc_theta > pi/2:
                old_disc_theta -= pi

            disc_theta_dot = (new_disc_theta - old_disc_theta)/self.tau
            out_state = (x, x_dot, new_disc_theta, disc_theta_dot)
        else:
            out_state = self.state


        #variable to check, if pole is upright
        if abs(theta) < self.theta_threshold_radians:
            self.swingup_done = True
        elif abs(theta) > self.theta_threshold_radians:
            self.swingup_done = False


        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or abs(theta_dot) > 5*pi
            or (abs(theta) > self.theta_threshold_radians and not self.swingup)
        )

        if not done:
            if self.evaluation:
                reward = cos(theta / 2) * (1-abs(x)/self.x_threshold)
            else:
                reward = cos(theta / 2) * (1-abs(x)/self.x_threshold)
            #additional addon for training swingup faster 
            
            if self.swingup and abs(theta) < 5/180*pi:
                reward += 5
                if abs(theta_dot) < 6/180*pi:
                    reward += 5
            

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0


        self.num_steps += 1
        self.last_action = action
        self.last_reward = reward

        #append action to state if neccessary
        if self.action_in_state:
            action_arr = np.zeros(len(self.force_steps)*2+1)
            action_arr[action+len(self.force_steps)] = 1
            return np.concatenate((np.array(out_state, dtype=np.float32), action_arr)), reward, done, {}
        else:
            return np.array(out_state, dtype=np.float32), reward, done, {}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        #reset with slight perturbation
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        #vertical 
        self.state[2] = self.np_random.uniform(low = -2/180*pi, high = 2/180*pi)
            
        if self.swingup:

            #for evaluation purposes start always hanging
            #for training, start with an arbitary pole angle
            if self.evaluation:
                self.state = self.np_random.uniform(low=-0.01, high=0.01, size=(4,))
                self.state[2] = angle_normalize(pi - self.np_random.uniform(low=-0.01, high=0.01))
            else:
                self.state[2] = self.np_random.uniform(low=-pi, high=pi)


        #perturbate resistance parameters
        self.cart_resistance = self.mean_cart_resistance#np.random.uniform(low = self.mean_cart_resistance*0.95, high = self.mean_cart_resistance*1.05)
        self.pole_resistance = self.mean_pole_resistance#np.random.uniform(low = self.mean_pole_resistance*0.97, high = min(self.mean_pole_resistance*1.03,0.9999))



        #used to train for multiple poles on one agent
        """"
        pend = self.np_random.randint(2)d

        if pend == 0:

            self.masspole = 0.127
            self.total_mass = self.masspole + self.masscart
            self.length = 0.1778  # actually half the pole's length
            self.polemass_length = self.masspole * self.length

        else:

            self.masspole = 0.23
            self.total_mass = self.masspole + self.masscart
            self.length = 0.3302  # actually half the pole's length
            self.polemass_length = self.masspole * self.length
        """


        self.num_steps = 0
        self.steps_beyond_done = None
        self.last_action = 0
        self.swingup_done = False

        if self.action_in_state:
            action = np.zeros(len(self.force_steps)*2+1)
            out = np.concatenate((np.array(self.state, dtype=np.float32), action))
        else:
            out = np.array(self.state, dtype=np.float32)

        if not return_info:
            return out
        else:
            return out, {}




    def render(self, mode="human"):
        screen_width = 608
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * self.length
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        fontObj = pygame.font.Font(None, 32)
        cart_res = fontObj.render(str(self.cart_resistance), True, (0,0,0), None)

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = int(screen_height/2 + cartheight/2)  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        #draw force steps in render 
        for i in range(-len(self.force_steps), len(self.force_steps)+1):
            color = (0,0,0)
            if i == self.last_action:
                color = (255,0,0)
            gfxdraw.filled_circle(
                    self.surf,
                    int(screen_width/2+i*20),
                    int(50),
                    5,
                    color,
            )

        #reward value
        gfxdraw.rectangle(self.surf, (20,20,20,80),(0,0,0))

        gfxdraw.box(self.surf, (20,20,20,self.last_reward*80),(0,0,0))


        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi



#not used, possible distribution for perturbating physical parameters
def trunc_normal(mean, sigma, lower = 0, upper = 1):
    i = 0
    x = mean

    while i < 50:
        x =  np.random.normal(mean, sigma)
        if x >= lower and x <= upper:
            break
        i += 1

    if i >= 50:
        x = mean

    return x


def discretize_angle(theta):
    disc = int((theta+pi)/pi*2048)-2048 #range from -2048 to 2048 
    return disc/2048*pi #range from -pi to pi


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def sigmoid(x,tau):
    return 2/(1 + np.exp(-x*tau))-1



   

gin.parse_config_file('env_config.gin')
