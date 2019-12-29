class LunarLanderEnvironment(environment.BaseEnvironment):
    def __init__(self):
        self.current_state = None
        self.count = 0

    def env_init(self, env_info):
        # users set this up
        self.state = np.zeros(6)  # velocity x, y, angle, distance to ground, landing zone x, y

    def env_start(self):
        land_x, land_y = get_landing_zone()  # gets the x, y coordinate of the landing zone
        # At the start we initialize the agent to the top left hand corner (100, 20) with 0 velocity
        # in either any direction. The agent's angle is set to 0 and the landing zone is retrieved and set.
        # The lander starts with fuel of 100.
        # (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)
        self.current_state = (0, 0, 0, 100, 20, land_x, land_y, 100)
        return self.current_state

    def env_step(self, action):
        land_x, land_y = get_landing_zone()  # gets the x, y coordinate of the landing zone
        vel_x, vel_y = get_velocity(action)  # gets the x, y velocity of the lander
        angle = get_angle(action)  # gets the angle the lander is positioned in
        pos_x, pos_y = get_position(action)  # gets the x, y position of the lander
        fuel = get_fuel(action)  # get the amount of fuel remaining for the lander

        terminal = False
        reward = 0.0
        observation = (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)

        # use the above observations to decide what the reward will be, and if the
        # agent is in a terminal state.
        # Recall - if the agent crashes or lands terminal needs to be set to True

        # YOUR CODE HERE
        landed= pos_y == land_y or pos_y > land_y
        if fuel<=0 and not landed:
            terminal=True
            reward=-1000
        elif landed and  (vel_y<-3 or not (-10<=vel_x<=10) or 5<angle<355 or fuel<=0 or pos_x != land_x):
            terminal=True
            reward=-1000

        elif not landed:
            reward=-1
        else:
            terminal=True
            reward=1000

        raise NotImplementedError()

        self.reward_obs_term = (reward, observation, terminal)
        return self.reward_obs_term

    def env_cleanup(self):
        return None

    def env_message(self):
        return None
