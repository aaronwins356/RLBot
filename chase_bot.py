from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

class ChaseBot(BaseAgent):
    def initialize_agent(self):
        self.controller_state = SimpleControllerState()

    def get_output(self, packet):
        ball = packet.game_ball.physics.location
        car = packet.game_cars[self.index].physics.location

        # Always drive forward
        self.controller_state.throttle = 1.0

        # Turn towards ball
        self.controller_state.steer = 1.0 if ball.x > car.x else -1.0

        return self.controller_state
