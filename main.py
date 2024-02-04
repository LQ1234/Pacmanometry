import robomodules as rm
from messages import MsgType, message_buffers
import os
from modeling import realign_light_state, simulate
from visualization import initialize_plot, draw_game_state, update_plot

ADDRESS = os.environ.get("LOCAL_ADDRESS", "localhost")
PORT = os.environ.get("LOCAL_PORT", 11297)

class Pacmanometry(rm.ProtoModule):

    def __init__(self, addr, port):
        self.FREQUENCY = 5

        self.subscriptions = [MsgType.LIGHT_STATE]
        super().__init__(addr, port, message_buffers, MsgType, self.FREQUENCY, self.subscriptions)

        self.game_state = None
        initialize_plot()

    def tick(self):
        print(self.game_state["scatter"])
        if self.game_state is not None:
            simulation = simulate(self.game_state, 20)
            draw_game_state(self.game_state, simulation)

        update_plot()

    def msg_received(self, msg, msg_type):
        self.game_state = realign_light_state(msg, self.game_state)
        

def main():
    print("Running Pacmanometry module")
    module = Pacmanometry(ADDRESS, PORT)
    module.run()

if __name__ == "__main__":
    main()