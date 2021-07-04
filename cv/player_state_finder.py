from enum import Enum


class PlayerState(Enum):
    QUEUE = 1
    LEGEND_SELECT = 2
    DROP_SHIP = 3
    ALIVE = 4
    KNOCKED = 5
    DEAD = 6
    RESPAWNING = 7

class PlayerState:
    current_state = PlayerState(1)

    def findPlayerState(frame):
        placeholder = 0
