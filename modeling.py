import numpy as np
import numba as nb
import gameengine_reference.grid as grid_reference
from gameengine_reference.variables import I, n
from gameengine_reference import ghostpaths
from messages.lightState_pb2 import LightState
# ------ Information from Reference ------

grid = np.array(grid_reference.grid, dtype=np.int8)

pink_scatter_pos = np.array(ghostpaths.pink_scatter_pos, dtype=np.int64)
orange_scatter_pos = np.array(ghostpaths.orange_scatter_pos, dtype=np.int64)
blue_scatter_pos = np.array(ghostpaths.blue_scatter_pos, dtype=np.int64)
red_scatter_pos = np.array(ghostpaths.red_scatter_pos, dtype=np.int64)

ghost_no_up_tiles = np.array(ghostpaths.ghost_no_up_tiles, dtype=np.int64)

# ------Datatypes------

# movement order: [right, up, left, down, stay]
right_i = 0
up_i = 1
left_i = 2
down_i = 3
stay_i = 4

dir_offsets = np.array([
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (0, 0)
], dtype=np.int64)

ghost_dir_offsets = np.array([
    (1, 0),
    (-1, 1), # replicate "up" issue
    (-1, 0),
    (0, -1),
], dtype=np.int64)

ghost_state_npdt = np.dtype([
    ('next', np.int64, (2,)),
    ('current', np.int64, (2,)),
    ('scatter_pos', np.int64, (2,)),
    ('deterministic', np.bool_),
])

ghost_state_nbdt = nb.from_dtype(ghost_state_npdt)

game_state_npdt = np.dtype([
    ('pacbot_pos', np.int64, (2,)),
    ('pacbot_dir', np.int64),
    ('pink', ghost_state_npdt),
    ('orange', ghost_state_npdt),
    ('blue', ghost_state_npdt),
    ('red', ghost_state_npdt),
    ('scatter', bool),
])

game_state_nbdt = nb.from_dtype(game_state_npdt)

simulation_result_npdt = np.dtype([
    ('pink', np.int64, (2,)),
    ('orange', np.int64, (2,)),
    ('blue', np.int64, (2,)),
    ('red', np.int64, (2,)),
    ('scatter', bool),
])

# ------Functions------

@nb.njit(nb.bool_(ghost_state_nbdt, nb.int64[:]))
def is_move_legal(ghost, move):
    new_pos = ghost.next + move
    if np.all(new_pos == ghost.current):
        return False
    
    if grid[new_pos[0], new_pos[1]] == I:
        return False
    
    if grid[new_pos[0], new_pos[1]] == n:
        return False
    
    return True

@nb.njit(nb.bool_[:](ghost_state_nbdt), cache=True)
def find_possible_moves(ghost):
    potential_movements = np.zeros(5, dtype = np.bool_)

    for i in range(4):
        potential_movements[i] = is_move_legal(ghost, dir_offsets[i].copy())

    for i in range(len(ghost_no_up_tiles)):
        if np.all(ghost.current == ghost_no_up_tiles[i]):
            potential_movements[up_i] = False

    if not np.any(potential_movements):
        potential_movements[4] = True

    return potential_movements

@nb.njit(nb.int64(ghost_state_nbdt, nb.int64[:], nb.bool_), cache=True)
def get_move_based_on_target(ghost, target, update_deterministic):
    potential_movements = find_possible_moves(ghost)
    if np.sum(potential_movements) > 1:
        if update_deterministic:
            ghost.deterministic = False

    positions = dir_offsets + ghost.next
    distances = np.empty(5, dtype=np.float64)

    for i in range(5):
        if potential_movements[i]:
            distances[i] = np.linalg.norm((positions[i] - target).astype(np.float64))
        else:
            distances[i] = np.inf

    return np.argmin(distances)

@nb.njit(nb.int64(game_state_nbdt), cache=True)
def get_next_blue_chase_move(game_state):
    ghost = game_state.blue
    target = game_state.pacbot_pos + 2 * ghost_dir_offsets[game_state.pacbot_dir]
    target += target - game_state.red.current
    return get_move_based_on_target(ghost, target, True)

@nb.njit(nb.int64(game_state_nbdt), cache=True)
def get_next_pink_chase_move(game_state):
    ghost = game_state.pink
    target = game_state.pacbot_pos + 4 * ghost_dir_offsets[game_state.pacbot_dir]
    return get_move_based_on_target(ghost, target, True)

@nb.njit(nb.int64(game_state_nbdt), cache=True)
def get_next_red_chase_move(game_state):
    ghost = game_state.red
    return get_move_based_on_target(ghost, game_state.pacbot_pos, True)

@nb.njit(nb.int64(game_state_nbdt), cache=True)
def get_next_orange_chase_move(game_state):
    ghost = game_state.orange
    distance = np.linalg.norm((game_state.pacbot_pos - ghost.current).astype(np.float64))
    if distance < 8:
        target = ghost.scatter_pos
    else:
        target = game_state.pacbot_pos
    return get_move_based_on_target(ghost, target, True)

@nb.njit(nb.int64(ghost_state_nbdt), cache=True)
def get_next_scatter_move(ghost):
    return get_move_based_on_target(ghost, ghost.scatter_pos, False)

@nb.njit(nb.void(ghost_state_nbdt, nb.int64), cache=True)
def update_ghost(ghost, direction):
    ghost.current = ghost.next
    ghost.next += dir_offsets[direction]

    


@nb.njit(nb.void(game_state_nbdt), cache=True)
def step(game_state):

    pink_move, orange_move, blue_move, red_move = 0, 0, 0, 0
    if game_state.scatter:
        pink_move = get_next_scatter_move(game_state.pink)
        orange_move = get_next_scatter_move(game_state.orange)
        blue_move = get_next_scatter_move(game_state.blue)
        red_move = get_next_scatter_move(game_state.red)

    else:
        pink_move = get_next_pink_chase_move(game_state)
        orange_move = get_next_orange_chase_move(game_state)
        blue_move = get_next_blue_chase_move(game_state)
        red_move = get_next_red_chase_move(game_state)

    update_ghost(game_state.pink, pink_move)
    update_ghost(game_state.orange, orange_move)
    update_ghost(game_state.blue, blue_move)
    update_ghost(game_state.red, red_move)


@nb.njit(game_state_nbdt[:](game_state_nbdt, nb.int64), cache=True)
def simulate(game_state, n_steps):
    game_states = np.zeros(n_steps, dtype=game_state_nbdt)
    game_states[0] = game_state
    for i in range(1, n_steps):
        game_states[i] = game_states[i - 1]
        step(game_states[i])
    return game_states
# ------Main------

def determine_direction(now, next):
    if next[0] > now[0]:
        return right_i
    elif next[0] < now[0]:
        return left_i
    elif next[1] > now[1]:
        return up_i
    elif next[1] < now[1]:
        return down_i
    else:
        return stay_i

def realign_light_state(msg, game_state = None):
    pacbot_pos = (msg.pacman.x, msg.pacman.y)
    if game_state is None:
        game_state = np.zeros(1, dtype=game_state_nbdt)[0]
        game_state["pacbot_pos"] = pacbot_pos
        game_state["pacbot_dir"] = left_i

        game_state["pink"] = (np.array([msg.pink_ghost.x, msg.pink_ghost.y], dtype=np.int64), np.array([msg.pink_ghost.x, msg.pink_ghost.y], dtype=np.int64), pink_scatter_pos, True)
        game_state["orange"] = (np.array([msg.orange_ghost.x, msg.orange_ghost.y], dtype=np.int64), np.array([msg.orange_ghost.x, msg.orange_ghost.y], dtype=np.int64), orange_scatter_pos, True)
        game_state["blue"] = (np.array([msg.blue_ghost.x, msg.blue_ghost.y], dtype=np.int64), np.array([msg.blue_ghost.x, msg.blue_ghost.y], dtype=np.int64), blue_scatter_pos, True)
        game_state["red"] = (np.array([msg.red_ghost.x, msg.red_ghost.y], dtype=np.int64), np.array([msg.red_ghost.x, msg.red_ghost.y], dtype=np.int64), red_scatter_pos, True)
        game_state["scatter"] = True

        return game_state
    
    if tuple(game_state["pacbot_pos"]) != pacbot_pos:
        game_state["pacbot_dir"] = determine_direction(game_state["pacbot_pos"], pacbot_pos)
        game_state["pacbot_pos"] = pacbot_pos

    did_reverse = False

    pink_pos = (msg.pink_ghost.x, msg.pink_ghost.y)
    if tuple(game_state["pink"]["next"]) != pink_pos:
        if tuple(game_state["pink"]["current"]) == pink_pos and grid[pink_pos[0], pink_pos[1]] != n and not msg.pink_ghost.state == LightState.FRIGHTENED:
            
            did_reverse = True

        dir = determine_direction(game_state["pink"]["next"], pink_pos)
        game_state["pink"]["next"] = pink_pos
        game_state["pink"]["current"] = game_state["pink"]["next"] - dir_offsets[dir]
        game_state["pink"]["deterministic"] = True

    orange_pos = (msg.orange_ghost.x, msg.orange_ghost.y)
    if tuple(game_state["orange"]["next"]) != orange_pos:
        if tuple(game_state["orange"]["current"]) == orange_pos and grid[orange_pos[0], orange_pos[1]] != n and not msg.orange_ghost.state == LightState.FRIGHTENED:
            did_reverse = True

        dir = determine_direction(game_state["orange"]["next"], orange_pos)
        game_state["orange"]["next"] = orange_pos
        game_state["orange"]["current"] = game_state["orange"]["next"] - dir_offsets[dir]
        game_state["orange"]["deterministic"] = True

    blue_pos = (msg.blue_ghost.x, msg.blue_ghost.y)
    if tuple(game_state["blue"]["next"]) != blue_pos:
        if tuple(game_state["blue"]["current"]) == blue_pos and grid[blue_pos[0], blue_pos[1]] != n and not msg.blue_ghost.state == LightState.FRIGHTENED:
            did_reverse = True

        dir = determine_direction(game_state["blue"]["next"], blue_pos)
        game_state["blue"]["next"] = blue_pos
        game_state["blue"]["current"] = game_state["blue"]["next"] - dir_offsets[dir]
        game_state["blue"]["deterministic"] = True

    red_pos = (msg.red_ghost.x, msg.red_ghost.y)
    if tuple(game_state["red"]["next"]) != red_pos:
        if tuple(game_state["red"]["current"]) == red_pos and grid[red_pos[0], red_pos[1]] != n and not msg.red_ghost.state == LightState.FRIGHTENED:
            did_reverse = True

        dir = determine_direction(game_state["red"]["next"], red_pos)
        game_state["red"]["next"] = red_pos
        game_state["red"]["current"] = game_state["red"]["next"] - dir_offsets[dir]
        game_state["red"]["deterministic"] = True
   
    if did_reverse:
        game_state["scatter"] = not game_state["scatter"]

    return game_state