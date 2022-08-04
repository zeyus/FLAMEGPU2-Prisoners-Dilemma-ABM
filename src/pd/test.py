MAX_PLAY_DISTANCE = 1
SEARCH_GRID_SIZE: int = 1 + 2 * MAX_PLAY_DISTANCE
SEARCH_GRID_OFFSET: int = SEARCH_GRID_SIZE // 2

SPACES_WITHIN_RADIUS_INCL: int = SEARCH_GRID_SIZE**2
SPACES_WITHIN_RADIUS: int = SPACES_WITHIN_RADIUS_INCL - 1
SPACES_WITHIN_RADIUS_ZERO_INDEXED: int = SPACES_WITHIN_RADIUS - 1
CENTER_SPACE: int = SPACES_WITHIN_RADIUS // 2


import random
from time import sleep
def pos_from_moore_seq(x, y, move_index, env_max):
  if move_index >= CENTER_SPACE:
    move_index += 1
  elif move_index > SPACES_WITHIN_RADIUS_INCL:
    move_index = move_index % SPACES_WITHIN_RADIUS_INCL
  new_x_offset = move_index // SEARCH_GRID_SIZE - SEARCH_GRID_OFFSET
  new_y_offset = move_index % SEARCH_GRID_SIZE - SEARCH_GRID_OFFSET
  new_x = (x + new_x_offset) % env_max
  new_y = (y + new_y_offset) % env_max

  return new_x, new_y, new_x_offset, new_y_offset, move_index


ENV_MAX = 16

validate = [
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
  [1, 1],
]

while True:
  x = random.randint(0, ENV_MAX-1)
  y = random.randint(0, ENV_MAX-1)
  move_index = random.randint(0, SPACES_WITHIN_RADIUS_ZERO_INDEXED)
  new_x, new_y, new_x_offset, new_y_offset, new_move_index = pos_from_moore_seq(x, y, move_index, ENV_MAX)
  print(f"x: {x}, y: {y}, move_index: {move_index}, new_x: {new_x}, new_y: {new_y}, new_x_offset: {new_x_offset}, new_y_offset: {new_y_offset}, new_move_index: {new_move_index}")
  print(validate[move_index][0] == new_x_offset, validate[move_index][1] == new_y_offset)
  sleep(0.1)