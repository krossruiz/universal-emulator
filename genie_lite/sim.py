"""
Tiny hand-written stand-in for the real game's logic.

We don't have a real GBA emulator here, so we can't read actual RAM state
out of "Legend of Zelda - A Link to the Past & Four Swords.gba" while it
runs. This is a small deterministic ALTTP-flavored grid world (rooms,
position, health, one item) used purely to generate (state, action,
next_state) training examples for the dynamics model in genie_lite/model.py.
It stands in for "the real emulator" that a full project would use as the
data source.
"""
import random

ROOMS = 4          # 0..3, simple 2x2 room grid
GRID = 8           # each room is an 8x8 tile grid
ACTIONS = ["up", "down", "left", "right", "attack", "wait"]

# fixed enemy position per room, fixed item pickup tile per room
ENEMY_POS = {0: (5, 2), 1: (2, 6), 2: (6, 5), 3: (1, 1)}
ITEM_POS = {0: (7, 7), 1: (0, 0), 2: (3, 3), 3: (6, 6)}


def initial_state():
    return {"room": 0, "x": 0, "y": 0, "hp": 3, "item": 0}


def step(state, action):
    """Deterministic transition: this plays the role of 'ground truth'
    the dynamics model is trying to learn to imitate."""
    s = dict(state)
    x, y, room, hp, item = s["x"], s["y"], s["room"], s["hp"], s["item"]

    if action == "up":
        y = max(0, y - 1)
    elif action == "down":
        y = min(GRID - 1, y + 1)
    elif action == "left":
        x = max(0, x - 1)
    elif action == "right":
        x = min(GRID - 1, x + 1)
    elif action == "attack":
        ex, ey = ENEMY_POS[room]
        if abs(x - ex) <= 1 and abs(y - ey) <= 1:
            pass  # enemy defeated; no persistent enemy-dead flag, kept tiny
    # "wait" does nothing

    # room transitions when walking off an edge, 2x2 room layout
    if x == 0 and action == "left" and room in (1, 3):
        room -= 1
        x = GRID - 1
    elif x == GRID - 1 and action == "right" and room in (0, 2):
        room += 1
        x = 0
    elif y == 0 and action == "up" and room in (2, 3):
        room -= 2
        y = GRID - 1
    elif y == GRID - 1 and action == "down" and room in (0, 1):
        room += 2
        y = 0

    # touching the enemy tile (any action) costs HP
    ex, ey = ENEMY_POS[room]
    if (x, y) == (ex, ey) and hp > 0:
        hp -= 1

    # standing on the item tile picks it up
    if (x, y) == ITEM_POS[room]:
        item = 1

    return {"room": room, "x": x, "y": y, "hp": hp, "item": item}


def random_episode(length=40, seed=None):
    rng = random.Random(seed)
    s = initial_state()
    transitions = []
    for _ in range(length):
        a = rng.choice(ACTIONS)
        s2 = step(s, a)
        transitions.append((s, a, s2))
        s = s2
    return transitions
