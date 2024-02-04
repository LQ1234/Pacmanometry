import matplotlib.pyplot as plt
from gameengine_reference.grid import grid
from gameengine_reference.variables import I
from matplotlib import collections  as mc

# o = normal pellet, e = empty space, O = power pellet, c = cherry position
# I = wall, n = ghost chambers

width = len(grid)
height = len(grid[0])
blit = None

fig, ax = None, None

artists = []

def initialize_plot():
    global blit, fig, ax

    fig, ax = plt.subplots()
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    lines = []

    for x in range(width):
        for y in range(height):
            if grid[x][y] == I:
                if x == 0 or grid[x-1][y] != I:
                    lines.append([(x, y), (x, y+1)])
                if y == 0 or grid[x][y-1] != I:
                    lines.append([(x, y), (x+1, y)])
                if x == width-1 or grid[x+1][y] != I:
                    lines.append([(x+1, y), (x+1, y+1)])
                if y == height-1 or grid[x][y+1] != I:
                    lines.append([(x, y+1), (x+1, y+1)])
                    
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)
    plt.axis('equal')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)

    fig.canvas.draw()
    blit = fig.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)
    return fig, ax


def set_artists(artists_new):
    global artists
    for artist in artists:
        artist.remove()

    artists.clear()
    artists.extend(artists_new)

    for artist in artists:
        artist.set_animated(True)
        ax.add_artist(artist)

    fig.canvas.draw()

def create_short_arrow(x, y, dir, color):
    dx, dy = 0, 0
    if dir == 0:
        dx = 1
    elif dir == 1:
        dy = 1
    elif dir == 2:
        dx = -1
    elif dir == 3:
        dy = -1
    return plt.Arrow(
        x + 0.5 - dx / 2 ,
        y + 0.5 - dy / 2,
        dx,
        dy,
        color=color,
        width = 2,
    )

def create_arrow(x2, y2, x, y, color):
    return plt.Arrow(
        x + 0.5,
        y + 0.5,
        x2 - x,
        y2 - y,
        color=color,
        width = 2
    )
def draw_game_state(game_state, simulation= None):
    artists = []
    pacbot_pos = game_state["pacbot_pos"]
    pacbot_dir = game_state["pacbot_dir"]
    pacbot = create_short_arrow(pacbot_pos[0], pacbot_pos[1], pacbot_dir, "yellow")
    artists.append(pacbot)

    pink = create_arrow(*game_state["pink"]["next"], *game_state["pink"]["current"], "pink")
    artists.append(pink)

    orange = create_arrow(*game_state["orange"]["next"], *game_state["orange"]["current"], "orange")
    artists.append(orange)

    blue = create_arrow(*game_state["blue"]["next"], *game_state["blue"]["current"], "blue")
    artists.append(blue)

    red = create_arrow(*game_state["red"]["next"], *game_state["red"]["current"], "red")
    artists.append(red)


    if simulation is not None:
        pink_path = simulation[:]["pink"]["next"]
        pink_path_deterministic = simulation[:]["pink"]["deterministic"]
        line = plt.Line2D(pink_path[:,0][pink_path_deterministic] + 0.5, pink_path[:,1][pink_path_deterministic] + 0.5, color="pink", linestyle="dotted", linewidth=2)
        artists.append(line)
        line = plt.Line2D(pink_path[:,0][~pink_path_deterministic] + 0.5, pink_path[:,1][~pink_path_deterministic] + 0.5, color="pink", linestyle="dotted", linewidth=2, alpha=0.5)
        artists.append(line)

        orange_path = simulation[:]["orange"]["next"]
        orange_path_deterministic = simulation[:]["orange"]["deterministic"]
        line = plt.Line2D(orange_path[:,0][orange_path_deterministic] + 0.5, orange_path[:,1][orange_path_deterministic] + 0.5, color="orange", linestyle="dotted", linewidth=2)
        artists.append(line)
        line = plt.Line2D(orange_path[:,0][~orange_path_deterministic] + 0.5, orange_path[:,1][~orange_path_deterministic] + 0.5, color="orange", linestyle="dotted", linewidth=2, alpha=0.5)
        artists.append(line)

        blue_path = simulation[:]["blue"]["next"]
        blue_path_deterministic = simulation[:]["blue"]["deterministic"]
        line = plt.Line2D(blue_path[:,0][blue_path_deterministic] + 0.5, blue_path[:,1][blue_path_deterministic] + 0.5, color="blue", linestyle="dotted", linewidth=2)
        artists.append(line)
        line = plt.Line2D(blue_path[:,0][~blue_path_deterministic] + 0.5, blue_path[:,1][~blue_path_deterministic] + 0.5, color="blue", linestyle="dotted", linewidth=2, alpha=0.5)
        artists.append(line)

        red_path = simulation[:]["red"]["next"]
        red_path_deterministic = simulation[:]["red"]["deterministic"]
        line = plt.Line2D(red_path[:,0][red_path_deterministic] + 0.5, red_path[:,1][red_path_deterministic] + 0.5, color="red", linestyle="dotted", linewidth=2)
        artists.append(line)
        line = plt.Line2D(red_path[:,0][~red_path_deterministic] + 0.5, red_path[:,1][~red_path_deterministic] + 0.5, color="red", linestyle="dotted", linewidth=2, alpha=0.5)
        artists.append(line)


        

        

    set_artists(artists)

def update_plot():
    global blit
    fig.canvas.restore_region(blit)
    for artist in artists:
        ax.draw_artist(artist)
    fig.canvas.blit(ax.bbox)
    fig.canvas.start_event_loop(0.001)
