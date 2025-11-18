from game import Game
import pygame
import numpy as np
import neat

def draw_game(window, game, width=600, height=600):
    window.fill((0, 0, 0))  # Clear screen with black

    # Draw player
    player_pos = (int(game.positions[0, 0] * width), int(game.positions[0, 1] * height))
    pygame.draw.circle(window, (0, 255, 0), player_pos, int(game.radius_player * width))

    # Draw bullets
    for i in range(1, game.num_bullets + 1):
        bullet_pos = (int(game.positions[i, 0] * width), int(game.positions[i, 1] * height))
        pygame.draw.circle(window, (255, 0, 0), bullet_pos, int(game.radius_bullets * width))

    pygame.display.flip()  # Update the display


def game_loop(game: Game, actor: neat.nn.FeedForwardNetwork = None):
    width, height = 600, 600

    pygame.init()
    window = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    ticks = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if actor is not None:
            state = game.state()
            output = actor.activate(state)
            player_direction = (output[0], output[1])
        else:
            # get mouse position in x y and convert to [0, 1) as the angle with respect player position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            player_x, player_y = game.positions[0, 0] * width, game.positions[0, 1] * height
            # angle = np.arctan2(mouse_y - player_y, mouse_x - player_x)
            # if angle < 0:
            #     angle += 2.0 * np.pi
            # player_direction = angle / (2.0 * np.pi)
            player_direction = (mouse_x - player_x, mouse_y - player_y)

        if not game.step(player_direction):
            running = False  # Game over

        draw_game(window, game, width, height)
        clock.tick(60)  # Limit to 60 FPS
        ticks += 1
    pygame.quit()

    print(f"Game over! Survived for {ticks} ticks.")

if __name__ == "__main__":
    game = Game(num_bullets=36)
    game_loop(game)
