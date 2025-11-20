import numpy as np

class Game:

    def __init__(self, 
                 num_bullets, 
                 radius_bullets = 0.02, 
                 radius_player = 0.02, 
                 bullets_step = 0.005, 
                 player_step = 0.005):
        
        self.num_bullets = num_bullets
        self.radius_bullets = radius_bullets
        self.radius_player = radius_player
        self.bullets_step = bullets_step
        self.player_step = player_step

        
        self.positions = np.random.rand(num_bullets + 1, 2) * (1.0 - 2.0 * (radius_bullets+bullets_step)) + radius_bullets + bullets_step # Random positions in range [radius_bullets+bullets_step, 1 - (radius_bullets+bullets_step)]
        self.positions[0, :] = 0.5  # Player starts at the center

        half_bullets = num_bullets // 2
        self.positions[1:half_bullets//2 + 1, 0] = radius_bullets + bullets_step # Top bullets
        self.positions[half_bullets//2 + 1:half_bullets + 1, 0] = 1.0 - (radius_bullets + bullets_step) # Bottom bullets
        self.positions[half_bullets + 1:half_bullets + half_bullets//2 + 1, 1] = radius_bullets + bullets_step # Left bullets
        self.positions[half_bullets + half_bullets//2 + 1:, 1] = 1.0 - (radius_bullets + bullets_step) # Right bullets


        self.directions = np.random.rand(num_bullets + 1, 2)
        self.directions = self.directions * 2.0 - 1.0  # Random directions in range [-1, 1]

        half_bullets = num_bullets // 2
        self.directions[1:half_bullets//2 + 1, 0] = np.random.rand() * 0.8 + 0.2 # Top bullets
        self.directions[half_bullets//2 + 1:half_bullets + 1, 0] = -(np.random.rand() * 0.8 + 0.2) # Bottom bullets
        self.directions[half_bullets + 1:half_bullets + half_bullets//2 + 1, 1] = np.random.rand() * 0.8 + 0.2 # Left bullets
        self.directions[half_bullets + half_bullets//2 + 1:, 1] = -(np.random.rand() * 0.8 + 0.2) # Right bullets

        # Force first bullet to point to the player
        vec_to_player = self.positions[0, :] - self.positions[1, :]
        self.directions[1, :] = vec_to_player / np.linalg.norm(vec_to_player)

        self.normalize_directions()
    
    def normalize_directions(self):
        norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.directions = self.directions / norms

    def get_state(self):
        """
        Returns the current game state. State consists of x, y of the player and
        x, y of each bullet, all between 0 and 1.

        Returns
        -------
        np.ndarray
            The current game state as a 1D array.
        """

        return self.positions.flatten()
    
    def get_state_velocities(self):
        """
        Returns the current game state. State consists of x, y of the player and
        x, y of each bullet, all between 0 and 1, as well as the x, y velocities
        of each entity.

        Returns
        -------
        np.ndarray
            The current game state as a 1D array.
        """

        return np.concatenate((self.positions.flatten(), self.directions.flatten()))
    
    def get_local_state(self):
        """
        Returns the position of the player and the relative position of the nearest bullet.

        Returns
        -------
        np.ndarray
            The local game state as a 1D array.
        """

        player_pos = self.positions[0, :]
        bullet_positions = self.positions[1:, :]
        relative_positions = bullet_positions - player_pos
        distances = np.linalg.norm(relative_positions, axis=1)
        nearest_bullet_idx = np.argmin(distances)
        nearest_bullet_rel_pos = relative_positions[nearest_bullet_idx, :]

        return np.concatenate((player_pos, nearest_bullet_rel_pos))
    
    def get_local_state_velocities(self):
        """
        Returns the position of the player and the relative position and velocity of the nearest bullet.

        Returns
        -------
        np.ndarray
            The local game state with velocities as a 1D array.
        """

        player_pos = self.positions[0, :]
        bullet_positions = self.positions[1:, :]
        bullet_directions = self.directions[1:, :]
        relative_positions = bullet_positions - player_pos
        distances = np.linalg.norm(relative_positions, axis=1)
        nearest_bullet_idx = np.argmin(distances)
        nearest_bullet_rel_pos = relative_positions[nearest_bullet_idx, :]
        nearest_bullet_dir = bullet_directions[nearest_bullet_idx, :]

        return np.concatenate((player_pos, nearest_bullet_rel_pos, nearest_bullet_dir))

    def step(self, player_direction) -> bool:
        """
        Advances the game state by one step.

        Parameters
        ----------
        player_direction: float or tuple
            If float, represents an angle in [0, 1)
            If tuple, represents the (x, y) direction vector.
            
        Returns
        -------
        bool
            True if the game continues, False if the player has collided (game over).
        """

        if type(player_direction) is not tuple:
            angle = player_direction * 2.0 * np.pi
            player_direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            player_direction = np.array(player_direction)
            norm = np.linalg.norm(player_direction)
            if norm == 0:
                player_direction[0] = 1.0
            else:
                player_direction = player_direction / np.linalg.norm(player_direction)

        # Check player collision
        distances = np.linalg.norm(self.positions[1:, :] - self.positions[0, :], axis=1)
        if np.any(distances < self.radius_bullets + self.radius_player) or np.any(self.positions[0, :] < self.radius_player) or np.any(self.positions[0, :] > 1.0 - self.radius_player):
            return False # Game over      

        self.directions[0, :] = player_direction

        # Move game entities
        self.positions[1:, :] += self.directions[1:, :] * self.bullets_step
        self.positions[0, :] += self.directions[0, :] * self.player_step
        

        # Handle boundary collisions for bullets
        under_mask = self.positions[1:, :] < self.radius_bullets
        self.positions[1:, :][under_mask] = self.radius_bullets
        over_mask = self.positions[1:, :] > 1.0 - self.radius_bullets
        self.positions[1:, :][over_mask] = 1.0 - self.radius_bullets

        compound_mask = under_mask * 1.0 + over_mask * -1.0

        self.directions[1:, :] *= (compound_mask == 0)
        self.directions[1:, :] += compound_mask * (np.random.rand(*compound_mask.shape) * 0.8 + 0.2)
        aux_mask = np.tile(np.any(compound_mask, axis=1, keepdims=True), 2) * (compound_mask == 0)
        self.directions[1:, :] *= ~aux_mask
        self.directions[1:, :] += aux_mask * (np.random.rand(*aux_mask.shape) * 2.0 - 1.0)

        self.normalize_directions()

        return True
