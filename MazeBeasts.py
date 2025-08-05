# -*- coding: utf-8 -*-
import pygame
import random
import time
import math
from collections import deque

class MazeGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        info = pygame.display.Info()
        self.screen_width = info.current_w
        self.screen_height = info.current_h
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Maze FPS")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.wall_texture = pygame.image.load('wall_texture.png').convert()
        self.boundary_texture = pygame.image.load('boundary_texture.png').convert()
        self.monster_texture = pygame.image.load('monster_texture.png').convert()
        self.monster_texture.set_colorkey((0, 0, 0))
        self.monster2_texture = pygame.image.load('monster2_texture.png').convert()
        self.monster2_texture.set_colorkey((0, 0, 0))
        self.medpack_texture = pygame.image.load('medpack.png').convert()
        self.medpack_texture.set_colorkey((0, 0, 0))
        self.wall_texture_width, self.wall_texture_height = self.wall_texture.get_size()
        self.boundary_texture_width, self.boundary_texture_height = self.boundary_texture.get_size()
        self.monster_texture_width, self.monster_texture_height = self.monster_texture.get_size()
        self.monster2_texture_width, self.monster2_texture_height = self.monster2_texture.get_size()
        self.medpack_texture_width, self.medpack_texture_height = self.medpack_texture.get_size()
        
        self.monster_sound = pygame.mixer.Sound('monster_sound.flac')
        self.boss_sound = pygame.mixer.Sound('boss_sound.flac')
        self.ambient_channel = pygame.mixer.Channel(0)
        
        self.grid_size = 33
        self.generate_maze()
        
        self.player_pos_x = self.start[0] + 0.5
        self.player_pos_y = self.start[1] + 0.5
        self.dir_x = 1.0
        self.dir_y = 0.0
        self.plane_x = 0.0
        self.plane_y = 0.66
        
        self.monsters = []
        self.spawn_monsters()
        
        self.wall_hits = {}
        self.projectiles = []
        self.monster_projectiles = []
        
        self.min_dist = 0.31
        
        self.minimap_size = 200
        self.minimap_view_radius = 5
        self.minimap_alpha = 128
        self.large_minimap_size = int(self.minimap_size * math.sqrt(2)) + 10
        
        self.player_hp = 6
        self.health_packs = []
        self.spawn_health_packs()
        
        self.damage_cooldown = 0
        
        self.vert_look = 0
        
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.run()
    
    def spawn_monsters(self):
        self.monsters = []
        # Regular monsters
        path_cells = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size) if self.grid[y][x] == 0 and (x, y) != self.start and (x, y) != self.end and not any(room[0] <= x < room[0] + room[2] and room[1] <= y < room[1] + room[2] for room in self.rooms)]
        random.shuffle(path_cells)
        num_monsters = random.randint(5, 15)
        for i in range(min(num_monsters, len(path_cells))):
            mx, my = path_cells[i]
            hp = random.randint(50, 200)
            self.monsters.append({'x': mx + 0.5, 'y': my + 0.5, 'hp': hp, 'type': 1, 'target_x': mx + 0.5, 'target_y': my + 0.5, 'cooldown': random.randint(0, 180)})
        
        # Boss monsters in rooms
        for room_x, room_y, room_size in self.rooms:
            mx = room_x + room_size // 2
            my = room_y + room_size // 2
            self.monsters.append({'x': mx + 0.5, 'y': my + 0.5, 'hp': 700, 'type': 2, 'target_x': mx + 0.5, 'target_y': my + 0.5, 'cooldown': random.randint(0, 60)})
    
    def spawn_health_packs(self):
        all_path_cells = [(x, y) for y in range(self.grid_size) for x in range(self.grid_size) if self.grid[y][x] == 0 and (x, y) != self.start and (x, y) != self.end]
        random.shuffle(all_path_cells)
        self.health_packs = []
        for i in range(2):
            if i < len(all_path_cells):
                hx, hy = all_path_cells[i]
                self.health_packs.append({'x': hx + random.uniform(0.2, 0.8), 'y': hy + random.uniform(0.2, 0.8)})
    
    def find_path(self):
        visited = set()
        queue = deque([(self.start, [self.start])])
        visited.add(self.start)
        
        while queue:
            pos, path = queue.popleft()
            if pos == self.end:
                return path
            for neigh in self.connections.get(pos, set()):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, path + [neigh]))
        return None

    def is_solvable(self):
        return self.find_path() is not None

    def generate_maze(self):
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            self.grid = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            self.connections = {}
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            self.start = (0, 0)
            half_size = self.grid_size // 2
            self.end = (random.randint(half_size, self.grid_size - 1), random.randint(half_size, self.grid_size - 1))
            
            visited = set()
            self.grid[0][0] = 0
            visited.add(self.start)
            stack = [(self.start, None)]  # (position, incoming_direction)
            
            max_dist = self.grid_size * 2
            extra_branch_prob = 0.3
            
            while stack:
                current, in_dir = stack[-1]
                x, y = current
                unvisited_neighbors = []
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in visited:
                        unvisited_neighbors.append((nx, ny, (dx, dy)))
                
                if unvisited_neighbors:
                    # Calculate weights with bias towards end and straight
                    weights = []
                    for nx, ny, d in unvisited_neighbors:
                        dist = abs(nx - self.end[0]) + abs(ny - self.end[1])
                        w = (max_dist - dist) ** 2
                        if in_dir and d == in_dir:
                            w *= 5  # Prefer straight
                        weights.append(w if w > 0 else 0.001)
                    
                    chosen_idx = random.choices(range(len(unvisited_neighbors)), weights=weights, k=1)[0]
                    nx, ny, d = unvisited_neighbors[chosen_idx]
                    self.connections.setdefault(current, set()).add((nx, ny))
                    self.connections.setdefault((nx, ny), set()).add(current)
                    self.grid[ny][nx] = 0
                    visited.add((nx, ny))
                    stack.append(((nx, ny), d))
                    
                    del unvisited_neighbors[chosen_idx]
                    weights.pop(chosen_idx)
                    
                    # Possibly add an extra branch
                    if random.random() < extra_branch_prob and unvisited_neighbors:
                        extra_chosen_idx = random.randint(0, len(unvisited_neighbors) - 1)
                        nx, ny, d = unvisited_neighbors[extra_chosen_idx]
                        self.connections.setdefault(current, set()).add((nx, ny))
                        self.connections.setdefault((nx, ny), set()).add(current)
                        self.grid[ny][nx] = 0
                        visited.add((nx, ny))
                        stack.append(((nx, ny), d))
                else:
                    stack.pop()
            
            if not stack or stack[-1][0] != self.end:
                current_x, current_y = stack[-1][0] if stack else self.start
                while current_x != self.end[0]:
                    step_dir = 1 if self.end[0] > current_x else -1
                    next_x = current_x + step_dir
                    if 0 <= next_x < self.grid_size and self.grid[current_y][next_x] == 1:
                        self.grid[current_y][next_x] = 0
                        visited.add((next_x, current_y))
                        self.connections.setdefault((current_x, current_y), set()).add((next_x, current_y))
                        self.connections.setdefault((next_x, current_y), set()).add((current_x, current_y))
                    current_x = next_x
                while current_y != self.end[1]:
                    step_dir = 1 if self.end[1] > current_y else -1
                    next_y = current_y + step_dir
                    if 0 <= next_y < self.grid_size and self.grid[next_y][current_x] == 1:
                        self.grid[next_y][current_x] = 0
                        visited.add((current_x, next_y))
                        self.connections.setdefault((current_x, current_y), set()).add((current_x, next_y))
                        self.connections.setdefault((current_x, next_y), set()).add((current_x, current_y))
                    current_y = next_y
            
            # Add very few randomly placed large rooms
            self.rooms = []
            num_rooms = random.randint(1, 3)
            for _ in range(num_rooms):
                room_size = random.randint(3, 5)
                max_start = self.grid_size - room_size
                room_x = random.randint(0, max_start)
                room_y = random.randint(0, max_start)
                self.rooms.append((room_x, room_y, room_size))
                # Connect all cells within the room to form an open space
                for ry in range(room_y, room_y + room_size):
                    for rx in range(room_x, room_x + room_size):
                        self.grid[ry][rx] = 0
                        # Connect to the right
                        if rx < room_x + room_size - 1:
                            self.connections.setdefault((rx, ry), set()).add((rx + 1, ry))
                            self.connections.setdefault((rx + 1, ry), set()).add((rx, ry))
                        # Connect down
                        if ry < room_y + room_size - 1:
                            self.connections.setdefault((rx, ry), set()).add((rx, ry + 1))
                            self.connections.setdefault((rx, ry + 1), set()).add((rx, ry))
            
            # Compute original shortest path length
            original_path = self.find_path()
            original_len = len(original_path) if original_path else 0
            
            # Add loops without shortening the shortest path
            possible_edges = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[y][x] == 0:
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if nx > x or (nx == x and ny > y):  # Avoid duplicates
                                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny][nx] == 0:
                                    cell1 = (x, y)
                                    cell2 = (nx, ny)
                                    if cell2 not in self.connections.get(cell1, set()):
                                        possible_edges.append((cell1, cell2))
            
            random.shuffle(possible_edges)
            added_loops = 0
            max_loops = 10
            for edge in possible_edges:
                u, v = edge
                self.connections.setdefault(u, set()).add(v)
                self.connections.setdefault(v, set()).add(u)
                new_path = self.find_path()
                new_len = len(new_path) if new_path else 0
                if new_path and new_len >= original_len:
                    added_loops += 1
                    if added_loops >= max_loops:
                        break
                else:
                    self.connections[u].remove(v)
                    self.connections[v].remove(u)
                    if len(self.connections[u]) == 0:
                        del self.connections[u]
                    if len(self.connections[v]) == 0:
                        del self.connections[v]
            
            if self.is_solvable():
                break
            attempt += 1
        
        if attempt >= max_attempts:
            raise Exception("Failed to generate a solvable maze")
        
        print("Maze generation complete")

    def regenerate_maze(self):
        self.generate_maze()
        self.player_pos_x = self.start[0] + 0.5
        self.player_pos_y = self.start[1] + 0.5
        self.spawn_monsters()
        self.wall_hits = {}
        self.projectiles = []
        self.monster_projectiles = []
        self.player_hp = 6
        self.spawn_health_packs()
        self.damage_cooldown = 0
        self.vert_look = 0

    def try_move(self, new_x, new_y):
        new_cell_x = int(new_x)
        new_cell_y = int(new_y)
        if not (0 <= new_cell_x < self.grid_size and 0 <= new_cell_y < self.grid_size):
            return False
        if self.grid[new_cell_y][new_cell_x] != 0:
            return False
        old_cell_x = int(self.player_pos_x)
        old_cell_y = int(self.player_pos_y)
        if new_cell_x == old_cell_x and new_cell_y == old_cell_y:
            return True
        if (new_cell_x, new_cell_y) in self.connections.get((old_cell_x, old_cell_y), set()):
            return True
        return False
    
    def shoot(self):
        # Cast ray for center column
        camera_x = 0
        ray_dir_x = self.dir_x
        ray_dir_y = self.dir_y
        map_x = int(self.player_pos_x)
        map_y = int(self.player_pos_y)
        delta_dist_x = 1e30 if ray_dir_x == 0 else abs(1 / ray_dir_x)
        delta_dist_y = 1e30 if ray_dir_y == 0 else abs(1 / ray_dir_y)
        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (self.player_pos_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1 - self.player_pos_x) * delta_dist_x
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (self.player_pos_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1 - self.player_pos_y) * delta_dist_y
        hit = False
        side = 0
        is_boundary = False
        while not hit:
            old_map_x = map_x
            old_map_y = map_y
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
                if map_x < 0 or map_x >= self.grid_size:
                    is_boundary = True
                    break
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
                if map_y < 0 or map_y >= self.grid_size:
                    is_boundary = True
                    break
            if self.grid[map_y][map_x] != 0 or (map_x, map_y) not in self.connections.get((old_map_x, old_map_y), set()):
                hit = True
        if is_boundary:
            if side == 0:
                boundary = 0 if step_x < 0 else self.grid_size
                perp_dist = (boundary - self.player_pos_x) / ray_dir_x if ray_dir_x != 0 else 1e30
            else:
                boundary = 0 if step_y < 0 else self.grid_size
                perp_dist = (boundary - self.player_pos_y) / ray_dir_y if ray_dir_y != 0 else 1e30
        else:
            if side == 0:
                perp_dist = side_dist_x - delta_dist_x
            else:
                perp_dist = side_dist_y - delta_dist_y
            if perp_dist <= 0:
                perp_dist = 0.01
        wall_dist = perp_dist

        # Check for monster hits
        hit_monster = None
        min_dist = wall_dist
        hit_vert_pos = 0
        for monster in self.monsters:
            sprite_x = monster['x'] - self.player_pos_x
            sprite_y = monster['y'] - self.player_pos_y
            inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
            transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
            transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
            if transform_y > 0:
                sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
                sprite_height = abs(int(self.screen_height / transform_y))
                sprite_width = abs(int(self.screen_height / transform_y))
                draw_start_x = sprite_screen_x - sprite_width // 2
                draw_end_x = sprite_screen_x + sprite_width // 2
                center_col = self.screen_width // 2
                if draw_start_x <= center_col <= draw_end_x and transform_y < min_dist:
                    min_dist = transform_y
                    hit_monster = monster
                    draw_start_y = -sprite_height // 2 + self.screen_height // 2 + self.vert_look
                    if monster['type'] == 2:
                        draw_start_y += sprite_height // 2
                    draw_end_y = draw_start_y + sprite_height
                    vert_pos = (self.screen_height // 2 - draw_start_y) / sprite_height
                    hit_vert_pos = vert_pos
        if hit_monster:
            if hit_monster['type'] == 1 and hit_vert_pos < 0.3:
                hit_monster['hp'] = 0
            else:
                hit_monster['hp'] -= 50
            if hit_monster['hp'] <= 0:
                self.monsters.remove(hit_monster)
        
        # Add visual projectile
        self.projectiles.append({'x': self.player_pos_x, 'y': self.player_pos_y, 'dir_x': self.dir_x, 'dir_y': self.dir_y})
    
    def update_projectiles(self):
        for proj in self.projectiles[:]:
            new_x = proj['x'] + proj['dir_x'] * 0.2
            new_y = proj['y'] + proj['dir_y'] * 0.2
            hit_wall = False
            old_cell_x = int(proj['x'])
            old_cell_y = int(proj['y'])
            new_cell_x = int(new_x)
            new_cell_y = int(new_y)
            if not (0 <= new_cell_x < self.grid_size and 0 <= new_cell_y < self.grid_size):
                hit_wall = True
            elif self.grid[new_cell_y][new_cell_x] != 0:
                hit_wall = True
            elif (new_cell_x != old_cell_x or new_cell_y != old_cell_y) and (new_cell_x, new_cell_y) not in self.connections.get((old_cell_x, old_cell_y), set()):
                hit_wall = True
            
            if hit_wall:
                self.projectiles.remove(proj)
            else:
                proj['x'] = new_x
                proj['y'] = new_y
    
    def update_monster_projectiles(self):
        for proj in self.monster_projectiles[:]:
            new_x = proj['x'] + proj['dir_x'] * 0.2
            new_y = proj['y'] + proj['dir_y'] * 0.2
            hit_wall = False
            old_cell_x = int(proj['x'])
            old_cell_y = int(proj['y'])
            new_cell_x = int(new_x)
            new_cell_y = int(new_y)
            if not (0 <= new_cell_x < self.grid_size and 0 <= new_cell_y < self.grid_size):
                hit_wall = True
            elif self.grid[new_cell_y][new_cell_x] != 0:
                hit_wall = True
            elif (new_cell_x != old_cell_x or new_cell_y != old_cell_y) and (new_cell_x, new_cell_y) not in self.connections.get((old_cell_x, old_cell_y), set()):
                hit_wall = True
            
            hit_player = math.hypot(new_x - self.player_pos_x, new_y - self.player_pos_y) < 0.3
            
            if hit_wall or hit_player:
                self.monster_projectiles.remove(proj)
            
            if hit_player:
                self.player_hp -= 1
            elif not hit_wall:
                proj['x'] = new_x
                proj['y'] = new_y

    def draw_3d(self):
        horizon_y = self.screen_height // 2 + self.vert_look
        horizon_y = max(0, min(horizon_y, self.screen_height))
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 0, self.screen_width, horizon_y))  # ceiling
        pygame.draw.rect(self.screen, (28, 3, 0), (0, horizon_y, self.screen_width, self.screen_height - horizon_y))  # floor
        
        zbuffer = [1e30 for _ in range(self.screen_width)]
        for col in range(self.screen_width):
            camera_x = 2 * col / self.screen_width - 1
            ray_dir_x = self.dir_x + self.plane_x * camera_x
            ray_dir_y = self.dir_y + self.plane_y * camera_x
            map_x = int(self.player_pos_x)
            map_y = int(self.player_pos_y)
            delta_dist_x = 1e30 if ray_dir_x == 0 else abs(1 / ray_dir_x)
            delta_dist_y = 1e30 if ray_dir_y == 0 else abs(1 / ray_dir_y)
            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (self.player_pos_x - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1 - self.player_pos_x) * delta_dist_x
            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (self.player_pos_y - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1 - self.player_pos_y) * delta_dist_y
            hit = False
            side = 0
            is_boundary = False
            while not hit:
                old_map_x = map_x
                old_map_y = map_y
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                    if map_x < 0 or map_x >= self.grid_size:
                        is_boundary = True
                        break
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                    if map_y < 0 or map_y >= self.grid_size:
                        is_boundary = True
                        break
                if self.grid[map_y][map_x] != 0 or (map_x, map_y) not in self.connections.get((old_map_x, old_map_y), set()):
                    hit = True
            if is_boundary:
                if side == 0:
                    boundary = 0 if step_x < 0 else self.grid_size
                    perp_dist = (boundary - self.player_pos_x) / ray_dir_x
                    wall_x = self.player_pos_y + perp_dist * ray_dir_y
                else:
                    boundary = 0 if step_y < 0 else self.grid_size
                    perp_dist = (boundary - self.player_pos_y) / ray_dir_y
                    wall_x = self.player_pos_x + perp_dist * ray_dir_x
                if perp_dist <= 0:
                    perp_dist = 0.01
                wall_x -= math.floor(wall_x)
                tex_x = int(wall_x * self.boundary_texture_width)
                tex_x = max(0, min(tex_x, self.boundary_texture_width - 1))
                if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
                    tex_x = self.boundary_texture_width - tex_x - 1
                line_height = int(self.screen_height / perp_dist)
                draw_start = -line_height // 2 + self.screen_height // 2 + self.vert_look
                if draw_start < 0:
                    draw_start = 0
                draw_end = line_height // 2 + self.screen_height // 2 + self.vert_look
                if draw_end > self.screen_height:
                    draw_end = self.screen_height
                tex_column = self.boundary_texture.subsurface((tex_x, 0, 1, self.boundary_texture_height))
                scaled_column = pygame.transform.scale(tex_column, (1, draw_end - draw_start))
                self.screen.blit(scaled_column, (col, draw_start))
                zbuffer[col] = perp_dist
                continue
            if side == 0:
                perp_dist = side_dist_x - delta_dist_x
            else:
                perp_dist = side_dist_y - delta_dist_y
            if perp_dist <= 0:
                perp_dist = 0.01
        
            # Calculate texture coordinates
            if side == 0:
                wall_x = self.player_pos_y + perp_dist * ray_dir_y
            else:
                wall_x = self.player_pos_x + perp_dist * ray_dir_x
            wall_x -= math.floor(wall_x)
            tex_x = int(wall_x * self.wall_texture_width)
            tex_x = max(0, min(tex_x, self.wall_texture_width - 1))
            if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
                tex_x = self.wall_texture_width - tex_x - 1
            
            line_height = int(self.screen_height / perp_dist)
            draw_start = -line_height // 2 + self.screen_height // 2 + self.vert_look
            if draw_start < 0:
                draw_start = 0
            draw_end = line_height // 2 + self.screen_height // 2 + self.vert_look
            if draw_end > self.screen_height:
                draw_end = self.screen_height
        
            # Extract and scale texture column
            tex_column = self.wall_texture.subsurface((tex_x, 0, 1, self.wall_texture_height))
            scaled_column = pygame.transform.scale(tex_column, (1, draw_end - draw_start))
            self.screen.blit(scaled_column, (col, draw_start))
            
            zbuffer[col] = perp_dist
        
        monsters_sorted = sorted(self.monsters, key=lambda m: -((m['x'] - self.player_pos_x)**2 + (m['y'] - self.player_pos_y)**2))
        for monster in monsters_sorted:
            sprite_x = monster['x'] - self.player_pos_x
            sprite_y = monster['y'] - self.player_pos_y
            inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
            transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
            transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
            if transform_y > 0:
                sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
                sprite_height = abs(int(self.screen_height / transform_y))
                draw_start_y = -sprite_height // 2 + self.screen_height // 2 + self.vert_look
                if draw_start_y < 0:
                    draw_start_y = 0
                draw_end_y = sprite_height // 2 + self.screen_height // 2 + self.vert_look
                if draw_end_y > self.screen_height:
                    draw_end_y = self.screen_height
                sprite_width = abs(int(self.screen_height / transform_y))
                draw_start_x = sprite_screen_x - sprite_width // 2
                draw_end_x = sprite_screen_x + sprite_width // 2
                if monster.get('type', 1) == 2:
                    tex = self.monster2_texture
                    tex_width = self.monster2_texture_width
                    tex_height = self.monster2_texture_height
                    vertical_shift = sprite_height // 2
                    draw_start_y += vertical_shift
                    draw_end_y += vertical_shift
                    if draw_end_y > self.screen_height:
                        draw_end_y = self.screen_height
                    if draw_start_y > self.screen_height:
                        continue
                else:
                    tex = self.monster_texture
                    tex_width = self.monster_texture_width
                    tex_height = self.monster_texture_height
                for strip in range(max(draw_start_x, 0), min(draw_end_x, self.screen_width)):
                    if transform_y < zbuffer[strip]:
                        tex_x = int((strip - draw_start_x) * tex_width / sprite_width)
                        tex_x = max(0, min(tex_x, tex_width - 1))
                        tex_column = tex.subsurface((tex_x, 0, 1, tex_height))
                        scaled_column = pygame.transform.smoothscale(tex_column, (1, draw_end_y - draw_start_y))
                        self.screen.blit(scaled_column, (strip, draw_start_y))
        
        # Draw player projectiles
        for proj in self.projectiles:
            sprite_x = proj['x'] - self.player_pos_x
            sprite_y = proj['y'] - self.player_pos_y
            inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
            transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
            transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
            if transform_y > 0:
                sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
                sprite_height = abs(int(self.screen_height / transform_y)) // 40  # Even smaller size
                draw_start_y = -sprite_height // 2 + self.screen_height // 2
                if draw_start_y < 0:
                    draw_start_y = 0
                draw_end_y = sprite_height // 2 + self.screen_height // 2
                if draw_end_y > self.screen_height:
                    draw_end_y = self.screen_height
                sprite_width = sprite_height  # Square
                draw_start_x = sprite_screen_x - sprite_width // 2
                draw_end_x = sprite_screen_x + sprite_width // 2
                for strip in range(max(draw_start_x, 0), min(draw_end_x, self.screen_width)):
                    if transform_y < zbuffer[strip]:
                        pygame.draw.line(self.screen, (255, 255, 0), (strip, draw_start_y), (strip, draw_end_y))
        
        # Draw monster projectiles
        for proj in self.monster_projectiles:
            sprite_x = proj['x'] - self.player_pos_x
            sprite_y = proj['y'] - self.player_pos_y
            inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
            transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
            transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
            if transform_y > 0:
                sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
                sprite_height = abs(int(self.screen_height / transform_y)) // 40  # Even smaller size
                draw_start_y = -sprite_height // 2 + self.screen_height // 2 + self.vert_look
                if proj.get('type', 1) == 2:
                    draw_start_y += sprite_height // 2 + 180
                if draw_start_y < 0:
                    draw_start_y = 0
                draw_end_y = draw_start_y + sprite_height  # Keep height fixed to make square
                if draw_end_y > self.screen_height:
                    draw_end_y = self.screen_height
                sprite_width = sprite_height  # Square
                draw_start_x = sprite_screen_x - sprite_width // 2
                draw_end_x = sprite_screen_x + sprite_width // 2
                for strip in range(max(draw_start_x, 0), min(draw_end_x, self.screen_width)):
                    if transform_y < zbuffer[strip]:
                        pygame.draw.line(self.screen, (255, 0, 0), (strip, draw_start_y), (strip, draw_end_y))
        
        # Draw end sprite
        end_x = self.end[0] + 0.5
        end_y = self.end[1] + 0.5
        sprite_x = end_x - self.player_pos_x
        sprite_y = end_y - self.player_pos_y
        inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
        transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
        transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
        if transform_y > 0:
            sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
            sprite_height = abs(int(self.screen_height / transform_y))
            draw_start_y = -sprite_height // 2 + self.screen_height // 2 + self.vert_look
            if draw_start_y < 0:
                draw_start_y = 0
            draw_end_y = sprite_height // 2 + self.screen_height // 2 + self.vert_look
            if draw_end_y > self.screen_height:
                draw_end_y = self.screen_height
            sprite_width = abs(int(self.screen_height / transform_y))
            draw_start_x = sprite_screen_x - sprite_width // 2
            draw_end_x = sprite_screen_x + sprite_width // 2
            for strip in range(max(draw_start_x, 0), min(draw_end_x, self.screen_width)):
                if transform_y < zbuffer[strip]:
                    pygame.draw.line(self.screen, (255, 0, 0), (strip, draw_start_y), (strip, draw_end_y))
        
        # Draw health packs
        for hp_pack in self.health_packs:
            sprite_x = hp_pack['x'] - self.player_pos_x
            sprite_y = hp_pack['y'] - self.player_pos_y
            inv_det = 1.0 / (self.plane_x * self.dir_y - self.dir_x * self.plane_y)
            transform_x = inv_det * (self.dir_y * sprite_x - self.dir_x * sprite_y)
            transform_y = inv_det * (-self.plane_y * sprite_x + self.plane_x * sprite_y)
            if transform_y > 0:
                sprite_screen_x = int((self.screen_width / 2) * (1 + transform_x / transform_y))
                sprite_height = abs(int(self.screen_height / transform_y)) // 4  # Smaller size
                #vertical_shift = sprite_height // 2
                vertical_shift = sprite_height + 120
                draw_start_y = -sprite_height // 2 + self.screen_height // 2 + self.vert_look + vertical_shift
                if draw_start_y < 0:
                    draw_start_y = 0
                draw_end_y = sprite_height // 2 + self.screen_height // 2 + self.vert_look + vertical_shift
                if draw_end_y > self.screen_height:
                    draw_end_y = self.screen_height
                sprite_width = int(sprite_height * self.medpack_texture_width / self.medpack_texture_height)
                draw_start_x = sprite_screen_x - sprite_width // 2
                draw_end_x = sprite_screen_x + sprite_width // 2
                tex = self.medpack_texture
                tex_width = self.medpack_texture_width
                tex_height = self.medpack_texture_height
                for strip in range(max(draw_start_x, 0), min(draw_end_x, self.screen_width)):
                    if transform_y < zbuffer[strip]:
                        tex_x = int((strip - draw_start_x) * tex_width / sprite_width)
                        tex_x = max(0, min(tex_x, tex_width - 1))
                        tex_column = tex.subsurface((tex_x, 0, 1, tex_height))
                        scaled_column = pygame.transform.smoothscale(tex_column, (1, draw_end_y - draw_start_y))
                        self.screen.blit(scaled_column, (strip, draw_start_y))
        
        # Draw health bar
        health_bar_width = 200
        bar_x = 10
        bar_y = self.screen_height - 40
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, health_bar_width, 20))
        green_width = int(health_bar_width * (self.player_hp / 6.0))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, green_width, 20))
        
        # Draw boss counter
        boss_count = len([m for m in self.monsters if m['type'] == 2])
        boss_text = self.font.render(f"Bosses: {boss_count}", True, (255, 255, 255))
        self.screen.blit(boss_text, (bar_x, bar_y - 30))
        
        # Draw crosshair
        crosshair_color = (255, 255, 255)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        crosshair_size = 10
        pygame.draw.line(self.screen, crosshair_color, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), 2)
        pygame.draw.line(self.screen, crosshair_color, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), 2)
    
    def draw_map(self):
        cell_size = min(self.screen_width // self.grid_size, self.screen_height // self.grid_size)
        offset_x = (self.screen_width - self.grid_size * cell_size) // 2
        offset_y = (self.screen_height - self.grid_size * cell_size) // 2
        
        # Draw paths
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pygame.draw.rect(self.screen, (200, 200, 200), (offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size))
        
        # Draw walls as lines
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if x < self.grid_size - 1 and (x+1, y) not in self.connections.get((x, y), set()):
                    pygame.draw.line(self.screen, (50, 50, 50), (offset_x + (x+1)*cell_size, offset_y + y*cell_size), (offset_x + (x+1)*cell_size, offset_y + (y+1)*cell_size), width=2)
                if y < self.grid_size - 1 and (x, y+1) not in self.connections.get((x, y), set()):
                    pygame.draw.line(self.screen, (50, 50, 50), (offset_x + x*cell_size, offset_y + (y+1)*cell_size), (offset_x + (x+1)*cell_size, offset_y + (y+1)*cell_size), width=2)
        
        # Draw end
        pygame.draw.rect(self.screen, (255, 0, 0), (offset_x + self.end[0] * cell_size, offset_y + self.end[1] * cell_size, cell_size, cell_size))
        
        # Draw player as circle with direction line
        px = offset_x + int(self.player_pos_x * cell_size)
        py = offset_y + int(self.player_pos_y * cell_size)
        pygame.draw.circle(self.screen, (0, 255, 0), (px, py), cell_size // 4)
        dir_length = cell_size // 2
        end_dir_x = px + int(self.dir_x * dir_length)
        end_dir_y = py + int(self.dir_y * dir_length)
        pygame.draw.line(self.screen, (0, 255, 0), (px, py), (end_dir_x, end_dir_y), width=2)
        
        # Draw monsters
        for m in self.monsters:
            mx = offset_x + int(m['x'] * cell_size)
            my = offset_y + int(m['y'] * cell_size)
            pygame.draw.circle(self.screen, (36, 57, 196), (mx, my), cell_size // 4)
        
        # Draw health packs
        for hp in self.health_packs:
            hx = offset_x + int(hp['x'] * cell_size)
            hy = offset_y + int(hp['y'] * cell_size)
            pygame.draw.rect(self.screen, (173, 216, 230), (hx - cell_size // 4, hy - cell_size // 4, cell_size // 2, cell_size // 2))
    
    def draw_minimap(self):
        minimap_surf = pygame.Surface((self.large_minimap_size, self.large_minimap_size), pygame.SRCALPHA)
        center = self.large_minimap_size // 2
        cell_size = self.minimap_size // (2 * self.minimap_view_radius + 1)

        min_gx = max(0, int(self.player_pos_x) - self.minimap_view_radius)
        max_gx = min(self.grid_size - 1, int(self.player_pos_x) + self.minimap_view_radius)
        min_gy = max(0, int(self.player_pos_y) - self.minimap_view_radius)
        max_gy = min(self.grid_size - 1, int(self.player_pos_y) + self.minimap_view_radius)

        # Draw floor
        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if self.grid[gy][gx] == 0:
                    mx = int((gx - self.player_pos_x) * cell_size) + center
                    my = int((gy - self.player_pos_y) * cell_size) + center
                    pygame.draw.rect(minimap_surf, (200, 200, 200, self.minimap_alpha), (mx, my, cell_size, cell_size))

        # Draw vertical walls
        for vpos in range(min_gx, max_gx + 2):
            for gy in range(min_gy, max_gy + 1):
                draw = vpos == 0 or vpos == self.grid_size
                if 0 < vpos < self.grid_size:
                    left_x = vpos - 1
                    right_x = vpos
                    draw = self.grid[gy][left_x] != 0 or self.grid[gy][right_x] != 0 or (right_x, gy) not in self.connections.get((left_x, gy), set())
                if draw:
                    mx = int((vpos - self.player_pos_x) * cell_size) + center
                    my_top = int((gy - self.player_pos_y) * cell_size) + center
                    my_bottom = my_top + cell_size
                    pygame.draw.line(minimap_surf, (50, 50, 50, 255), (mx, my_top), (mx, my_bottom), width=2)

        # Draw horizontal walls
        for hpos in range(min_gy, max_gy + 2):
            for gx in range(min_gx, max_gx + 1):
                draw = hpos == 0 or hpos == self.grid_size
                if 0 < hpos < self.grid_size:
                    top_y = hpos - 1
                    bottom_y = hpos
                    draw = self.grid[top_y][gx] != 0 or self.grid[bottom_y][gx] != 0 or (gx, bottom_y) not in self.connections.get((gx, top_y), set())
                if draw:
                    my = int((hpos - self.player_pos_y) * cell_size) + center
                    mx_left = int((gx - self.player_pos_x) * cell_size) + center
                    mx_right = mx_left + cell_size
                    pygame.draw.line(minimap_surf, (50, 50, 50, 255), (mx_left, my), (mx_right, my), width=2)

        # Draw end if in range
        end_gx, end_gy = self.end
        if min_gx <= end_gx <= max_gx and min_gy <= end_gy <= max_gy:
            mx = int((end_gx + 0.5 - self.player_pos_x) * cell_size) + center
            my = int((end_gy + 0.5 - self.player_pos_y) * cell_size) + center
            pygame.draw.circle(minimap_surf, (255, 0, 0, 255), (mx, my), cell_size // 4)

        # Draw monsters
        for m in self.monsters:
            m_gx = int(m['x'])
            m_gy = int(m['y'])
            if min_gx <= m_gx <= max_gx and min_gy <= m_gy <= max_gy:
                mx = int((m['x'] - self.player_pos_x) * cell_size) + center
                my = int((m['y'] - self.player_pos_y) * cell_size) + center
                pygame.draw.circle(minimap_surf, (0, 0, 255, 255), (mx, my), cell_size // 4)

        # Rotate the surface
        angle_facing = math.atan2(self.dir_y, self.dir_x)
        rotation_angle = math.degrees(angle_facing) + 90
        rotated_surf = pygame.transform.rotate(minimap_surf, rotation_angle)

        rot_center_x = rotated_surf.get_width() // 2
        rot_center_y = rotated_surf.get_height() // 2
        crop_x = rot_center_x - self.minimap_size // 2
        crop_y = rot_center_y - self.minimap_size // 2
        crop_rect = pygame.Rect(crop_x, crop_y, self.minimap_size, self.minimap_size)
        cropped_surf = rotated_surf.subsurface(crop_rect)

        center_final = self.minimap_size // 2

        # Draw player
        pygame.draw.circle(cropped_surf, (0, 255, 0, 255), (center_final, center_final), cell_size // 4)

        # Direction fixed up
        dir_length = cell_size * 0.4
        end_dir_x = center_final
        end_dir_y = center_final - dir_length
        pygame.draw.line(cropped_surf, (0, 255, 0, 255), (center_final, center_final), (end_dir_x, end_dir_y), width=2)

        # Blit to screen
        self.screen.blit(cropped_surf, (10, 10))
    
    def clip_position(self, new_pos, current_pos, is_x):
        min_dist = self.min_dist
        if is_x:
            cell = int(current_pos)
            other_cell = int(self.player_pos_y)
            if new_pos > current_pos:  # moving right
                has_wall = cell == self.grid_size - 1 or (cell + 1, other_cell) not in self.connections.get((cell, other_cell), set())
                if has_wall:
                    new_pos = min(new_pos, cell + 1 - min_dist)
            elif new_pos < current_pos:  # moving left
                has_wall = cell == 0 or (cell - 1, other_cell) not in self.connections.get((cell, other_cell), set())
                if has_wall:
                    new_pos = max(new_pos, cell + min_dist)
        else:
            cell = int(current_pos)
            other_cell = int(self.player_pos_x)
            if new_pos > current_pos:  # moving down
                has_wall = cell == self.grid_size - 1 or (other_cell, cell + 1) not in self.connections.get((other_cell, cell), set())
                if has_wall:
                    new_pos = min(new_pos, cell + 1 - min_dist)
            elif new_pos < current_pos:  # moving up
                has_wall = cell == 0 or (other_cell, cell - 1) not in self.connections.get((other_cell, cell), set())
                if has_wall:
                    new_pos = max(new_pos, cell + min_dist)
        return new_pos
    
    def has_line_of_sight(self, tx, ty):
        dx = tx - self.player_pos_x
        dy = ty - self.player_pos_y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return True
        ray_dir_x = dx / dist
        ray_dir_y = dy / dist
        map_x = int(self.player_pos_x)
        map_y = int(self.player_pos_y)
        delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else 1e30
        delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else 1e30
        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (self.player_pos_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - self.player_pos_x) * delta_dist_x
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (self.player_pos_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - self.player_pos_y) * delta_dist_y
        side = 0
        while True:
            old_map_x = map_x
            old_map_y = map_y
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            if map_x < 0 or map_x >= self.grid_size or map_y < 0 or map_y >= self.grid_size:
                return False
            perp_dist = (side_dist_x - delta_dist_x) if side == 0 else (side_dist_y - delta_dist_y)
            if perp_dist > dist:
                return True
            if self.grid[map_y][map_x] != 0 or (map_x, map_y) not in self.connections.get((old_map_x, old_map_y), set()):
                return False
    
    def update_monsters(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for monster in self.monsters:
            dx = monster['target_x'] - monster['x']
            dy = monster['target_y'] - monster['y']
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.01:
                current_cell_x = int(monster['x'])
                current_cell_y = int(monster['y'])
                possible_targets = []
                for dx_dir, dy_dir in directions:
                    nx = current_cell_x + dx_dir
                    ny = current_cell_y + dy_dir
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny][nx] == 0 and (nx, ny) in self.connections.get((current_cell_x, current_cell_y), set()):
                        possible_targets.append((nx + 0.5, ny + 0.5))
                if possible_targets:
                    monster['target_x'], monster['target_y'] = random.choice(possible_targets)
            else:
                speed = 0.01
                monster['x'] += (dx / dist) * speed if dist > 0 else 0
                monster['y'] += (dy / dist) * speed if dist > 0 else 0
            
            monster['cooldown'] -= 1
            if monster['cooldown'] <= 0:
                dx = self.player_pos_x - monster['x']
                dy = self.player_pos_y - monster['y']
                dist = math.sqrt(dx**2 + dy**2)
                if dist > 0:
                    dir_x = dx / dist
                    dir_y = dy / dist
                    start_x = monster['x']
                    start_y = monster['y']
                    proj_type = monster['type']
                    if proj_type == 2:
                        offset_h = 0.2
                        perp_x = -dir_y * offset_h
                        perp_y = dir_x * offset_h
                        start_x += perp_x
                        start_y += perp_y
                    self.monster_projectiles.append({'x': start_x, 'y': start_y, 'dir_x': dir_x, 'dir_y': dir_y, 'type': proj_type})
                if monster.get('type', 1) == 2:
                    monster['cooldown'] = 60
                else:
                    monster['cooldown'] = 180
    
    def run(self):
        while True:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.VIDEORESIZE:
                    self.screen_width, self.screen_height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.shoot()
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                pygame.quit()
                return
            
            if keys[pygame.K_F8]:
                self.regenerate_maze()
            
            mouse_rel = pygame.mouse.get_rel()
            rot_speed = mouse_rel[0] * 0.003
            if rot_speed:
                old_dir_x = self.dir_x
                self.dir_x = self.dir_x * math.cos(rot_speed) - self.dir_y * math.sin(rot_speed)
                self.dir_y = old_dir_x * math.sin(rot_speed) + self.dir_y * math.cos(rot_speed)
                old_plane_x = self.plane_x
                self.plane_x = self.plane_x * math.cos(rot_speed) - self.plane_y * math.sin(rot_speed)
                self.plane_y = old_plane_x * math.sin(rot_speed) + self.plane_y * math.cos(rot_speed)
            self.vert_look -= mouse_rel[1] * 0.3
            self.vert_look = max(-300, min(300, self.vert_look))
            pygame.mouse.set_pos(self.screen_width // 2, self.screen_height // 2)
            
            move_speed = 0.05
            if keys[pygame.K_w]:
                new_x = self.player_pos_x + self.dir_x * move_speed
                new_y = self.player_pos_y + self.dir_y * move_speed
                new_x = self.clip_position(new_x, self.player_pos_x, True)
                new_y = self.clip_position(new_y, self.player_pos_y, False)
                if self.try_move(new_x, self.player_pos_y):
                    self.player_pos_x = new_x
                if self.try_move(self.player_pos_x, new_y):
                    self.player_pos_y = new_y
            if keys[pygame.K_s]:
                new_x = self.player_pos_x - self.dir_x * move_speed
                new_y = self.player_pos_y - self.dir_y * move_speed
                new_x = self.clip_position(new_x, self.player_pos_x, True)
                new_y = self.clip_position(new_y, self.player_pos_y, False)
                if self.try_move(new_x, self.player_pos_y):
                    self.player_pos_x = new_x
                if self.try_move(self.player_pos_x, new_y):
                    self.player_pos_y = new_y
            if keys[pygame.K_d]:
                new_x = self.player_pos_x + self.plane_x * move_speed
                new_y = self.player_pos_y + self.plane_y * move_speed
                new_x = self.clip_position(new_x, self.player_pos_x, True)
                new_y = self.clip_position(new_y, self.player_pos_y, False)
                if self.try_move(new_x, self.player_pos_y):
                    self.player_pos_x = new_x
                if self.try_move(self.player_pos_x, new_y):
                    self.player_pos_y = new_y
            if keys[pygame.K_a]:
                new_x = self.player_pos_x - self.plane_x * move_speed
                new_y = self.player_pos_y - self.plane_y * move_speed
                new_x = self.clip_position(new_x, self.player_pos_x, True)
                new_y = self.clip_position(new_y, self.player_pos_y, False)
                if self.try_move(new_x, self.player_pos_y):
                    self.player_pos_x = new_x
                if self.try_move(self.player_pos_x, new_y):
                    self.player_pos_y = new_y
            
            self.update_projectiles()
            self.update_monster_projectiles()
            self.update_monsters()
            
            # Manage sound effects
            nearest_regular = 999
            nearest_boss = 999
            for m in self.monsters:
                if self.has_line_of_sight(m['x'], m['y']):
                    d = math.hypot(m['x'] - self.player_pos_x, m['y'] - self.player_pos_y)
                    if m['type'] == 1:
                        nearest_regular = min(nearest_regular, d)
                    else:
                        nearest_boss = min(nearest_boss, d)
            threshold = 5.0
            current_sound = self.ambient_channel.get_sound()
            if nearest_boss < threshold:
                if current_sound != self.boss_sound:
                    self.ambient_channel.play(self.boss_sound, loops=-1)
            elif nearest_regular < threshold:
                if current_sound != self.monster_sound:
                    self.ambient_channel.play(self.monster_sound, loops=-1)
            else:
                self.ambient_channel.stop()
            
            # Check monster touch damage
            if self.damage_cooldown > 0:
                self.damage_cooldown -= 1
            else:
                for monster in self.monsters:
                    dist = math.hypot(monster['x'] - self.player_pos_x, monster['y'] - self.player_pos_y)
                    if dist < 1.0:
                        self.player_hp -= 1
                        self.damage_cooldown = 30
                        break
            
            # Check health packs
            for hp_pack in self.health_packs[:]:
                if math.hypot(hp_pack['x'] - self.player_pos_x, hp_pack['y'] - self.player_pos_y) < 0.3:
                    self.player_hp = 6
                    self.health_packs.remove(hp_pack)
            
            boss_count = len([m for m in self.monsters if m['type'] == 2])
            if int(self.player_pos_x) == self.end[0] and int(self.player_pos_y) == self.end[1] and boss_count == 0:
                win_font = pygame.font.Font(None, 50)
                text = win_font.render("You Won!", True, (255, 255, 0))
                self.screen.blit(text, (self.screen_width // 2 - text.get_width() // 2, self.screen_height // 2 - text.get_height() // 2))
                pygame.display.flip()
                time.sleep(1)
                self.regenerate_maze()
            
            if self.player_hp <= 0:
                self.screen.fill((0, 0, 0))
                win_font = pygame.font.Font(None, 50)
                text = win_font.render("You Died!", True, (255, 0, 0))
                self.screen.blit(text, (self.screen_width // 2 - text.get_width() // 2, self.screen_height // 2 - text.get_height() // 2))
                pygame.display.flip()
                time.sleep(1)
                self.regenerate_maze()
                continue
            
            self.screen.fill((0, 0, 0))
            if keys[pygame.K_TAB]:
                self.draw_map()
            else:
                self.draw_3d()
                self.draw_minimap()
            
            pygame.display.flip()

if __name__ == "__main__":
    try:
        game = MazeGame()
    except Exception as e:
        print(f"Error launching game: {e}")