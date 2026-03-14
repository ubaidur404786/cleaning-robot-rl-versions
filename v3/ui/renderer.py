"""
Pygame renderer for the 15x15 cleaning robot apartment.

Draws walls, furniture, dirt, the charger, and a directional robot arrow.
Shows a side panel with a live battery bar and episode stats.

Optional sprites
----------------
Put PNG images in a  sprites/  sub-folder to get nicer visuals.
If any are missing the renderer falls back to plain coloured shapes.

  sprites/wall.png        – wall tile  (42×42)
  sprites/furniture.png   – furniture tile
  sprites/charger.png     – charger tile
  sprites/dirt.png        – dirt overlay (transparent background)
  sprites/robot.png       – robot; should face North (up) by default
"""

import os
import pygame
import numpy as np

# use a deferred import so this module doesn't crash when pygame is absent
from env.phase2_cleaning_env import (
    Phase2CleaningEnv,
    ORIENT_NORTH, ORIENT_EAST, ORIENT_SOUTH, ORIENT_WEST,
)

# ── layout constants ──────────────────────────────────────────────────────────
CELL   = 42          # pixels per grid cell
MARGIN = 10          # gap around the grid
INFO_W = 252         # width of the right-side info panel

WIN_W  = MARGIN + 15 * CELL + MARGIN + INFO_W
WIN_H  = MARGIN + 15 * CELL + MARGIN

# ── colour palette (Catppuccin-inspired dark theme) ───────────────────────────
BG_COL      = (24,  24,  37)   # window background
GRID_COL    = (45,  45,  63)   # grid divider lines
FLOOR_COL   = (210, 210, 220)  # clean, walkable tile
WALL_COL    = (72,  72,  90)   # wall tile
FURN_COL    = (130, 90,  45)   # furniture tile
CHARGER_COL = (255, 210,  0)   # charger station
DIRT_COL    = (175, 150, 85)   # dirt dot
ROBOT_COL   = (0,   170, 230)  # robot body
ARROW_COL   = (255, 255, 255)  # direction arrow on robot
PANEL_COL   = (18,  18,  28)   # side-panel background
TXT_COL     = (210, 210, 225)  # all panel text
BAR_OK_COL  = ( 90, 200,  90)  # battery bar — healthy
BAR_LOW_COL = (220,  60,  60)  # battery bar — critical

SPRITES_DIR = "sprites"


class ApartmentRenderer:
    """
    Create one instance before the training loop, then call
    render() after every env.step().

    Example usage inside a training loop:
        renderer = ApartmentRenderer()
        obs, _ = env.reset()
        for step in range(max_steps):
            action = agent.choose_action(...)
            obs, reward, done, _, info = env.step(action)
            alive = renderer.render(env, episode=ep, step=step, ...)
            if not alive or done:
                break
        renderer.close()
    """

    def __init__(self, caption: str = "Cleaning Robot RL"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(caption)
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)
        self._sprites: dict = {}
        self._load_sprites()

    # ── sprite loading ────────────────────────────────────────────────────────

    def _load_sprites(self):
        """
        Try to load PNG files from the sprites/ folder.
        Missing files are silently skipped; draw methods use shapes instead.
        """
        want = {
            "wall":    "wall.png",
            "furn":    "furniture.png",
            "charger": "charger.png",
            "dirt":    "dirt.png",
            "robot":   "robot.png",
        }
        for key, fname in want.items():
            path = os.path.join(SPRITES_DIR, fname)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                self._sprites[key] = pygame.transform.smoothscale(img, (CELL, CELL))

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _rect(self, r: int, c: int) -> pygame.Rect:
        """Grid cell (row, col) → pygame Rect on screen."""
        return pygame.Rect(MARGIN + c * CELL, MARGIN + r * CELL, CELL, CELL)

    # ── tile drawing ──────────────────────────────────────────────────────────

    def _tile(self, r: int, c: int, color, key: str = ""):
        """Draw one grid tile — sprite if loaded, filled rect otherwise."""
        rect = self._rect(r, c)
        if key and key in self._sprites:
            self.screen.blit(self._sprites[key], rect)
        else:
            pygame.draw.rect(self.screen, color, rect)

    def _draw_robot(self, r: int, c: int, orientation: int):
        """
        Draw the robot on top of its cell.
        Uses a rotated sprite when sprites/robot.png is present;
        otherwise draws a filled circle with a direction arrow.
        """
        rect   = self._rect(r, c)
        cx, cy = rect.centerx, rect.centery
        radius = CELL // 2 - 3

        if "robot" in self._sprites:
            # sprite assumed to face North by default
            angle = {
                ORIENT_NORTH:  0,
                ORIENT_EAST:  -90,
                ORIENT_SOUTH: 180,
                ORIENT_WEST:   90,
            }[orientation]
            rotated = pygame.transform.rotate(self._sprites["robot"], angle)
            rr = rotated.get_rect(center=(cx, cy))
            self.screen.blit(rotated, rr)
        else:
            # fallback: circle + direction arrow
            pygame.draw.circle(self.screen, ROBOT_COL, (cx, cy), radius)
            dx, dy = {
                ORIENT_NORTH: ( 0, -1),
                ORIENT_EAST:  ( 1,  0),
                ORIENT_SOUTH: ( 0,  1),
                ORIENT_WEST:  (-1,  0),
            }[orientation]
            tip = (cx + dx * (radius - 2), cy + dy * (radius - 2))
            pygame.draw.line(self.screen, ARROW_COL, (cx, cy), tip, 3)
            pygame.draw.circle(self.screen, ARROW_COL, tip, 4)

    # ── grid dividers ─────────────────────────────────────────────────────────

    def _draw_grid_lines(self):
        for i in range(16):
            x = MARGIN + i * CELL
            y = MARGIN + i * CELL
            pygame.draw.line(
                self.screen, GRID_COL,
                (x, MARGIN), (x, MARGIN + 15 * CELL),
            )
            pygame.draw.line(
                self.screen, GRID_COL,
                (MARGIN, y), (MARGIN + 15 * CELL, y),
            )

    # ── info panel ────────────────────────────────────────────────────────────

    def _draw_panel(
        self,
        env: Phase2CleaningEnv,
        episode: int,
        step: int,
        ep_reward: float,
        agent_name: str,
    ):
        """Draw the right-side stats panel with battery bar and episode info."""
        px = MARGIN + 15 * CELL + 8   # left edge of panel content
        pw = INFO_W - 16              # usable width

        # battery bar
        bat_ratio = env.battery / max(env.battery_capacity, 1)
        bar_bg    = pygame.Rect(px, MARGIN + 8, pw, 18)
        bar_fill  = pygame.Rect(px, MARGIN + 8, max(1, int(pw * bat_ratio)), 18)
        bar_color = BAR_OK_COL if bat_ratio > 0.30 else BAR_LOW_COL
        pygame.draw.rect(self.screen, (55, 55, 75), bar_bg)
        pygame.draw.rect(self.screen, bar_color,    bar_fill)

        self.screen.blit(
            self.font_sm.render(
                f"Battery  {env.battery}/{env.battery_capacity}",
                True, TXT_COL,
            ),
            (px, MARGIN + 28),
        )

        # stats text
        lines = [
            "",
            f"Agent    : {agent_name}",
            f"Episode  : {episode}",
            f"Step     : {step}",
            f"Reward   : {ep_reward:+.1f}",
            "",
            f"Dirt left: {int(env.dirt.sum())}",
            f"Facing   : {['N','E','S','W'][env.orientation]}",
            f"Position : {env.robot_pos}",
        ]
        y = MARGIN + 50
        for line in lines:
            if line:
                self.screen.blit(
                    self.font_sm.render(line, True, TXT_COL), (px, y)
                )
            y += 17

        # colour legend
        y += 6
        legend = [
            (WALL_COL,    "Wall"),
            (FURN_COL,    "Furniture"),
            (CHARGER_COL, "Charger"),
            (DIRT_COL,    "Dirt"),
            (ROBOT_COL,   "Robot"),
            (FLOOR_COL,   "Clean floor"),
        ]
        self.screen.blit(
            self.font_sm.render("Legend:", True, TXT_COL), (px, y)
        )
        y += 17
        for color, label in legend:
            pygame.draw.rect(self.screen, color, pygame.Rect(px, y + 1, 13, 13))
            self.screen.blit(
                self.font_sm.render(label, True, TXT_COL), (px + 18, y)
            )
            y += 17

    # ── public API ────────────────────────────────────────────────────────────

    def render(
        self,
        env: Phase2CleaningEnv,
        episode: int = 0,
        step: int = 0,
        ep_reward: float = 0.0,
        agent_name: str = "agent",
        fps: int = 10,
    ) -> bool:
        """
        Draw one frame of the environment.

        Returns False if the user pressed ESC or closed the window
        so the caller can break out of the episode loop cleanly.
        """
        # ── event handling (must be called every frame) ───────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(BG_COL)

        # ── draw all grid cells ───────────────────────────────────────────
        for r in range(env.rows):
            for c in range(env.cols):
                pos = (r, c)

                if pos in env.walls:
                    self._tile(r, c, WALL_COL, "wall")

                elif pos in env.furniture:
                    self._tile(r, c, FURN_COL, "furn")

                elif pos == env.charger_pos:
                    self._tile(r, c, CHARGER_COL, "charger")

                else:
                    # floor tile first
                    self._tile(r, c, FLOOR_COL)
                    # then optionally a dirt dot on top
                    if env.dirt[r, c] == 1:
                        if "dirt" in self._sprites:
                            self.screen.blit(
                                self._sprites["dirt"], self._rect(r, c)
                            )
                        else:
                            cx = MARGIN + c * CELL + CELL // 2
                            cy = MARGIN + r * CELL + CELL // 2
                            pygame.draw.circle(
                                self.screen, DIRT_COL, (cx, cy), 5
                            )

        # ── robot drawn after cells so it appears on top ──────────────────
        rr, rc = env.robot_pos
        self._draw_robot(rr, rc, env.orientation)

        # ── grid lines drawn after tiles, before panel ────────────────────
        self._draw_grid_lines()

        # ── right-side panel ──────────────────────────────────────────────
        panel_x = MARGIN + 15 * CELL
        pygame.draw.rect(
            self.screen, PANEL_COL,
            pygame.Rect(panel_x, 0, INFO_W, WIN_H),
        )
        self._draw_panel(env, episode, step, ep_reward, agent_name)

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        """Shut pygame down cleanly. Call this when training/testing ends."""
        pygame.quit()
