"""3-D wireframe renderer using Pygame.

Renders aircraft, WEZ cones, flight trails, a ground grid, and a HUD overlay.
Camera is an orbit camera controlled with mouse and keyboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    pygame = None  # type: ignore[assignment]
    gfxdraw = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (60, 60, 60)
DARK_GREY = (30, 30, 30)
LIGHT_GREY = (120, 120, 120)
BLUE = (50, 130, 255)
BLUE_DIM = (30, 70, 140)
RED = (255, 60, 60)
RED_DIM = (140, 30, 30)
GREEN = (60, 220, 80)
CYAN = (0, 220, 220)
YELLOW = (255, 220, 50)
BLUE_WEZ = (50, 130, 255, 60)
RED_WEZ = (255, 60, 60, 60)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AircraftState:
    """Snapshot of aircraft state for the renderer."""
    x: float
    y: float
    h: float
    v: float
    psi: float       # heading [rad]
    gamma: float     # flight-path angle [rad]
    label: str = "Aircraft"
    colour: tuple = BLUE
    colour_dim: tuple = BLUE_DIM
    wez_colour: tuple = BLUE_WEZ
    wez_aperture_deg: float = 20.0
    wez_height_m: float = 300.0
    trail: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Spherical-coordinate orbit camera with pan."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.azimuth = math.radians(-45)   # horizontal angle
        self.elevation = math.radians(35)  # vertical angle
        self.distance = 3500.0             # zoom distance
        self.target = np.array([0.0, 0.0, 500.0])  # look-at point
        self._clamp()

    def rotate(self, daz: float, del_: float) -> None:
        self.azimuth += daz
        self.elevation += del_
        self._clamp()

    def zoom(self, delta: float) -> None:
        self.distance *= 1.0 - delta * 0.1
        self.distance = max(200.0, min(self.distance, 30000.0))

    def pan(self, dx: float, dy: float) -> None:
        # Pan in the camera's local right/up plane
        right = np.array([math.cos(self.azimuth + math.pi / 2), math.sin(self.azimuth + math.pi / 2), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        speed = self.distance * 0.002
        self.target += right * dx * speed + up * (-dy) * speed

    def _clamp(self) -> None:
        self.elevation = max(math.radians(5), min(math.radians(85), self.elevation))

    @property
    def eye(self) -> np.ndarray:
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        offset = np.array([ce * ca, ce * sa, se]) * self.distance
        return self.target + offset

    def view_matrix(self) -> np.ndarray:
        """4x4 look-at view matrix."""
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        up = np.cross(right, forward)

        m = np.eye(4)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[0, 3] = -np.dot(right, self.eye)
        m[1, 3] = -np.dot(up, self.eye)
        m[2, 3] = np.dot(forward, self.eye)
        return m


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer3D:
    """Interactive 3-D wireframe renderer.

    Controls:
        Left-drag   → orbit (rotate camera)
        Right-drag  → pan
        Scroll      → zoom
        R           → reset camera
    """

    WIDTH = 1280
    HEIGHT = 800
    FPS = 10

    def __init__(self) -> None:
        if pygame is None:
            raise ImportError(
                "Renderer3D requires pygame. Install with: pip install pygame"
            )
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Air Combat Gym")
        self.clock = pygame.time.Clock()
        self.camera = OrbitCamera()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_large = pygame.font.SysFont("consolas", 18, bold=True)
        self._running = True

        # Perspective projection params
        self.fov = 60.0
        self.near = 1.0
        self.far = 50000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(self, aircraft_states: Sequence[AircraftState], step: int = 0) -> bool:
        """Draw one frame. Returns False if the window was closed."""
        if not self._running:
            return False

        self._handle_events()
        if not self._running:
            return False

        self.screen.fill(DARK_GREY)
        view = self.camera.view_matrix()
        proj = self._perspective_matrix()
        vp = proj @ view

        self._draw_ground_grid(vp)
        self._draw_axes(vp)

        for ac in aircraft_states:
            self._draw_trail(vp, ac)
            self._draw_aircraft(vp, ac)
            self._draw_wez_cone(vp, ac)

        self._draw_hud(aircraft_states, step)

        pygame.display.flip()
        self.clock.tick(self.FPS)
        return True

    def close(self) -> None:
        if self._running:
            self._running = False
            pygame.quit()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:   # scroll up
                    self.camera.zoom(1)
                elif event.button == 5:  # scroll down
                    self.camera.zoom(-1)
            elif event.type == pygame.MOUSEMOTION:
                btns = pygame.mouse.get_pressed()
                dx, dy = event.rel
                if btns[0]:   # left drag → orbit
                    self.camera.rotate(-dx * 0.005, dy * 0.005)
                elif btns[2]:  # right drag → pan
                    self.camera.pan(dx, dy)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.camera.reset()
                elif event.key == pygame.K_ESCAPE:
                    self._running = False

    # ------------------------------------------------------------------
    # 3-D projection
    # ------------------------------------------------------------------

    def _perspective_matrix(self) -> np.ndarray:
        aspect = self.WIDTH / self.HEIGHT
        f = 1.0 / math.tan(math.radians(self.fov / 2))
        n, fa = self.near, self.far
        m = np.zeros((4, 4))
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (fa + n) / (n - fa)
        m[2, 3] = 2 * fa * n / (n - fa)
        m[3, 2] = -1.0
        return m

    def _project(self, vp: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
        """Project a 3-D world point to 2-D screen coords. Returns None if behind camera."""
        p = np.array([point[0], point[1], point[2], 1.0])
        clip = vp @ p
        if clip[3] <= 0:
            return None
        ndc = clip[:3] / clip[3]
        if abs(ndc[0]) > 1.5 or abs(ndc[1]) > 1.5:
            return None  # off-screen, cull
        sx = int((ndc[0] + 1) * 0.5 * self.WIDTH)
        sy = int((1 - ndc[1]) * 0.5 * self.HEIGHT)  # flip Y
        return sx, sy

    def _project_batch(self, vp: np.ndarray, points: np.ndarray) -> list[tuple[int, int] | None]:
        """Project an Nx3 array of points."""
        n = points.shape[0]
        ones = np.ones((n, 1))
        homo = np.hstack([points, ones])  # Nx4
        clip = (vp @ homo.T).T            # Nx4
        result = []
        for i in range(n):
            if clip[i, 3] <= 0:
                result.append(None)
            else:
                ndc = clip[i, :3] / clip[i, 3]
                if abs(ndc[0]) > 1.5 or abs(ndc[1]) > 1.5:
                    result.append(None)
                else:
                    sx = int((ndc[0] + 1) * 0.5 * self.WIDTH)
                    sy = int((1 - ndc[1]) * 0.5 * self.HEIGHT)
                    result.append((sx, sy))
        return result

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_line_3d(self, vp: np.ndarray, a: np.ndarray, b: np.ndarray, colour: tuple) -> None:
        pa, pb = self._project(vp, a), self._project(vp, b)
        if pa and pb:
            pygame.draw.line(self.screen, colour, pa, pb, 1)

    def _draw_ground_grid(self, vp: np.ndarray) -> None:
        """Draw a reference grid on z=0 plane."""
        spacing = 1000
        extent = 10000
        for i in range(-extent, extent + 1, spacing):
            c = GREY if i != 0 else LIGHT_GREY
            self._draw_line_3d(vp, np.array([i, -extent, 0.0]), np.array([i, extent, 0.0]), c)
            self._draw_line_3d(vp, np.array([-extent, i, 0.0]), np.array([extent, i, 0.0]), c)

        # Labels at grid edges
        for v in range(-extent, extent + 1, spacing * 2):
            p = self._project(vp, np.array([v, -extent - 50, 0.0]))
            if p:
                lbl = self.font.render(f"{v}", True, LIGHT_GREY)
                self.screen.blit(lbl, p)

    def _draw_axes(self, vp: np.ndarray) -> None:
        """Draw X/Y/Z axes at origin."""
        length = 400
        self._draw_line_3d(vp, np.zeros(3), np.array([length, 0, 0.0]), RED)
        self._draw_line_3d(vp, np.zeros(3), np.array([0, length, 0.0]), GREEN)
        self._draw_line_3d(vp, np.zeros(3), np.array([0, 0, length]), CYAN)

        for axis, label, colour in [
            (np.array([length + 40, 0, 0.0]), "X", RED),
            (np.array([0, length + 40, 0.0]), "Y", GREEN),
            (np.array([0, 0, length + 40]), "Z", CYAN),
        ]:
            p = self._project(vp, axis)
            if p:
                lbl = self.font.render(label, True, colour)
                self.screen.blit(lbl, p)

    def _draw_aircraft(self, vp: np.ndarray, ac: AircraftState) -> None:
        """Draw an aircraft as a 3-D oriented arrow."""
        pos = np.array([ac.x, ac.y, ac.h])
        cp, sp = math.cos(ac.psi), math.sin(ac.psi)
        cg, sg = math.cos(ac.gamma), math.sin(ac.gamma)

        # Forward direction (heading vector)
        fwd = np.array([cp * cg, sp * cg, sg])
        # Right vector (perpendicular in horizontal plane)
        right = np.array([-sp, cp, 0.0])
        # Up vector
        up = np.cross(right, fwd)

        size = 60.0  # arrow half-length
        nose = pos + fwd * size * 1.5
        tail = pos - fwd * size
        left_wing = pos - fwd * size * 0.3 - right * size * 0.8
        right_wing = pos - fwd * size * 0.3 + right * size * 0.8
        fin_tip = pos - fwd * size * 0.5 + up * size * 0.5

        edges = [
            (nose, left_wing), (nose, right_wing),
            (left_wing, tail), (right_wing, tail),
            (left_wing, right_wing),
            (tail, fin_tip), (fin_tip, pos),
        ]
        for a, b in edges:
            self._draw_line_3d(vp, a, b, ac.colour)

        # Label
        lbl_pos = self._project(vp, pos + up * size)
        if lbl_pos:
            txt = self.font.render(ac.label, True, ac.colour)
            self.screen.blit(txt, (lbl_pos[0] - txt.get_width() // 2, lbl_pos[1] - 20))

        # Shadow on ground
        ground = np.array([ac.x, ac.y, 0.0])
        gp = self._project(vp, ground)
        if gp:
            pygame.draw.circle(self.screen, ac.colour_dim, gp, 4)
        # Altitude line
        self._draw_line_3d(vp, ground, pos, ac.colour_dim)

    def _draw_wez_cone(self, vp: np.ndarray, ac: AircraftState) -> None:
        """Draw the WEZ as a wireframe cone."""
        pos = np.array([ac.x, ac.y, ac.h])
        cp, sp = math.cos(ac.psi), math.sin(ac.psi)
        cg, sg = math.cos(ac.gamma), math.sin(ac.gamma)

        fwd = np.array([cp * cg, sp * cg, sg])
        right = np.array([-sp, cp, 0.0])
        rn = np.linalg.norm(right)
        if rn > 1e-9:
            right /= rn
        up = np.cross(right, fwd)
        un = np.linalg.norm(up)
        if un > 1e-9:
            up /= un

        half_angle = math.radians(ac.wez_aperture_deg / 2)
        height = ac.wez_height_m
        radius = height * math.tan(half_angle)
        tip = pos + fwd * height

        # Generate cone ring points
        n_segments = 16
        ring_points = []
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            offset = right * math.cos(angle) * radius + up * math.sin(angle) * radius
            ring_points.append(tip + offset)

        # Determine colour based on aircraft
        cone_col = ac.colour_dim

        # Draw cone lines from apex to ring
        for i in range(0, n_segments, 2):
            self._draw_line_3d(vp, pos, ring_points[i], cone_col)

        # Draw ring
        for i in range(n_segments):
            self._draw_line_3d(vp, ring_points[i], ring_points[(i + 1) % n_segments], cone_col)

    def _draw_trail(self, vp: np.ndarray, ac: AircraftState) -> None:
        """Draw the flight path trail."""
        if len(ac.trail) < 2:
            return
        # Downsample for performance — keep every Nth point
        step = max(1, len(ac.trail) // 200)
        pts = ac.trail[::step]
        if len(pts) < 2:
            return

        points = np.array(pts)
        projected = self._project_batch(vp, points)

        for i in range(len(projected) - 1):
            p0, p1 = projected[i], projected[i + 1]
            if p0 and p1:
                # Fade trail: older = dimmer
                t = i / max(len(projected) - 1, 1)
                c = tuple(int(ac.colour_dim[j] + (ac.colour[j] - ac.colour_dim[j]) * t) for j in range(3))
                pygame.draw.line(self.screen, c, p0, p1, 1)

    def _draw_hud(self, aircraft_states: Sequence[AircraftState], step: int) -> None:
        """Draw a heads-up display overlay."""
        x, y = 15, 12
        lh = 18  # line height

        # Title
        title = self.font_large.render("AIR COMBAT GYM", True, WHITE)
        self.screen.blit(title, (x, y))
        y += lh + 6

        # Step / FPS
        fps = self.clock.get_fps()
        info = self.font.render(f"Step: {step:>5d}   FPS: {fps:4.0f}", True, LIGHT_GREY)
        self.screen.blit(info, (x, y))
        y += lh + 8

        # Aircraft info
        for ac in aircraft_states:
            header = self.font_large.render(f"[ {ac.label} ]", True, ac.colour)
            self.screen.blit(header, (x, y))
            y += lh

            psi_deg = math.degrees(ac.psi) % 360
            gamma_deg = math.degrees(ac.gamma)
            lines = [
                f"Pos   x={ac.x:>8.1f}  y={ac.y:>8.1f}  h={ac.h:>7.1f}",
                f"Speed V={ac.v:>7.1f} m/s",
                f"Hdg  Psi={psi_deg:>6.1f}   Gam={gamma_deg:>6.1f}",
            ]
            for line in lines:
                txt = self.font.render(line, True, WHITE)
                self.screen.blit(txt, (x + 8, y))
                y += lh
            y += 6

        # Controls hint at bottom
        hint = self.font.render("LMB:Rotate  RMB:Pan  Scroll:Zoom  R:Reset  ESC:Quit", True, GREY)
        self.screen.blit(hint, (x, self.HEIGHT - 28))
