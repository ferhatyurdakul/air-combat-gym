from __future__ import annotations

import numpy as np


class Aircraft:
    """Simple 3-DoF point-mass aircraft model.

    State vector: (x, y, h, V, ψ, γ)
        x, y  — horizontal position [m]
        h     — altitude [m]
        V     — true airspeed magnitude [m/s]
        ψ     — heading angle [rad], measured from +x axis, positive toward +y
        γ     — flight-path angle [rad], positive nose-up

    Equations of motion (load-factor form)::

        ẋ = V cos ψ cos γ
        ẏ = V sin ψ cos γ
        ḣ = V sin γ
        V̇ = g (nₓ − sin γ)
        ψ̇ = g nz sin μ / (V cos γ)
        γ̇ = (g / V)(nz cos μ − cos γ)

    where g = 9.81 m/s² and the control inputs are defined below.
    """

    def __init__(
        self,
        x: float,
        y: float,
        h: float,
        v: float,
        psi: float,
        gamma: float,
        constant_speed: bool = False,
        wez_aperture_deg: float = 20.0,
        wez_height_m: float = 300.0,
    ):
        """Initialise an aircraft.

        Args:
            x: Initial x-position [m].
            y: Initial y-position [m].
            h: Initial altitude [m].
            v: Initial true airspeed [m/s].
            psi: Initial heading angle [rad].
            gamma: Initial flight-path angle [rad].
            constant_speed: If True, velocity is held constant (v_dot ignored).
            wez_aperture_deg: Default WEZ half-cone full aperture [deg].
            wez_height_m: Default WEZ cone slant height [m].
        """
        # Positions
        self.x = x
        self.y = y
        self.h = h

        # Velocity magnitude
        self.v = v

        # Angles
        self.psi = psi  # heading [rad]
        self.gamma = gamma  # flight-path angle [rad]

        # Constants
        self.dt = 0.01
        self.g = 9.81

        # Limits
        self.constant_speed = constant_speed
        self.max_speed_limit = 600
        self.min_speed_limit = 10
        self.psi_limit = 2 * np.pi
        self.gamma_limit = np.pi / 4

        # WEZ parameters
        self.wez_aperture_deg = wez_aperture_deg
        self.wez_height_m = wez_height_m

        # Log
        self.log: dict[str, list[float]] = {
            "x": [x],
            "y": [y],
            "h": [h],
            "v": [v],
            "psi": [psi],
            "gamma": [gamma],
        }

    def update(self, nx: float, nz: float, mu: float) -> None:
        """Integrate the equations of motion for one environment step (1 s).

        Uses forward-Euler integration with the internal time-step ``self.dt``.

        Args:
            nx: Tangential (excess-thrust) load factor — dimensionless,
                defined as (T − D) / (m g).  Positive ⇒ accelerating.
            nz: Normal load factor — dimensionless, defined as L / (m g).
                In steady level flight nz = 1 / cos(μ).
            mu: Bank (velocity-roll) angle [rad].  Positive ⇒ right wing down.
        """
        for _ in np.arange(0, 1, self.dt):
            x_dot = self.v * np.cos(self.psi) * np.cos(self.gamma)
            y_dot = self.v * np.sin(self.psi) * np.cos(self.gamma)
            h_dot = self.v * np.sin(self.gamma)

            v_dot = self.g * (nx - np.sin(self.gamma))
            psi_dot = (self.g * nz / self.v) * (np.sin(mu) / np.cos(self.gamma))
            gamma_dot = (self.g / self.v) * (nz * np.cos(mu) - np.cos(self.gamma))

            self.x += x_dot * self.dt
            self.y += y_dot * self.dt
            self.h += h_dot * self.dt

            if not self.constant_speed:
                self.v += v_dot * self.dt

            self.psi += psi_dot * self.dt
            self.gamma += gamma_dot * self.dt

            # Clamp
            self.v = float(np.clip(self.v, self.min_speed_limit, self.max_speed_limit))
            self.psi %= self.psi_limit
            self.gamma = float(np.clip(self.gamma, -self.gamma_limit, self.gamma_limit))

            # Logs
            self.log["x"].append(self.x)
            self.log["y"].append(self.y)
            self.log["h"].append(self.h)
            self.log["v"].append(self.v)
            self.log["psi"].append(self.psi)
            self.log["gamma"].append(self.gamma)

    def wez(
        self,
        px: float,
        py: float,
        ph: float,
        aperture: float | None = None,
        height: float | None = None,
    ) -> bool:
        """Weapon Engagement Zone (WEZ) cone test.

        Returns True if the point (px, py, ph) falls inside the aircraft's
        forward-facing engagement cone defined by an aperture angle and slant
        height.

        Args:
            px: Target x-position [m].
            py: Target y-position [m].
            ph: Target altitude [m].
            aperture: Full cone aperture [deg].  Defaults to ``self.wez_aperture_deg``.
            height: Cone slant height [m].  Defaults to ``self.wez_height_m``.
        """
        if aperture is None:
            aperture = self.wez_aperture_deg
        if height is None:
            height = self.wez_height_m
        dx = px - self.x
        dy = py - self.y
        dh = ph - self.h

        point_distance = float(np.sqrt(dx**2 + dy**2 + dh**2))
        if point_distance <= 1e-9:
            return True

        heading_vector = np.array(
            [
                np.cos(self.psi) * np.cos(self.gamma),
                np.sin(self.psi) * np.cos(self.gamma),
                np.sin(self.gamma),
            ]
        )
        point_vector = np.array([dx, dy, dh]) / point_distance

        cosine_angle = float(np.clip(np.dot(heading_vector, point_vector), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine_angle)))

        cone_distance = height / np.cos(np.radians(angle))
        return (angle <= aperture / 2) and (point_distance <= cone_distance)
