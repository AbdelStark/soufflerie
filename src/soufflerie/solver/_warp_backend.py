# mypy: ignore-errors
"""Warp implementation detail loaded only by :class:`WarpKernelAdapter`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp  # type: ignore[import-untyped]

from soufflerie.solver.kernels import WarpArray


@wp.func
def _feq(
    weight: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    rho: wp.float32,
    u: wp.float32,
    v: wp.float32,
) -> wp.float32:
    cu = cx * u + cy * v
    return weight * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (u * u + v * v))


@wp.func
def _direction_x(direction: int) -> int:
    if direction == 1 or direction == 5 or direction == 8:
        return 1
    if direction == 3 or direction == 6 or direction == 7:
        return -1
    return 0


@wp.func
def _direction_y(direction: int) -> int:
    if direction == 2 or direction == 5 or direction == 6:
        return 1
    if direction == 4 or direction == 7 or direction == 8:
        return -1
    return 0


@wp.func
def _opposite(direction: int) -> int:
    if direction == 1:
        return 3
    if direction == 2:
        return 4
    if direction == 3:
        return 1
    if direction == 4:
        return 2
    if direction == 5:
        return 7
    if direction == 6:
        return 8
    if direction == 7:
        return 5
    if direction == 8:
        return 6
    return 0


@wp.func
def _pull_channel(
    post_collision: wp.array3d(dtype=wp.float32),
    y: int,
    x: int,
    direction: int,
    nx: int,
    ny: int,
) -> wp.float32:
    source_x = x - _direction_x(direction)
    source_y = y - _direction_y(direction)
    if source_y <= 0 or source_y >= ny - 1:
        return post_collision[y, x, _opposite(direction)]
    if source_x < 0 or source_x >= nx:
        return post_collision[y, x, direction]
    return post_collision[source_y, source_x, direction]


@wp.func
def _regularized_population(
    weight: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    equilibrium: wp.float32,
    pi_xx: wp.float32,
    pi_xy: wp.float32,
    pi_yy: wp.float32,
) -> wp.float32:
    contraction = (
        (cx * cx - 1.0 / 3.0) * pi_xx + 2.0 * cx * cy * pi_xy + (cy * cy - 1.0 / 3.0) * pi_yy
    )
    return equilibrium + 4.5 * weight * contraction


@wp.kernel
def collision_kernel(
    populations: wp.array3d(dtype=wp.float32),
    omega: wp.float32,
    post_collision: wp.array3d(dtype=wp.float32),
):
    y, x = wp.tid()
    f0 = populations[y, x, 0]
    f1 = populations[y, x, 1]
    f2 = populations[y, x, 2]
    f3 = populations[y, x, 3]
    f4 = populations[y, x, 4]
    f5 = populations[y, x, 5]
    f6 = populations[y, x, 6]
    f7 = populations[y, x, 7]
    f8 = populations[y, x, 8]
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    u = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    v = (f2 - f4 + f5 + f6 - f7 - f8) / rho

    post_collision[y, x, 0] = f0 - omega * (f0 - _feq(4.0 / 9.0, 0.0, 0.0, rho, u, v))
    post_collision[y, x, 1] = f1 - omega * (f1 - _feq(1.0 / 9.0, 1.0, 0.0, rho, u, v))
    post_collision[y, x, 2] = f2 - omega * (f2 - _feq(1.0 / 9.0, 0.0, 1.0, rho, u, v))
    post_collision[y, x, 3] = f3 - omega * (f3 - _feq(1.0 / 9.0, -1.0, 0.0, rho, u, v))
    post_collision[y, x, 4] = f4 - omega * (f4 - _feq(1.0 / 9.0, 0.0, -1.0, rho, u, v))
    post_collision[y, x, 5] = f5 - omega * (f5 - _feq(1.0 / 36.0, 1.0, 1.0, rho, u, v))
    post_collision[y, x, 6] = f6 - omega * (f6 - _feq(1.0 / 36.0, -1.0, 1.0, rho, u, v))
    post_collision[y, x, 7] = f7 - omega * (f7 - _feq(1.0 / 36.0, -1.0, -1.0, rho, u, v))
    post_collision[y, x, 8] = f8 - omega * (f8 - _feq(1.0 / 36.0, 1.0, -1.0, rho, u, v))


@wp.kernel
def pull_stream_periodic_kernel(
    post_collision: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    streamed: wp.array3d(dtype=wp.float32),
):
    y, x, direction = wp.tid()
    source_x = (x - _direction_x(direction) + nx) % nx
    source_y = (y - _direction_y(direction) + ny) % ny
    streamed[y, x, direction] = post_collision[source_y, source_x, direction]


@wp.kernel
def sponge_kernel(
    post_collision: wp.array3d(dtype=wp.float32),
    inlet_velocity_lu: wp.float32,
    sponge_start_x: int,
    sponge_columns: int,
):
    y, x, direction = wp.tid()
    ny = post_collision.shape[0]
    if y > 0 and y < ny - 1 and x >= sponge_start_x:
        fraction = wp.float32(x - sponge_start_x) / wp.float32(sponge_columns - 1)
        strength = 0.15 * fraction * fraction
        cx = wp.float32(_direction_x(direction))
        cy = wp.float32(_direction_y(direction))
        weight = wp.float32(1.0 / 36.0)
        if direction == 0:
            weight = wp.float32(4.0 / 9.0)
        elif direction == 1 or direction == 2 or direction == 3 or direction == 4:
            weight = wp.float32(1.0 / 9.0)
        target = _feq(weight, cx, cy, 1.0, inlet_velocity_lu, 0.0)
        current = post_collision[y, x, direction]
        post_collision[y, x, direction] = current + strength * (target - current)


@wp.kernel
def channel_boundaries_kernel(
    post_collision: wp.array3d(dtype=wp.float32),
    inlet_velocity_lu: wp.float32,
    nx: int,
    ny: int,
    streamed: wp.array3d(dtype=wp.float32),
):
    y, x = wp.tid()
    f0 = post_collision[y, x, 0]
    f1 = post_collision[y, x, 1]
    f2 = post_collision[y, x, 2]
    f3 = post_collision[y, x, 3]
    f4 = post_collision[y, x, 4]
    f5 = post_collision[y, x, 5]
    f6 = post_collision[y, x, 6]
    f7 = post_collision[y, x, 7]
    f8 = post_collision[y, x, 8]

    if y > 0 and y < ny - 1:
        f0 = _pull_channel(post_collision, y, x, 0, nx, ny)
        f1 = _pull_channel(post_collision, y, x, 1, nx, ny)
        f2 = _pull_channel(post_collision, y, x, 2, nx, ny)
        f3 = _pull_channel(post_collision, y, x, 3, nx, ny)
        f4 = _pull_channel(post_collision, y, x, 4, nx, ny)
        f5 = _pull_channel(post_collision, y, x, 5, nx, ny)
        f6 = _pull_channel(post_collision, y, x, 6, nx, ny)
        f7 = _pull_channel(post_collision, y, x, 7, nx, ny)
        f8 = _pull_channel(post_collision, y, x, 8, nx, ny)

        if x == 0:
            rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - inlet_velocity_lu)
            f1 = f3 + (2.0 / 3.0) * rho * inlet_velocity_lu
            provisional_f5 = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * inlet_velocity_lu
            provisional_f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * inlet_velocity_lu
            if y > 1 and y < ny - 2:
                f5 = provisional_f5
                f8 = provisional_f8
                e0 = _feq(4.0 / 9.0, 0.0, 0.0, rho, inlet_velocity_lu, 0.0)
                e1 = _feq(1.0 / 9.0, 1.0, 0.0, rho, inlet_velocity_lu, 0.0)
                e2 = _feq(1.0 / 9.0, 0.0, 1.0, rho, inlet_velocity_lu, 0.0)
                e3 = _feq(1.0 / 9.0, -1.0, 0.0, rho, inlet_velocity_lu, 0.0)
                e4 = _feq(1.0 / 9.0, 0.0, -1.0, rho, inlet_velocity_lu, 0.0)
                e5 = _feq(1.0 / 36.0, 1.0, 1.0, rho, inlet_velocity_lu, 0.0)
                e6 = _feq(1.0 / 36.0, -1.0, 1.0, rho, inlet_velocity_lu, 0.0)
                e7 = _feq(1.0 / 36.0, -1.0, -1.0, rho, inlet_velocity_lu, 0.0)
                e8 = _feq(1.0 / 36.0, 1.0, -1.0, rho, inlet_velocity_lu, 0.0)
                n1 = f1 - e1
                n2 = f2 - e2
                n3 = f3 - e3
                n4 = f4 - e4
                n5 = f5 - e5
                n6 = f6 - e6
                n7 = f7 - e7
                n8 = f8 - e8
                pi_xx = n1 + n3 + n5 + n6 + n7 + n8
                pi_xy = n5 - n6 + n7 - n8
                pi_yy = n2 + n4 + n5 + n6 + n7 + n8
                f0 = _regularized_population(4.0 / 9.0, 0.0, 0.0, e0, pi_xx, pi_xy, pi_yy)
                f1 = _regularized_population(1.0 / 9.0, 1.0, 0.0, e1, pi_xx, pi_xy, pi_yy)
                f2 = _regularized_population(1.0 / 9.0, 0.0, 1.0, e2, pi_xx, pi_xy, pi_yy)
                f3 = _regularized_population(1.0 / 9.0, -1.0, 0.0, e3, pi_xx, pi_xy, pi_yy)
                f4 = _regularized_population(1.0 / 9.0, 0.0, -1.0, e4, pi_xx, pi_xy, pi_yy)
                f5 = _regularized_population(1.0 / 36.0, 1.0, 1.0, e5, pi_xx, pi_xy, pi_yy)
                f6 = _regularized_population(1.0 / 36.0, -1.0, 1.0, e6, pi_xx, pi_xy, pi_yy)
                f7 = _regularized_population(1.0 / 36.0, -1.0, -1.0, e7, pi_xx, pi_xy, pi_yy)
                f8 = _regularized_population(1.0 / 36.0, 1.0, -1.0, e8, pi_xx, pi_xy, pi_yy)
            else:
                if y != 1:
                    f5 = provisional_f5
                if y != ny - 2:
                    f8 = provisional_f8

        elif x == nx - 1:
            neighbor_x = nx - 2
            g0 = _pull_channel(post_collision, y, neighbor_x, 0, nx, ny)
            g1 = _pull_channel(post_collision, y, neighbor_x, 1, nx, ny)
            g2 = _pull_channel(post_collision, y, neighbor_x, 2, nx, ny)
            g3 = _pull_channel(post_collision, y, neighbor_x, 3, nx, ny)
            g4 = _pull_channel(post_collision, y, neighbor_x, 4, nx, ny)
            g5 = _pull_channel(post_collision, y, neighbor_x, 5, nx, ny)
            g6 = _pull_channel(post_collision, y, neighbor_x, 6, nx, ny)
            g7 = _pull_channel(post_collision, y, neighbor_x, 7, nx, ny)
            g8 = _pull_channel(post_collision, y, neighbor_x, 8, nx, ny)
            neighbor_rho = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8
            neighbor_u = (g1 - g3 + g5 - g6 - g7 + g8) / neighbor_rho
            neighbor_v = (g2 - g4 + g5 + g6 - g7 - g8) / neighbor_rho
            e3 = _feq(1.0 / 9.0, -1.0, 0.0, neighbor_rho, neighbor_u, neighbor_v)
            f3 = e3 + (g3 - e3)
            if y != 1:
                e6 = _feq(1.0 / 36.0, -1.0, 1.0, neighbor_rho, neighbor_u, neighbor_v)
                f6 = e6 + (g6 - e6)
            if y != ny - 2:
                e7 = _feq(1.0 / 36.0, -1.0, -1.0, neighbor_rho, neighbor_u, neighbor_v)
                f7 = e7 + (g7 - e7)

    streamed[y, x, 0] = f0
    streamed[y, x, 1] = f1
    streamed[y, x, 2] = f2
    streamed[y, x, 3] = f3
    streamed[y, x, 4] = f4
    streamed[y, x, 5] = f5
    streamed[y, x, 6] = f6
    streamed[y, x, 7] = f7
    streamed[y, x, 8] = f8


@wp.kernel
def macroscopic_kernel(
    populations: wp.array3d(dtype=wp.float32),
    rho_out: wp.array2d(dtype=wp.float32),
    velocity_out: wp.array3d(dtype=wp.float32),
):
    y, x = wp.tid()
    f0 = populations[y, x, 0]
    f1 = populations[y, x, 1]
    f2 = populations[y, x, 2]
    f3 = populations[y, x, 3]
    f4 = populations[y, x, 4]
    f5 = populations[y, x, 5]
    f6 = populations[y, x, 6]
    f7 = populations[y, x, 7]
    f8 = populations[y, x, 8]
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    rho_out[y, x] = rho
    velocity_out[y, x, 0] = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    velocity_out[y, x, 1] = (f2 - f4 + f5 + f6 - f7 - f8) / rho


def resolve_device(requested: str) -> str:
    wp.init()
    return str(wp.get_device(requested))


def allocate_state(
    shape: tuple[int, int], device: str
) -> tuple[WarpArray, WarpArray, WarpArray, WarpArray]:
    ny, nx = shape
    return (
        wp.zeros((ny, nx, 9), dtype=wp.float32, device=device),
        wp.zeros((ny, nx, 9), dtype=wp.float32, device=device),
        wp.zeros((ny, nx), dtype=wp.float32, device=device),
        wp.zeros((ny, nx, 2), dtype=wp.float32, device=device),
    )


def upload_state(
    populations: npt.NDArray[np.float32],
    rho: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    device: str,
) -> tuple[WarpArray, WarpArray, WarpArray, WarpArray]:
    return (
        wp.array(populations, dtype=wp.float32, device=device),
        wp.zeros(populations.shape, dtype=wp.float32, device=device),
        wp.array(rho, dtype=wp.float32, device=device),
        wp.array(velocity, dtype=wp.float32, device=device),
    )


def launch_collision(
    populations: WarpArray,
    post_collision: WarpArray,
    omega: float,
    shape: tuple[int, int],
    device: str,
) -> None:
    wp.launch(
        collision_kernel,
        dim=shape,
        inputs=[populations, wp.float32(omega)],
        outputs=[post_collision],
        device=device,
        record_tape=False,
    )
    wp.synchronize_device(device)


def launch_pull_stream(
    post_collision: WarpArray,
    streamed: WarpArray,
    shape: tuple[int, int],
    device: str,
) -> None:
    ny, nx = shape
    wp.launch(
        pull_stream_periodic_kernel,
        dim=(ny, nx, 9),
        inputs=[post_collision, nx, ny],
        outputs=[streamed],
        device=device,
        record_tape=False,
    )
    wp.synchronize_device(device)


def launch_sponge(
    post_collision: WarpArray,
    inlet_velocity_lu: float,
    sponge_start_x: int,
    sponge_columns: int,
    shape: tuple[int, int],
    device: str,
) -> None:
    ny, nx = shape
    wp.launch(
        sponge_kernel,
        dim=(ny, nx, 9),
        inputs=[
            post_collision,
            wp.float32(inlet_velocity_lu),
            sponge_start_x,
            sponge_columns,
        ],
        device=device,
        record_tape=False,
    )
    wp.synchronize_device(device)


def launch_channel_boundaries(
    post_collision: WarpArray,
    streamed: WarpArray,
    inlet_velocity_lu: float,
    shape: tuple[int, int],
    device: str,
) -> None:
    ny, nx = shape
    wp.launch(
        channel_boundaries_kernel,
        dim=(ny, nx),
        inputs=[post_collision, wp.float32(inlet_velocity_lu), nx, ny],
        outputs=[streamed],
        device=device,
        record_tape=False,
    )
    wp.synchronize_device(device)


def launch_macroscopic(
    populations: WarpArray,
    rho: WarpArray,
    velocity: WarpArray,
    shape: tuple[int, int],
    device: str,
) -> None:
    wp.launch(
        macroscopic_kernel,
        dim=shape,
        inputs=[populations],
        outputs=[rho, velocity],
        device=device,
        record_tape=False,
    )
    wp.synchronize_device(device)


def synchronize(device: str) -> None:
    wp.synchronize_device(device)


def array_is_float32(array: WarpArray) -> bool:
    return array.dtype == wp.float32


__all__ = [
    "allocate_state",
    "array_is_float32",
    "channel_boundaries_kernel",
    "collision_kernel",
    "launch_channel_boundaries",
    "launch_collision",
    "launch_macroscopic",
    "launch_pull_stream",
    "launch_sponge",
    "macroscopic_kernel",
    "pull_stream_periodic_kernel",
    "resolve_device",
    "sponge_kernel",
    "synchronize",
    "upload_state",
]
