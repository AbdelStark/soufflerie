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
    "collision_kernel",
    "launch_collision",
    "launch_macroscopic",
    "launch_pull_stream",
    "macroscopic_kernel",
    "pull_stream_periodic_kernel",
    "resolve_device",
    "synchronize",
    "upload_state",
]
