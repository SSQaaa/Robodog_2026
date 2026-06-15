# -*- coding: utf-8 -*-
"""
Interactive path planner for task1 cone avoidance.

Coordinate system, unit: millimetres
- x: forward direction in the plan frame, from corridor entrance to exit.
- y: right direction in the plan frame, across the corridor width.
- task1 places this plan frame in ROS world at runtime:
  start_mm maps to the dog start ROS pose, plan +x follows the dog start yaw,
  and each live ROS pose is converted back into this plan frame.
- corridor: x=0..4000, y=0..1500.
- robot: 600 long along x, 400 wide along y.
- start/finish centers come from task1.py START_PLAN_* and FINISH_PLAN_* constants.

Run:
    python 2026Project/task1_path_planner.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[int, int]
GridPoint = Tuple[int, int]
TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent
DEFAULT_PLAN_SVG_PATH = TOOLS_DIR / "task1_path_plan.svg"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from task1 import FINISH_PLAN_X_M, FINISH_PLAN_Y_M, START_PLAN_X_M, START_PLAN_Y_M

TASK1_START_MM = (int(round(START_PLAN_X_M * 1000)), int(round(START_PLAN_Y_M * 1000)))
TASK1_FINISH_MM = (int(round(FINISH_PLAN_X_M * 1000)), int(round(FINISH_PLAN_Y_M * 1000)))


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y

    @property
    def top(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Point:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def inflate(self, dx: int, dy: int) -> "Rect":
        return Rect(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.top <= other.bottom
            or other.top <= self.bottom
        )

    def contains_point(self, point: Point) -> bool:
        px, py = point
        return self.left <= px <= self.right and self.bottom <= py <= self.top


@dataclass
class PlanResult:
    path_mm: List[Point]
    cones_mm: List[Rect]
    corridor_mm: Rect
    robot_mm: Rect
    start_mm: Point
    finish_mm: Point
    grid_step_mm: int
    clearance_mm: int
    svg_path: Optional[Path] = None


class CorridorPlanner:
    def __init__(
        self,
        corridor_l_mm: int = 4000,
        corridor_w_mm: int = 1500,
        robot_l_mm: int = 600,
        robot_w_mm: int = 400,
        cone_w_mm: int = 320,
        cone_l_mm: int = 320,
        grid_step_mm: int = 50,
        clearance_mm: int = 200,
        start_mm: Point = TASK1_START_MM,
        finish_mm: Point = TASK1_FINISH_MM,
    ) -> None:
        self.corridor = Rect(0, 0, corridor_l_mm, corridor_w_mm)
        self.robot = Rect(0, 0, robot_l_mm, robot_w_mm)
        self.cone_size = (cone_l_mm, cone_w_mm)
        self.grid_step = grid_step_mm
        self.clearance = clearance_mm

        half_w = robot_w_mm // 2
        self.start = start_mm
        self.finish = finish_mm

        self.center_min_x = self.start[0]
        self.center_max_x = self.finish[0]
        self.center_min_y = half_w
        self.center_max_y = corridor_w_mm - half_w

        self.cone_center_min_x = cone_l_mm // 2
        self.cone_center_max_x = corridor_l_mm - cone_l_mm // 2
        self.cone_center_min_y = cone_w_mm // 2
        self.cone_center_max_y = corridor_w_mm - cone_w_mm // 2

    def cone_from_center(self, center: Point) -> Rect:
        cone_l, cone_w = self.cone_size
        cx, cy = self.clamp_cone_center(center)
        return Rect(cx - cone_l // 2, cy - cone_w // 2, cone_l, cone_w)

    def cone_keepout_rect(self, cone: Rect) -> Rect:
        return cone.inflate(self.clearance, self.clearance)

    def clamp_cone_center(self, center: Point) -> Point:
        x, y = center
        return (
            clamp(x, self.cone_center_min_x, self.cone_center_max_x),
            clamp(y, self.cone_center_min_y, self.cone_center_max_y),
        )

    def plan_with_cones(self, cones: Sequence[Rect], svg_path: Optional[Path] = None) -> PlanResult:
        self.validate_cones(cones)
        path = self.plan(cones)
        if path is None:
            raise RuntimeError("no valid path found for the specified cone positions")

        result = PlanResult(
            list(path),
            list(cones),
            self.corridor,
            self.robot,
            self.start,
            self.finish,
            self.grid_step,
            self.clearance,
        )
        if svg_path is not None:
            write_svg(result, svg_path)
            result.svg_path = svg_path
        return result

    def validate_cones(self, cones: Sequence[Rect]) -> None:
        for i, cone in enumerate(cones, start=1):
            if (
                cone.left < self.corridor.left
                or cone.right > self.corridor.right
                or cone.bottom < self.corridor.bottom
                or cone.top > self.corridor.top
            ):
                raise ValueError("cone{} is outside the corridor: {}".format(i, cone))

    def plan(self, cones: Sequence[Rect]) -> Optional[List[Point]]:
        start = self.mm_to_grid(self.start)
        goal = self.mm_to_grid(self.finish)
        blocked = self.blocked_cells(cones)

        if start in blocked or goal in blocked:
            return None

        path_grid = astar_min_turns(start, goal, blocked, self.grid_bounds())
        if path_grid is None:
            return None
        return [self.grid_to_mm(point) for point in simplify_grid_path(path_grid)]

    def blocked_cells(self, cones: Sequence[Rect]) -> set[GridPoint]:
        blocked: set[GridPoint] = set()
        inflated = [
            self.cone_keepout_rect(cone).inflate(self.robot.w // 2, self.robot.h // 2)
            for cone in cones
        ]
        min_gx, max_gx, min_gy, max_gy = self.grid_bounds()

        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                point = self.grid_to_mm((gx, gy))
                if any(rect.contains_point(point) for rect in inflated):
                    blocked.add((gx, gy))
        return blocked

    def grid_bounds(self) -> Tuple[int, int, int, int]:
        return (
            math.ceil(self.center_min_x / self.grid_step),
            math.floor(self.center_max_x / self.grid_step),
            math.ceil(self.center_min_y / self.grid_step),
            math.floor(self.center_max_y / self.grid_step),
        )

    def mm_to_grid(self, point: Point) -> GridPoint:
        return (round(point[0] / self.grid_step), round(point[1] / self.grid_step))

    def grid_to_mm(self, point: GridPoint) -> Point:
        return (point[0] * self.grid_step, point[1] * self.grid_step)


def astar_min_turns(
    start: GridPoint,
    goal: GridPoint,
    blocked: set[GridPoint],
    bounds: Tuple[int, int, int, int],
) -> Optional[List[GridPoint]]:
    min_gx, max_gx, min_gy, max_gy = bounds
    start_state = (start, None)
    open_heap: List[Tuple[int, int, int, GridPoint, Optional[GridPoint]]] = []
    heapq.heappush(open_heap, (0, 0, manhattan(start, goal), start, None))

    came_from: Dict[Tuple[GridPoint, Optional[GridPoint]], Tuple[GridPoint, Optional[GridPoint]]] = {}
    best_cost: Dict[Tuple[GridPoint, Optional[GridPoint]], Tuple[int, int]] = {start_state: (0, 0)}
    best_goal_state: Optional[Tuple[GridPoint, Optional[GridPoint]]] = None

    while open_heap:
        turns, steps, _, current, direction = heapq.heappop(open_heap)
        state = (current, direction)
        if (turns, steps) != best_cost[current, direction]:
            continue
        if current == goal:
            best_goal_state = state
            break

        for nxt, next_direction in neighbours_4_with_direction(current):
            gx, gy = nxt
            if gx < min_gx or gx > max_gx or gy < min_gy or gy > max_gy or nxt in blocked:
                continue
            next_turns = turns
            if direction is not None and next_direction != direction:
                next_turns += 1
            next_steps = steps + 1
            next_state = (nxt, next_direction)
            next_cost = (next_turns, next_steps)
            if next_cost >= best_cost.get(next_state, (10**9, 10**9)):
                continue
            came_from[next_state] = state
            best_cost[next_state] = next_cost
            heapq.heappush(open_heap, (next_turns, next_steps, manhattan(nxt, goal), nxt, next_direction))

    if best_goal_state is None:
        return None
    return reconstruct_state_path(came_from, best_goal_state)


def neighbours_4(point: GridPoint) -> Iterable[GridPoint]:
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def neighbours_4_with_direction(point: GridPoint) -> Iterable[Tuple[GridPoint, GridPoint]]:
    x, y = point
    yield (x + 1, y), (1, 0)
    yield (x - 1, y), (-1, 0)
    yield (x, y + 1), (0, 1)
    yield (x, y - 1), (0, -1)


def manhattan(a: GridPoint, b: GridPoint) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[GridPoint, GridPoint], current: GridPoint) -> List[GridPoint]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def reconstruct_state_path(
    came_from: Dict[Tuple[GridPoint, Optional[GridPoint]], Tuple[GridPoint, Optional[GridPoint]]],
    current_state: Tuple[GridPoint, Optional[GridPoint]],
) -> List[GridPoint]:
    states = [current_state]
    while current_state in came_from:
        current_state = came_from[current_state]
        states.append(current_state)
    states.reverse()
    return [state[0] for state in states]


def simplify_grid_path(path: Sequence[GridPoint]) -> List[GridPoint]:
    if len(path) <= 2:
        return list(path)

    simplified = [path[0]]
    prev_dx = path[1][0] - path[0][0]
    prev_dy = path[1][1] - path[0][1]

    for i in range(1, len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        if (dx, dy) != (prev_dx, prev_dy):
            simplified.append(path[i])
            prev_dx, prev_dy = dx, dy
    simplified.append(path[-1])
    return simplified


class CanvasMapper:
    def __init__(self, planner: CorridorPlanner, scale: float = 0.16, margin: int = 90) -> None:
        self.planner = planner
        self.scale = scale
        self.margin = margin
        self.min_x = planner.start[0]
        self.max_x = planner.finish[0]
        self.min_y = 0
        self.max_y = planner.corridor.h
        self.width = int((self.max_x - self.min_x) * scale + margin * 2)
        self.height = int((self.max_y - self.min_y) * scale + margin * 2)

    def to_px(self, point: Point) -> Tuple[int, int]:
        x, y = point
        px = self.margin + int(round((x - self.min_x) * self.scale))
        py = self.margin + int(round((y - self.min_y) * self.scale))
        return px, py

    def to_mm(self, px: int, py: int) -> Point:
        x = int(round((px - self.margin) / self.scale + self.min_x))
        y = int(round((py - self.margin) / self.scale + self.min_y))
        return x, y

    def rect_to_px(self, rect: Rect) -> Tuple[int, int, int, int]:
        x1, y1 = self.to_px((rect.left, rect.bottom))
        x2, y2 = self.to_px((rect.right, rect.top))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_scene(
    planner: CorridorPlanner,
    cones: Sequence[Rect],
    result: Optional[PlanResult],
    mapper: CanvasMapper,
    message: str = "",
    preview_cone: Optional[Rect] = None,
) -> "object":
    import cv2
    import numpy as np

    img = np.full((mapper.height, mapper.width, 3), 248, dtype=np.uint8)

    draw_rect(img, mapper, planner.corridor, (255, 255, 255), (30, 30, 30), thickness=2)
    draw_grid(img, mapper)
    draw_rect(img, mapper, planner.corridor, None, (30, 30, 30), thickness=2)

    for cone in cones:
        keepout = planner.cone_keepout_rect(cone)
        draw_rect(img, mapper, keepout, (210, 225, 255), (80, 100, 210), thickness=1)
    for index, cone in enumerate(cones, start=1):
        draw_rect(img, mapper, cone, (0, 130, 255), (0, 70, 150), thickness=2)
        cx, cy = mapper.to_px(cone.center)
        cv2.putText(img, "cone{}".format(index), (cx - 26, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

    if preview_cone is not None:
        preview_keepout = planner.cone_keepout_rect(preview_cone)
        draw_rect(img, mapper, preview_keepout, (235, 240, 255), (120, 140, 220), thickness=1)
        draw_rect(img, mapper, preview_cone, (80, 190, 255), (0, 100, 180), thickness=2)
        cx, cy = mapper.to_px(preview_cone.center)
        cv2.circle(img, (cx, cy), 3, (0, 0, 180), -1)

    if result is not None:
        draw_polyline(img, mapper, result.path_mm, (216, 102, 11), 5)
        start_robot = robot_rect_at(result.start_mm, planner.robot)
        finish_robot = robot_rect_at(result.finish_mm, planner.robot)
        draw_rect(img, mapper, start_robot, (125, 225, 160), (20, 120, 65), thickness=2)
        draw_rect(img, mapper, finish_robot, (190, 190, 245), (70, 60, 180), thickness=2)

    sx, sy = mapper.to_px(planner.start)
    fx, fy = mapper.to_px(planner.finish)
    cv2.circle(img, (sx, sy), 6, (45, 150, 55), -1)
    cv2.circle(img, (fx, fy), 6, (45, 60, 200), -1)
    cv2.putText(img, "start", (sx - 25, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 120, 45), 1)
    cv2.putText(img, "finish", (fx - 28, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 60, 180), 1)

    cv2.putText(img, "left click: cone center   r: reset   s: save svg   q/esc: quit", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    if message:
        cv2.putText(img, message, (18, mapper.height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 180), 2)
    return img


def draw_grid(img: "object", mapper: CanvasMapper) -> None:
    import cv2

    for x in range(mapper.min_x, mapper.max_x + 1, 100):
        p1 = mapper.to_px((x, mapper.min_y))
        p2 = mapper.to_px((x, mapper.max_y))
        color = (225, 225, 225)
        thickness = 1
        if x % 500 == 0:
            color = (180, 180, 180)
            thickness = 1
        cv2.line(img, p1, p2, color, thickness)
        if x % 500 == 0:
            label_x, label_y = mapper.to_px((x, mapper.min_y))
            cv2.putText(img, str(x), (label_x - 18, label_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

    for y in range(mapper.min_y, mapper.max_y + 1, 100):
        p1 = mapper.to_px((mapper.min_x, y))
        p2 = mapper.to_px((mapper.max_x, y))
        color = (225, 225, 225)
        thickness = 1
        if y % 500 == 0:
            color = (180, 180, 180)
            thickness = 1
        cv2.line(img, p1, p2, color, thickness)
        if y % 500 == 0:
            label_x, label_y = mapper.to_px((mapper.min_x, y))
            cv2.putText(img, str(y), (label_x - 58, label_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)


def draw_rect(
    img: "object",
    mapper: CanvasMapper,
    rect: Rect,
    fill: Optional[Tuple[int, int, int]],
    stroke: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    import cv2

    left, top, right, bottom = mapper.rect_to_px(rect)
    if fill is not None:
        cv2.rectangle(img, (left, top), (right, bottom), fill, -1)
    cv2.rectangle(img, (left, top), (right, bottom), stroke, thickness)


def draw_polyline(img: "object", mapper: CanvasMapper, path: Sequence[Point], color: Tuple[int, int, int], thickness: int) -> None:
    import cv2

    for p1, p2 in zip(path, path[1:]):
        cv2.line(img, mapper.to_px(p1), mapper.to_px(p2), color, thickness)


def robot_rect_at(center: Point, robot: Rect) -> Rect:
    cx, cy = center
    return Rect(cx - robot.w // 2, cy - robot.h // 2, robot.w, robot.h)


def run_interactive(planner: CorridorPlanner, out_path: Path) -> None:
    import cv2

    window_name = "task1_path_planner"
    mapper = CanvasMapper(planner)
    cones: List[Rect] = []
    result: Optional[PlanResult] = None
    preview_cone: Optional[Rect] = None
    message = "click cone1 center"

    def refresh() -> None:
        cv2.imshow(window_name, draw_scene(planner, cones, result, mapper, message, preview_cone))

    def mouse_callback(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal result, message, preview_cone
        if event == cv2.EVENT_MOUSEMOVE:
            if len(cones) < 2:
                preview_cone = planner.cone_from_center(mapper.to_mm(x, y))
                refresh()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(cones) >= 2:
            preview_cone = None
            message = "already have two cones, press r to reset"
            refresh()
            return

        center = planner.clamp_cone_center(mapper.to_mm(x, y))
        cone = planner.cone_from_center(center)
        cones.append(cone)
        preview_cone = None
        if len(cones) == 1:
            message = "click cone2 center"
        else:
            try:
                result = planner.plan_with_cones(cones)
                message = "path ready, press s to save"
                print_plan(result)
            except RuntimeError:
                result = None
                message = "no path, press r and choose again"
        refresh()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    refresh()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            cones = []
            result = None
            preview_cone = None
            message = "click cone1 center"
            refresh()
        elif key == ord("s"):
            if result is None:
                message = "no path to save"
            else:
                saved_svg_path = write_plan_files(result, out_path)
                result.svg_path = saved_svg_path
                message = "saved {}".format(saved_svg_path.resolve())
                print_plan(result)
            refresh()

    cv2.destroyWindow(window_name)


def write_svg(result: PlanResult, output_path: Path) -> None:
    mapper = CanvasMapper(
        CorridorPlanner(
            corridor_l_mm=result.corridor_mm.w,
            corridor_w_mm=result.corridor_mm.h,
            robot_l_mm=result.robot_mm.w,
            robot_w_mm=result.robot_mm.h,
            grid_step_mm=result.grid_step_mm,
            clearance_mm=result.clearance_mm,
        ),
        scale=0.16,
        margin=90,
    )

    def rect_svg(rect: Rect, fill: str, stroke: str, opacity: float = 1.0) -> str:
        left, top, right, bottom = mapper.rect_to_px(rect)
        return (
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" '
            'stroke="{}" stroke-width="2" opacity="{:.2f}" />'
        ).format(left, top, right - left, bottom - top, fill, stroke, opacity)

    grid_lines: List[str] = []
    for x in range(mapper.min_x, mapper.max_x + 1, 100):
        p1 = mapper.to_px((x, mapper.min_y))
        p2 = mapper.to_px((x, mapper.max_y))
        color = "#b8b8b8" if x % 500 == 0 else "#e4e4e4"
        grid_lines.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1" />'.format(p1[0], p1[1], p2[0], p2[1], color))
    for y in range(mapper.min_y, mapper.max_y + 1, 100):
        p1 = mapper.to_px((mapper.min_x, y))
        p2 = mapper.to_px((mapper.max_x, y))
        color = "#b8b8b8" if y % 500 == 0 else "#e4e4e4"
        grid_lines.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1" />'.format(p1[0], p1[1], p2[0], p2[1], color))

    robot_half_l = result.robot_mm.w // 2
    robot_half_w = result.robot_mm.h // 2
    inflated = [cone.inflate(result.clearance_mm, result.clearance_mm) for cone in result.cones_mm]
    path_points = " ".join("{},{}".format(*mapper.to_px(point)) for point in result.path_mm)
    start_robot = robot_rect_at(result.start_mm, result.robot_mm)
    finish_robot = robot_rect_at(result.finish_mm, result.robot_mm)

    labels = []
    for i, cone in enumerate(result.cones_mm, start=1):
        cx, cy = mapper.to_px(cone.center)
        labels.append('<text x="{}" y="{}" font-size="14" text-anchor="middle" fill="#111">cone{}</text>'.format(cx, cy + 5, i))

    sx, sy = mapper.to_px(result.start_mm)
    fx, fy = mapper.to_px(result.finish_mm)

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f8f8" />
  <text x="{title_x}" y="30" font-size="18" text-anchor="middle" fill="#111">Task1 cone path plan, unit: mm</text>
  {grid}
  {corridor}
  {inflated}
  {cones}
  <polyline points="{path_points}" fill="none" stroke="#0b66d8" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" />
  {start_robot}
  {finish_robot}
  <circle cx="{sx}" cy="{sy}" r="6" fill="#159947" />
  <circle cx="{fx}" cy="{fy}" r="6" fill="#d13b3b" />
  <text x="{sx}" y="{start_label_y}" font-size="14" text-anchor="middle" fill="#159947">start</text>
  <text x="{fx}" y="{finish_label_y}" font-size="14" text-anchor="middle" fill="#d13b3b">finish</text>
  {labels}
</svg>
""".format(
        width=mapper.width,
        height=mapper.height,
        title_x=mapper.width / 2,
        grid="\n  ".join(grid_lines),
        corridor=rect_svg(result.corridor_mm, "#ffffff", "#20242a"),
        inflated="\n  ".join(rect_svg(rect, "#f3b3ad", "#d13b3b", 0.28) for rect in inflated),
        cones="\n  ".join(rect_svg(cone, "#ff7a1a", "#923b00") for cone in result.cones_mm),
        path_points=path_points,
        start_robot=rect_svg(start_robot, "#32b36b", "#126c3d", 0.25),
        finish_robot=rect_svg(finish_robot, "#d13b3b", "#8d2020", 0.18),
        sx=sx,
        sy=sy,
        fx=fx,
        fy=fy,
        start_label_y=sy - 12,
        finish_label_y=fy - 12,
        labels="\n  ".join(labels),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def json_path_for_svg(svg_path: Path) -> Path:
    return svg_path.with_suffix(".json")


def resolve_output_svg_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    parts = path.parts
    if len(parts) >= 2 and parts[0] == "2026Project" and parts[1] == "tools":
        return TOOLS_DIR / parts[-1]

    return path.resolve()


def write_plan_json(result: PlanResult, output_path: Path) -> None:
    data = {
        "unit": "mm",
        "coordinate_system": {
            "frame": "task1_plan_frame",
            "x": "forward in plan frame; mapped to dog start yaw in ROS world by task1.py",
            "y": "right in plan frame; mapped to the right side of dog start yaw by task1.py",
            "start_finish_source": "2026Project/task1.py START_PLAN_* and FINISH_PLAN_* constants",
            "runtime_anchor": "start_mm is placed at the dog ROS pose when task1.py starts",
            "runtime_execution": "task1.py converts every live ROS pose into this plan frame, then moves and corrects using plan-coordinate error only",
            "corridor": {
                "x_min": result.corridor_mm.left,
                "x_max": result.corridor_mm.right,
                "y_min": result.corridor_mm.bottom,
                "y_max": result.corridor_mm.top,
            },
        },
        "robot_mm": {
            "length_x": result.robot_mm.w,
            "width_y": result.robot_mm.h,
        },
        "start_mm": list(result.start_mm),
        "finish_mm": list(result.finish_mm),
        "cones_mm": [
            {
                "center": list(cone.center),
                "rect": {
                    "x": cone.x,
                    "y": cone.y,
                    "w": cone.w,
                    "h": cone.h,
                },
            }
            for cone in result.cones_mm
        ],
        "waypoints_mm": [list(point) for point in result.path_mm],
        "grid_step_mm": result.grid_step_mm,
        "clearance_mm": result.clearance_mm,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_plan_files(result: PlanResult, svg_path: Path) -> Path:
    svg_path = resolve_output_svg_path(svg_path)
    write_svg(result, svg_path)
    write_plan_json(result, json_path_for_svg(svg_path))
    return svg_path


def print_plan(result: PlanResult) -> None:
    print("cone_centers_mm:")
    for i, cone in enumerate(result.cones_mm, start=1):
        cx, cy = cone.center
        print("  cone{}: ({}, {})".format(i, cx, cy))
    print("path_waypoints_mm:")
    for x, y in result.path_mm:
        print("  ({}, {})".format(x, y))
    if result.svg_path is not None:
        print("svg: {}".format(result.svg_path.resolve()))
        print("json: {}".format(json_path_for_svg(result.svg_path).resolve()))


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive cone path planner and SVG visualizer.")
    parser.add_argument("--cone1", type=int, nargs=2, metavar=("X", "Y"), help="cone 1 center position in mm")
    parser.add_argument("--cone2", type=int, nargs=2, metavar=("X", "Y"), help="cone 2 center position in mm")
    parser.add_argument("--out", type=Path, default=DEFAULT_PLAN_SVG_PATH, help="SVG output path")
    parser.add_argument("--grid", type=int, default=50, help="planner grid step in millimetres")
    parser.add_argument(
        "--clearance",
        type=int,
        default=200,
        help="extra cone clearance in millimetres; keepout side = cone side + 2 * clearance",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    planner = CorridorPlanner(grid_step_mm=args.grid, clearance_mm=args.clearance)

    if args.cone1 is not None and args.cone2 is not None:
        cones = [planner.cone_from_center(tuple(args.cone1)), planner.cone_from_center(tuple(args.cone2))]
        result = planner.plan_with_cones(cones)
        saved_svg_path = write_plan_files(result, args.out)
        result.svg_path = saved_svg_path
        print_plan(result)
        return

    run_interactive(planner, args.out)


if __name__ == "__main__":
    main()
