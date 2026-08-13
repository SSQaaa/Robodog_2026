# -*- coding: utf-8 -*-
"""
Task1 cone path planner.

The search can use horizontal/vertical motion and the measured diagonal
motion.  The first and last segment stay horizontal/vertical so the
robot can align with the start and finish points in the usual way.

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
    python tools/task1_path_planner.py
    input x and y for the left cone, then x and y for the right cone,
    all in millimetres in the task1 plan frame.
"""

from __future__ import annotations

import ctypes
import heapq
import json
import math
import sys
import time
from collections import defaultdict, deque
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

CONF_THRESH = 0.5
MIN_VALID_DEPTH_COUNT = 10
DEPTH_HISTORY_LEN = 5
DETECTION_FRAMES = 15
TARGET_CLASS = 7

DEPTH_FX = 478.547
DEPTH_FY = 478.547
DEPTH_CX = 321.087
DEPTH_CY = 201.625
REAL_CONE_WIDTH_M = 0.32
COLOR_FX = 453.72

CAM_ON_ROBOT_X_M = 0.2
CAM_ON_ROBOT_Y_M = 0.0

# 锥桶四周的默认硬安全距离，单位：毫米。
CONE_CLEARANCE_MM = 200

# Measured with dog.move(vx=20000, vy=25000): one second moves about
# 400 mm in plan x and 150 mm in plan y. Diagonal endpoints are rounded to
# the nearest 50 mm planning cell; final point alignment handles that small
# discretisation error.
GRID_STEP_MM = 50
DIAGONAL_TIME_STEP_S = 0.2
DIAGONAL_MIN_TIME_S = 1.0
FORWARD_SPEED_MM_S = 500.0
LATERAL_SPEED_MM_S = 250.0
DIAGONAL_X_SPEED_MM_S = 400.0
DIAGONAL_Y_SPEED_MM_S = 150.0
TURN_PENALTY_S = 0.15
AXIS_STEP_CELLS = 1
HEURISTIC_WEIGHT = 1.0
SOFT_CLEARANCE_MM = 600
SOFT_CLEARANCE_MAX_RATE = 0.60


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

    def contains_point(self, point: Point) -> bool:
        px, py = point
        return self.left < px < self.right and self.bottom < py < self.top


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
        grid_step_mm: int = GRID_STEP_MM,
        clearance_mm: int = CONE_CLEARANCE_MM,
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

        initial_clearance = max(0, self.clearance)
        clearance_candidates = list(range(initial_clearance, -1, -10))
        if clearance_candidates[-1] != 0:
            clearance_candidates.append(0)

        path = None
        for clearance_mm in clearance_candidates:
            self.clearance = clearance_mm
            path = self.plan(cones)
            if path is not None:
                break

        if path is None:
            raise RuntimeError(
                "no valid path found for the specified cone positions, even with 0 mm clearance"
            )

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

        clearance = clearance_from_blocked(blocked, self.grid_bounds())
        transition_x = self.between_cones_transition_x(cones)
        path_grid = astar_measured_diagonal(
            start,
            goal,
            blocked,
            self.grid_bounds(),
            clearance,
            transition_x,
        )
        if path_grid is None:
            return None
        return [self.grid_to_mm(point) for point in simplify_grid_path(path_grid)]

    def between_cones_transition_x(self, cones: Sequence[Rect]) -> Optional[int]:
        """Return the grid x for lateral motion between the two cones.

        Put the transition two thirds of the way from the smaller-x cone to
        the larger-x cone when that position is collision-free.  If the
        requested position enters either cone's robot-centre keepout area,
        clamp it to the nearest feasible grid column between the keepouts.
        This keeps it as close as safely possible to the front cone while
        leaving room for lateral-motion rear drift.
        """
        if len(cones) != 2:
            return None
        rear_cone, front_cone = sorted(cones, key=lambda cone: cone.center[0])
        rear_x = rear_cone.center[0]
        front_x = front_cone.center[0]
        transition_x_mm = rear_x + (2.0 / 3.0) * (front_x - rear_x)

        robot_half_length = self.robot.w // 2
        rear_keepout_right = self.cone_keepout_rect(rear_cone).right + robot_half_length
        front_keepout_left = self.cone_keepout_rect(front_cone).left - robot_half_length
        min_grid_x = math.ceil(rear_keepout_right / self.grid_step)
        max_grid_x = math.floor(front_keepout_left / self.grid_step)
        if min_grid_x > max_grid_x:
            # There is no grid column between the two hard keepouts at the
            # current clearance.  Returning an unreachable transition keeps
            # this attempt from silently moving laterally somewhere else;
            # plan_with_cones will retry with a smaller clearance.
            return max_grid_x
        desired_grid_x = round(transition_x_mm / self.grid_step)
        return clamp(desired_grid_x, min_grid_x, max_grid_x)

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


def astar_measured_diagonal(
    start: GridPoint,
    goal: GridPoint,
    blocked: set[GridPoint],
    bounds: Tuple[int, int, int, int],
    clearance_cells: Optional[Dict[GridPoint, float]] = None,
    transition_x: Optional[int] = None,
) -> Optional[List[GridPoint]]:
    """Find a short collision-free path using the measured diagonal step.

    Cost is estimated movement time plus a small turn and proximity penalty.  This makes a
    diagonal useful for avoiding an obstacle while moving forward, instead of
    creating unnecessary diagonal zigzags, and avoids slow pure-Y movement.
    """
    min_gx, max_gx, min_gy, max_gy = bounds
    start_state = (start, None)
    open_heap: List[Tuple[float, float, int, int, GridPoint, Optional[GridPoint]]] = []
    push_order = 0
    heapq.heappush(open_heap, (HEURISTIC_WEIGHT * time_heuristic(start, goal), 0.0, 0, push_order, start, None))

    came_from: Dict[Tuple[GridPoint, Optional[GridPoint]], Tuple[GridPoint, Optional[GridPoint]]] = {}
    best_cost: Dict[Tuple[GridPoint, Optional[GridPoint]], Tuple[float, int]] = {start_state: (0.0, 0)}
    best_goal_state: Optional[Tuple[GridPoint, Optional[GridPoint]]] = None

    while open_heap:
        _, distance_cost, turns, _, current, direction = heapq.heappop(open_heap)
        state = (current, direction)
        if (distance_cost, turns) != best_cost[state]:
            continue
        if current == goal:
            best_goal_state = state
            break

        for nxt, next_direction, is_diagonal in motion_neighbours(current):
            gx, gy = nxt
            if gx < min_gx or gx > max_gx or gy < min_gy or gy > max_gy:
                continue
            # Start/finish alignment remains horizontal or vertical.  The
            # diagonal is used only for the obstacle-avoidance part between
            # those alignment legs.
            if is_diagonal and (current == start or nxt == goal):
                continue
            # Keep the main lateral transition between the cones at the
            # requested x.  Pure lateral alignment remains allowed at the
            # start and finish x positions.
            is_lateral = nxt[0] == current[0] and nxt[1] != current[1]
            if (
                is_lateral
                and transition_x is not None
                and current[0] not in (start[0], transition_x, goal[0])
            ):
                continue
            if not segment_is_clear(current, nxt, blocked):
                continue

            next_turns = turns
            if direction is not None and next_direction != direction:
                next_turns += 1
            step_time = motion_time_s(current, nxt, is_diagonal)
            turn_cost = TURN_PENALTY_S if direction is not None and next_direction != direction else 0.0
            proximity_cost = segment_clearance_penalty_s(current, nxt, step_time, clearance_cells)
            next_distance_cost = distance_cost + step_time + turn_cost + proximity_cost
            next_state = (nxt, next_direction)
            next_cost = (next_distance_cost, next_turns)
            if next_cost >= best_cost.get(next_state, (float("inf"), 10**9)):
                continue
            came_from[next_state] = state
            best_cost[next_state] = next_cost
            push_order += 1
            estimated_total = next_distance_cost + HEURISTIC_WEIGHT * time_heuristic(nxt, goal)
            heapq.heappush(
                open_heap,
                (estimated_total, next_distance_cost, next_turns, push_order, nxt, next_direction),
            )

    if best_goal_state is None:
        return None
    return reconstruct_state_path(came_from, best_goal_state)


def clearance_from_blocked(
    blocked: set[GridPoint], bounds: Tuple[int, int, int, int]
) -> Dict[GridPoint, float]:
    """Return Euclidean clearance to the nearest hard keepout cell."""
    if not blocked:
        return {}
    min_gx, max_gx, min_gy, max_gy = bounds
    result: Dict[GridPoint, float] = {}
    for gx in range(min_gx, max_gx + 1):
        for gy in range(min_gy, max_gy + 1):
            point = (gx, gy)
            if point in blocked:
                result[point] = 0.0
                continue
            result[point] = min(math.hypot(gx - bx, gy - by) for bx, by in blocked)
    return result


def segment_clearance_penalty_s(
    start: GridPoint,
    finish: GridPoint,
    motion_time: float,
    clearance_cells: Optional[Dict[GridPoint, float]],
) -> float:
    """Integrate a small proximity cost along one complete motion primitive."""
    if not clearance_cells:
        return 0.0
    dx = finish[0] - start[0]
    dy = finish[1] - start[1]
    sample_count = max(abs(dx), abs(dy), 1)
    risk_sum = 0.0
    for i in range(sample_count + 1):
        ratio = float(i) / sample_count
        point = (round(start[0] + dx * ratio), round(start[1] + dy * ratio))
        clearance_mm = clearance_cells.get(point, float("inf")) * GRID_STEP_MM
        proximity = max(0.0, (SOFT_CLEARANCE_MM - clearance_mm) / SOFT_CLEARANCE_MM)
        risk_sum += proximity * proximity
    mean_risk = risk_sum / (sample_count + 1)
    return motion_time * SOFT_CLEARANCE_MAX_RATE * mean_risk


def motion_neighbours(point: GridPoint) -> Iterable[Tuple[GridPoint, GridPoint, bool]]:
    x, y = point
    # Axis alignment uses the normal 50 mm planning resolution.
    for dx, dy in (
        (AXIS_STEP_CELLS, 0),
        (0, AXIS_STEP_CELLS),
        (0, -AXIS_STEP_CELLS),
    ):
        yield (x + dx, y + dy), (dx, dy), False

    for index in range(5):
        move_time = DIAGONAL_MIN_TIME_S + index * DIAGONAL_TIME_STEP_S
        step_x = int(round(DIAGONAL_X_SPEED_MM_S * move_time / GRID_STEP_MM))
        step_y = int(round(DIAGONAL_Y_SPEED_MM_S * move_time / GRID_STEP_MM))
        for sign_x, sign_y in ((1, 1), (1, -1)):
            dx = sign_x * step_x
            dy = sign_y * step_y
            # All scaled primitives share the same canonical direction, so
            # simplify_grid_path merges them into one diagonal leg.
            direction = (sign_x, sign_y)
            yield (x + dx, y + dy), direction, True


def motion_time_s(start: GridPoint, finish: GridPoint, is_diagonal: bool) -> float:
    dx_mm = abs(finish[0] - start[0]) * GRID_STEP_MM
    dy_mm = abs(finish[1] - start[1]) * GRID_STEP_MM
    if is_diagonal:
        return dx_mm / DIAGONAL_X_SPEED_MM_S
    if dx_mm:
        return dx_mm / FORWARD_SPEED_MM_S
    return dy_mm / LATERAL_SPEED_MM_S


def time_heuristic(current: GridPoint, goal: GridPoint) -> float:
    """Optimistic remaining time, ignoring obstacles and turn overhead."""
    dx_mm = max(goal[0] - current[0], 0) * GRID_STEP_MM
    dy_mm = abs(goal[1] - current[1]) * GRID_STEP_MM
    return max(dx_mm / FORWARD_SPEED_MM_S, dy_mm / LATERAL_SPEED_MM_S)


def segment_is_clear(start: GridPoint, finish: GridPoint, blocked: set[GridPoint]) -> bool:
    """Check the whole motion primitive, preventing diagonal corner cutting."""
    dx = finish[0] - start[0]
    dy = finish[1] - start[1]
    sample_count = max(abs(dx), abs(dy)) * 4
    for i in range(sample_count + 1):
        ratio = float(i) / sample_count if sample_count else 0.0
        cell = (round(start[0] + dx * ratio), round(start[1] + dy * ratio))
        if cell in blocked:
            return False
    return True


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
    prev_direction = normalized_grid_direction(path[0], path[1])

    for i in range(1, len(path) - 1):
        direction = normalized_grid_direction(path[i], path[i + 1])
        if direction != prev_direction:
            simplified.append(path[i])
            prev_direction = direction
    simplified.append(path[-1])
    return simplified


def normalized_grid_direction(start: GridPoint, finish: GridPoint) -> GridPoint:
    dx = finish[0] - start[0]
    dy = finish[1] - start[1]
    if dx and dy:
        return (1 if dx > 0 else -1, 1 if dy > 0 else -1)
    divisor = math.gcd(abs(dx), abs(dy))
    return (dx // divisor, dy // divisor)


class CanvasMapper:
    def __init__(self, planner: CorridorPlanner, scale: float = 0.16, margin: int = 90) -> None:
        self.planner = planner
        self.scale = scale
        self.margin = margin
        robot_half_l = planner.robot.w // 2
        self.min_x = planner.start[0] - robot_half_l
        self.max_x = planner.finish[0] + robot_half_l
        self.min_y = 0
        self.max_y = planner.corridor.h
        self.width = int((self.max_x - self.min_x) * scale + margin * 2)
        self.height = int((self.max_y - self.min_y) * scale + margin * 2)

    def to_px(self, point: Point) -> Tuple[int, int]:
        x, y = point
        px = self.margin + int(round((x - self.min_x) * self.scale))
        py = self.margin + int(round((y - self.min_y) * self.scale))
        return px, py

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

    cv2.putText(img, "Task1 path plan, unit: mm   q/esc: close", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
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


def scale_box(box: Tuple[int, int, int, int], src_size: Point, dst_size: Point) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    dx1 = clamp(int(round(x1 * dst_w / float(src_w))), 0, dst_w - 1)
    dx2 = clamp(int(round(x2 * dst_w / float(src_w))), 0, dst_w - 1)
    dy1 = clamp(int(round(y1 * dst_h / float(src_h))), 0, dst_h - 1)
    dy2 = clamp(int(round(y2 * dst_h / float(src_h))), 0, dst_h - 1)
    return dx1, dy1, dx2, dy2


def yolo_to_original(box: Sequence[float], img_w: int, img_h: int, input_size: int = 640) -> Tuple[int, int, int, int]:
    cx, cy, w, h = box
    scale = min(input_size / img_w, input_size / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    pad_x = (input_size - new_w) / 2
    pad_y = (input_size - new_h) / 2
    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    w = w / scale
    h = h / scale
    x1 = clamp(int(cx - w / 2), 0, img_w - 1)
    x2 = clamp(int(cx + w / 2), 0, img_w - 1)
    y1 = clamp(int(cy - h / 2), 0, img_h - 1)
    y2 = clamp(int(cy + h / 2), 0, img_h - 1)
    return x1, y1, x2, y2


def pixel_to_camera_3d(u: int, v: int, depth_mm: int) -> Tuple[float, float, float]:
    z = depth_mm / 1000.0
    x = (u - DEPTH_CX) * z / DEPTH_FX
    y = (v - DEPTH_CY) * z / DEPTH_FY
    return x, y, z


def get_cone_depth_roi(color_box: Tuple[int, int, int, int], color_size: Point, depth_size: Point) -> Optional[Tuple[int, int, int, int]]:
    dx1, dy1, dx2, dy2 = scale_box(color_box, color_size, depth_size)
    h = dy2 - dy1
    w = dx2 - dx1
    if h <= 0 or w <= 0:
        return None

    target_h = max(5, h // 4)
    roi_y2 = dy2
    roi_y1 = max(0, dy2 - target_h)
    if roi_y2 - roi_y1 < 5:
        roi_y1 = dy1
        roi_y2 = min(depth_size[1], dy1 + target_h)

    target_w = max(5, w // 2)
    center_x = (dx1 + dx2) // 2
    roi_x1 = max(0, center_x - target_w // 2)
    roi_x2 = min(depth_size[0], roi_x1 + target_w)
    if roi_x2 - roi_x1 < 5:
        roi_x1, roi_x2 = dx1, dx2

    return roi_x1, roi_y1, roi_x2, roi_y2


def load_detector():
    libs_dir = PROJECT_DIR / "libs"
    if not libs_dir.exists():
        libs_dir = Path("/home/ysc/Desktop/2026Project/libs")

    engine_path = libs_dir / "bigdog_0427.engine"
    sys.path.append(str(libs_dir))
    ctypes.CDLL(str(libs_dir / "libmyplugins.so"))
    import yolov5_trt_cpp

    print("[TRT] loading engine...")
    detector = yolov5_trt_cpp.Yolov5TRT(str(engine_path))
    print("[TRT] engine loaded")
    return detector


def detect_cone_y_mm(expected_count: int = 2) -> List[int]:
    import numpy as np
    import orbbec_native

    detector = load_detector()
    cam = orbbec_native.OrbbecCamera()
    cam.start()
    try:
        time.sleep(1.0)
        depth_w, depth_h = cam.get_depth_size()
        color_w, color_h = cam.get_color_size()
        print("[Orbbec] color {}x{}, depth {}x{}".format(color_w, color_h, depth_w, depth_h))

        depth_history = defaultdict(lambda: deque(maxlen=DEPTH_HISTORY_LEN))
        position_estimates: Dict[int, Tuple[float, float]] = {}
        last_centers: Dict[int, Point] = {}

        print("[Detect] collecting {} frames, keep the robot still...".format(DETECTION_FRAMES))
        for frame_idx in range(DETECTION_FRAMES):
            color_frame = cam.get_color_frame()
            if color_frame is None:
                time.sleep(0.01)
                continue
            frame = np.asarray(color_frame, dtype=np.uint8).copy()

            for det in detector.detect(frame):
                cx_y, cy_y, w_y, h_y, conf, cls_id = det
                if conf < CONF_THRESH or int(cls_id) != TARGET_CLASS:
                    continue

                color_box = yolo_to_original((cx_y, cy_y, w_y, h_y), color_w, color_h)
                box_center = ((color_box[0] + color_box[2]) // 2, (color_box[1] + color_box[3]) // 2)
                tid = None
                for existing_tid, last_center in last_centers.items():
                    if abs(box_center[0] - last_center[0]) < 50 and abs(box_center[1] - last_center[1]) < 50:
                        tid = existing_tid
                        break
                if tid is None:
                    tid = len(last_centers)
                last_centers[tid] = box_center

                depth_roi = get_cone_depth_roi(color_box, (color_w, color_h), (depth_w, depth_h))
                if depth_roi is None:
                    continue

                raw_depth, valid_cnt = cam.get_depth_in_box(*depth_roi)
                x1, _, x2, _ = color_box
                box_w = x2 - x1
                visual_z = (REAL_CONE_WIDTH_M * COLOR_FX) / max(box_w, 1) if box_w > 0 else 5.0

                u = (depth_roi[0] + depth_roi[2]) // 2
                v = depth_roi[3] - 1
                if raw_depth > 0 and valid_cnt >= MIN_VALID_DEPTH_COUNT:
                    depth_history[tid].append(raw_depth)
                    x_cam, _, z_cam = pixel_to_camera_3d(u, v, int(np.median(depth_history[tid])))
                else:
                    if visual_z <= 0:
                        continue
                    x_cam = (u - DEPTH_CX) * visual_z / DEPTH_FX
                    z_cam = visual_z

                alpha = 0.5
                if tid in position_estimates:
                    prev_x, prev_z = position_estimates[tid]
                    position_estimates[tid] = (alpha * x_cam + (1 - alpha) * prev_x, alpha * z_cam + (1 - alpha) * prev_z)
                else:
                    position_estimates[tid] = (x_cam, z_cam)

            print("\r[Detect] frame {}/{} ids={}".format(frame_idx + 1, DETECTION_FRAMES, list(position_estimates.keys())), end="")
            time.sleep(0.05)
        print()
    finally:
        cam.stop()

    if len(position_estimates) < expected_count:
        raise RuntimeError("detected {} cones, expected {}".format(len(position_estimates), expected_count))

    y_values = []
    for tid, (x_cam, z_cam) in position_estimates.items():
        plan_y = TASK1_START_MM[1] - int(round((CAM_ON_ROBOT_Y_M + x_cam) * 1000))
        plan_x_from_camera = TASK1_START_MM[0] + int(round((CAM_ON_ROBOT_X_M + z_cam) * 1000))
        print("camera cone{}: x_from_camera={} y={} mm".format(tid, plan_x_from_camera, plan_y))
        y_values.append(plan_y)

    return sorted(y_values)[:expected_count]


def input_int_mm(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        try:
            return int(round(float(value)))
        except ValueError:
            print("please input a number, unit: mm")


def show_plan_result(planner: CorridorPlanner, result: PlanResult) -> None:
    import cv2

    mapper = CanvasMapper(planner)
    message = "saved {}".format(result.svg_path.resolve() if result.svg_path else "")
    cv2.imshow("task1_path_planner", draw_scene(planner, result.cones_mm, result, mapper, message))
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyWindow("task1_path_planner")


def run_distance_input(planner: CorridorPlanner, out_path: Path) -> None:
    cones: List[Rect] = []
    for name in ("left", "right"):
        x = input_int_mm("{} cone x from x=0 line mm: ".format(name))
        y = input_int_mm("{} cone y from y=0 line mm: ".format(name))
        cone = planner.cone_from_center((x, y))
        print("{} cone plan center = ({}, {}) mm".format(name, cone.center[0], cone.center[1]))
        cones.append(cone)

    result = planner.plan_with_cones(cones)
    saved_svg_path = write_plan_files(result, out_path)
    result.svg_path = saved_svg_path
    print_plan(result)
    show_plan_result(planner, result)


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


def main() -> None:
    planner = CorridorPlanner()
    run_distance_input(planner, DEFAULT_PLAN_SVG_PATH)


if __name__ == "__main__":
    main()
