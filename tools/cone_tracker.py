# -*- coding: utf-8 -*-


TRACK_MAX_COST_PX = 220.0
DEPTH_INVALID_AS_MM = 2500


def cone_depth_mm(detection, invalid_as_mm=DEPTH_INVALID_AS_MM):
    return int(detection.depth_mm) if detection.depth_mm is not None else int(invalid_as_mm)


def assign_pair_relative_sides(cone_pair):
    left_cone, right_cone = sorted(cone_pair, key=lambda det: det.center[0])
    return {id(left_cone): "left", id(right_cone): "right"}


class ConeTrack:
    def __init__(self, info, detection):
        self.info = info
        self.name = info["name"]
        self.side = info["side"]
        self.detection = detection
        self.missing_count = 0
        self.bypassed = False
        self.update(detection)

    def update(self, detection):
        self.detection = detection
        self.center_x = float(detection.center[0])
        self.center_y = float(detection.center[1])
        self.depth_mm = cone_depth_mm(detection)
        self.missing_count = 0

    def mark_missing(self):
        self.missing_count += 1


class ConeTracker:
    def __init__(self, cone_infos, initial_detections, max_cost_px=TRACK_MAX_COST_PX):
        self.max_cost_px = float(max_cost_px)
        self.tracks = {
            info["name"]: ConeTrack(info, detection)
            for info, detection in zip(cone_infos, initial_detections)
        }
        self._print_tracks("init")

    def _cost(self, track, detection):
        return abs(float(detection.center[0]) - track.center_x) + 0.5 * abs(float(detection.center[1]) - track.center_y)

    def _print_tracks(self, tag):
        for name in ("cone1", "cone2"):
            track = self.tracks[name]
            print(
                "[Task1][Track] {} {} side={} bypassed={} missing={} depth_mm={} cx={:.1f} cy={:.1f}".format(
                    tag,
                    name,
                    track.side,
                    track.bypassed,
                    track.missing_count,
                    track.depth_mm,
                    track.center_x,
                    track.center_y,
                )
            )

    def update(self, detections):
        cones = sorted(detections, key=cone_depth_mm)[:2]
        if not cones:
            for track in self.tracks.values():
                track.mark_missing()
            self._print_tracks("missing")
            return

        active_names = ["cone1", "cone2"]
        assignments = []
        used_detection_ids = set()

        if len(active_names) == 2 and len(cones) >= 2:
            cone_a, cone_b = cones[0], cones[1]
            track1 = self.tracks["cone1"]
            track2 = self.tracks["cone2"]
            cost_keep = self._cost(track1, cone_a) + self._cost(track2, cone_b)
            cost_swap = self._cost(track1, cone_b) + self._cost(track2, cone_a)
            if cost_keep <= cost_swap:
                assignments = [("cone1", cone_a), ("cone2", cone_b)]
            else:
                assignments = [("cone1", cone_b), ("cone2", cone_a)]
        else:
            for name in active_names:
                track = self.tracks[name]
                candidates = [det for det in cones if id(det) not in used_detection_ids]
                if not candidates:
                    break
                best = min(candidates, key=lambda det: self._cost(track, det))
                assignments.append((name, best))
                used_detection_ids.add(id(best))

        assigned_names = set()
        for name, detection in assignments:
            track = self.tracks[name]
            cost = self._cost(track, detection)
            if cost <= self.max_cost_px or len(assignments) == 2:
                track.update(detection)
                assigned_names.add(name)
                print(
                    "[Task1][Track] update {} cost_px={:.1f} depth_mm={} cx={:.1f} cy={:.1f}".format(
                        name,
                        cost,
                        detection.depth_mm,
                        detection.center[0],
                        detection.center[1],
                    )
                )

        for name in active_names:
            if name not in assigned_names:
                self.tracks[name].mark_missing()
                print("[Task1][Track] {} missing_count={}".format(name, self.tracks[name].missing_count))

    def get_detection(self, name):
        track = self.tracks[name]
        if track.bypassed:
            return None
        if track.missing_count > 0:
            return None
        return track.detection

    def mark_bypassed(self, name):
        self.tracks[name].bypassed = True
        print("[Task1][Track] {} marked bypassed".format(name))
