# -*- coding: utf-8 -*-
"""Display and record the Orbbec color stream.

Examples:
    python tools/orbbec_video_recorder.py
    python tools/orbbec_video_recorder.py --output recordings/test.mp4 --fps 30
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Orbbec Color Stream"


def parse_args():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Preview and record an Orbbec color stream")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings") / ("orbbec_{}.mp4".format(timestamp)),
        help="output video path (default: recordings/orbbec_<time>.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS stored in the video file (default: 30)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="four-character OpenCV codec (default: mp4v)",
    )
    return parser.parse_args()


def open_writer(output, codec, fps, frame_size):
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "Cannot open video writer for {!s}; try '--codec MJPG --output recordings/out.avi'".format(
                output
            )
        )
    return writer, output


def main():
    args = parse_args()

    # This is the same camera wrapper used by tools/vision.py.
    import orbbec_native

    camera = orbbec_native.OrbbecCamera()
    writer = None
    output = None
    frame_count = 0
    start_time = None

    try:
        camera.start()
        time.sleep(0.8)
        print("[Orbbec] camera started, press q or Esc to stop")

        while True:
            raw_frame = camera.get_color_frame()
            if raw_frame is None:
                # Keep the UI responsive while the camera has no new frame.
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            frame = np.asarray(raw_frame, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError("Unexpected color frame shape: {}".format(frame.shape))
            frame = np.ascontiguousarray(frame)

            if writer is None:
                height, width = frame.shape[:2]
                writer, output = open_writer(
                    args.output,
                    args.codec,
                    args.fps,
                    (width, height),
                )
                start_time = time.monotonic()
                print("[Record] saving {}x{} at {:.2f} FPS to {}".format(
                    width, height, args.fps, output
                ))

            writer.write(frame)
            frame_count += 1

            preview = frame.copy()
            elapsed = max(time.monotonic() - start_time, 1e-6)
            cv2.putText(
                preview,
                "REC  frames={}  capture={:.1f} FPS".format(frame_count, frame_count / elapsed),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("\n[Record] interrupted")
    finally:
        if writer is not None:
            writer.release()
        camera.stop()
        cv2.destroyAllWindows()

    if output is not None:
        print("[Record] saved {} frames to {}".format(frame_count, output))
    else:
        print("[Record] no frame received; no video was created")


if __name__ == "__main__":
    main()
