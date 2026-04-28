import argparse
import time

from dashboard_letter_detector import SimpleInfer, analyze_infer_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="show realtime video stream")
    args = parser.parse_args()

    detector = SimpleInfer(show_stream=args.stream)
    frame_id = 0

    try:
        while True:
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            x_m = None
            if len(result["xyxy_list"]) > 0:
                _, x1, y1, x2, y2 = result["xyxy_list"][0]
                x_m = int((x2 - x1) / 2 + x1)

            frame_id += 1
            print(
                "frame={} dashboard_count={} area_list={} x_m={}".format(
                    frame_id, result["dashboard_count"], result["area_list"], x_m
                )
            )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("stop by keyboard")
    finally:
        detector.close()
        print("detector.close() done")


if __name__ == "__main__":
    main()
