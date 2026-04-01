from tasks.dashboard_letter_detector import SimpleInfer, print_dashboard_result, analyze_infer_output
import time


def task2(robot):
    detector = SimpleInfer()
    try:
        # move turn_left back_move
        # robot.move(last_time=10, vx=20000)
        # time.sleep(0.5)
        #
        # robot.revolve_90_l()
        # time.sleep(0.5)
        # robot.move(last_time=3, vx=-20000)
        # time.sleep(0.5)
        #
        # # first dashboard
        # robot.move(last_time=3, vy=-20000)
        # time.sleep(0.5)

        # realtime infer loop
        while True:
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            if result["dashboard_count"] <= 0:
                # robot.move(last_time=0.5, vy=-20000)
                print("没有仪表盘")
                time.sleep(1)
                continue

            if result["dashboard_count"] == 1:
                while True:
                    infer_output = detector.infer_once()
                    result = analyze_infer_output(infer_output)
                    if len(result["xyxy_list"]) < 1 or len(result["area_list"]) < 1:
                        continue
                    _, x1, y1, x2, y2 = result["xyxy_list"][0]
                    x_m = int((x2 - x1) / 2 + x1)
                    if x_m < 300:
                        # robot.move(last_time=0.2, vy=20000)
                        print("image is left, move right")
                        time.sleep(1)
                        continue
                    elif x_m > 340:
                        # robot.move(last_time=0.2, vy=-20000)
                        print("image is right, move left")
                        time.sleep(1)
                        continue
                    else:
                        print("image x centered")
                        break

                while True:
                    infer_output = detector.infer_once()
                    result = analyze_infer_output(infer_output)
                    if len(result["area_list"]) < 1:
                        continue
                    _, area = result["area_list"][0]
                    if area < 20000:
                        # robot.move(last_time=0.2, vx=20000)
                        print("image too small, move forward")
                        time.sleep(1)
                        continue
                    elif area > 30000:
                        # robot.move(last_time=0.2, vx=-20000)
                        print("image too big, move back")
                        time.sleep(1)
                        continue
                    else:
                        print("image size good")
                        break

                print("ready to report")
                print("dashboard state", result["state_list"][0][1])
                break

            elif result["dashboard_count"] == 2:
                if len(result["area_list"]) < 2 or len(result["xyxy_list"]) < 2:
                    continue
                max_area = max(result["area_list"], key=lambda x: x[1])
                index_max = max_area[0]
                min_area = min(result["area_list"], key=lambda x: x[1])
                index_min = min_area[0]
                x1_map = {int(n): x1 for n, x1, y1, x2, y2 in result["xyxy_list"]}
                x1_max = x1_map.get(int(index_max))
                x1_min = x1_map.get(int(index_min))
                if x1_max is None or x1_min is None:
                    continue
                if x1_max > x1_min:
                    robot.move(last_time=0.2, vy=20000)
                else:
                    robot.move(last_time=0.2, vy=-20000)
                continue

            else:
                robot.move(last_time=0.2, vy=20000)
                continue

        """
            # second dashboard
            robot.move(last_time=5, vy=-20000)
            time.sleep(0.5)
            infer_output = detector.infer_once()
            print_dashboard_result(infer_output, "second_dashboard")

            # third dashboard
            robot.revolve_90_r()
            time.sleep(0.5)
            robot.move(last_time=2, vx=20000)
            time.sleep(0.5)
            robot.move(last_time=2, vy=-20000)
            time.sleep(0.5)
            infer_output = detector.infer_once()
            print_dashboard_result(infer_output, "third_dashboard")

            # fourth dashboard
            robot.revolve_180()
            infer_output = detector.infer_once()
            print_dashboard_result(infer_output, "fourth_dashboard")
        """
    finally:
        detector.close()
        print("detector.close() done, task2 finished")
