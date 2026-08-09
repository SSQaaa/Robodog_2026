# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

from tools.vision import DashboardInfer


class VisionManager:
    """在后台初始化视觉，并确保同一时刻只有一个对象占用摄像头。"""

    def __init__(self, show_dashboard_stream=False):
        self.show_dashboard_stream = show_dashboard_stream
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.dashboard_future = None
        self.dashboard_detector = None

    def start_dashboard(self):
        print("[Main] initialize task2 vision in background")
        self.dashboard_future = self.executor.submit(
            DashboardInfer,
            show_stream=self.show_dashboard_stream,
        )

    def get_dashboard(self, retry=False):
        if self.dashboard_detector is not None:
            return self.dashboard_detector
        try:
            self.dashboard_detector = self.dashboard_future.result()
        except Exception:
            if not retry:
                raise
            print("[Main] retry task2_3 vision initialization")
            self.dashboard_detector = DashboardInfer(show_stream=self.show_dashboard_stream)
        return self.dashboard_detector

    def close_dashboard(self):
        if self.dashboard_detector is not None:
            self.dashboard_detector.close()
            self.dashboard_detector = None
        self.dashboard_future = None

    def take_task3(self):
        print("[Main] reuse task2 vision for task3")
        vision = self.get_dashboard()
        self.dashboard_detector = None
        self.dashboard_future = None
        return vision

    def close(self):
        self.executor.shutdown(wait=True)
        if self.dashboard_detector is None and self.dashboard_future is not None:
            try:
                self.dashboard_detector = self.dashboard_future.result()
            except Exception:
                pass
        if self.dashboard_detector is not None:
            self.dashboard_detector.close()
