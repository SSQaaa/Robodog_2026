INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.2
CONF_THRESH_DASHBOARD = 0.1
IOU_THRESHOLD = 0.4

categories = ["A", "B", "C", "D", "GC", "RC", "dashboard", "ssi"]

# Detection output element count for current engine.
# For this engine, each detection row length is 38.
DETECTION_SIZE = 38
DETECTION_SIZE_CANDIDATES = [6, 13, 38]
DEBUG_DETECTION_SIZE = ("--debug-size" in sys.argv)
DEBUG_PRINT_EVERY = 15


class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        self.debug_frame_count = 0

        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def infer(self, image_raw):
        self.cfx.push()
        try:
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            np.copyto(self.host_inputs[0], input_image.ravel())

            start = time.time()
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()
            end = time.time()

            output = self.host_outputs[0]
            result_boxes, result_scores, result_classid = self.post_process(output, origin_h, origin_w)
            return image_raw, result_boxes, result_scores, result_classid, end - start
        finally:
            self.cfx.pop()

    def destroy(self):
        self.cfx.pop()
        self.cfx.detach()

    def preprocess_image(self, image_raw):
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0

        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h

        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def nms(self, boxes, scores, iou_threshold=IOU_THRESHOLD):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        keep = []
        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]
            keep.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]

        return keep

    def nms_classwise(self, boxes, scores, classid, iou_threshold=IOU_THRESHOLD):
        keep_all = []
        classid_int = classid.astype(np.int32)
        unique_cls = np.unique(classid_int)

        for cid in unique_cls:
            idx = np.where(classid_int == cid)[0]
            if idx.size == 0:
                continue
            sub_keep = self.nms(boxes[idx, :], scores[idx], iou_threshold)
            keep_all.extend(idx[sub_keep].tolist())

        if len(keep_all) == 0:
            return np.empty((0,), dtype=np.int32)

        keep_all = np.array(keep_all, dtype=np.int32)
        order = scores[keep_all].argsort()[::-1]
        return keep_all[order]

    def post_process(self, output, origin_h, origin_w):
        if DEBUG_DETECTION_SIZE:
            self.debug_frame_count += 1
            if self.debug_frame_count % DEBUG_PRINT_EVERY == 1:
                self._debug_detection_size(output)

        num = int(output[0])
        if num <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        raw = output[1:]
        valid_len = (len(raw) // DETECTION_SIZE) * DETECTION_SIZE
        pred = np.reshape(raw[:valid_len], (-1, DETECTION_SIZE))
        if num > pred.shape[0]:
            num = pred.shape[0]
        pred = pred[:num, :]

        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        valid_class = (classid >= 0) & (classid < len(categories))
        boxes = boxes[valid_class, :]
        scores = scores[valid_class]
        classid = classid[valid_class]

        classid_int = classid.astype(np.int32)
        score_thresh = np.where(classid_int == 6, CONF_THRESH_DASHBOARD, CONF_THRESH)
        keep_score = scores > score_thresh
        boxes = boxes[keep_score, :]
        scores = scores[keep_score]
        classid = classid[keep_score]

        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        indices = self.nms_classwise(boxes, scores, classid, IOU_THRESHOLD)

        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        return result_boxes, result_scores, result_classid

    def _debug_detection_size(self, output):
        raw = output[1:]
        num_raw = int(output[0])
        print(
            "[debug_size] num_raw={}, raw_len={}, now_DETECTION_SIZE={}".format(
                num_raw, len(raw), DETECTION_SIZE
            )
        )

        for ds in DETECTION_SIZE_CANDIDATES:
            rows = len(raw) // ds
            if rows <= 0:
                print("[debug_size] ds={} rows=0 skip".format(ds))
                continue

            pred = np.reshape(raw[: rows * ds], (-1, ds))
            valid_num = min(max(num_raw, 0), pred.shape[0])
            pred = pred[:valid_num, :]

            if pred.shape[0] == 0:
                print("[debug_size] ds={} rows={} valid_num=0 skip".format(ds, rows))
                continue

            if pred.shape[1] < 6:
                print("[debug_size] ds={} columns<6 skip".format(ds))
                continue

            score = pred[:, 4]
            cid = pred[:, 5]
            score_ok = float(np.mean((score >= 0.0) & (score <= 1.0)))
            cid_int = float(np.mean(np.abs(cid - np.round(cid)) < 1e-3))
            cid_range = float(np.mean((cid >= 0.0) & (cid < float(len(categories)))))

            print(
                "[debug_size] ds={} rows={} used={} score01={:.3f} cid_int={:.3f} cid_in_range={:.3f}".format(
                    ds, rows, valid_num, score_ok, cid_int, cid_range
                )
            )