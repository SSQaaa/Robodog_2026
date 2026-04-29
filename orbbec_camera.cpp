#include <libobsensor/ObSensor.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
// Set this to false when the camera is mounted upright.
const bool kRotateFrames180 = true;
}

class OrbbecCamera {
public:
    OrbbecCamera()
        : running_(false),
          depth_width_(0),
          depth_height_(0),
          color_width_(0),
          color_height_(0),
          rotate_180_(kRotateFrames180) {}

    ~OrbbecCamera() {
        stop();
    }

    void start() {
        if (running_) return;

        config_ = std::make_shared<ob::Config>();

        auto depthProfiles = pipe_.getStreamProfileList(OB_SENSOR_DEPTH);
        auto depthProfile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
        config_->enableStream(depthProfile);

        auto colorProfiles = pipe_.getStreamProfileList(OB_SENSOR_COLOR);
        auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
        config_->enableStream(colorProfile);

        // 如果你的 SDK 支持 D2C 软件对齐，可以打开这一句。
        // 如果编译报 OB_ALIGN_D2C_SW_MODE 未定义，就保持注释。
        // config_->setAlignMode(OB_ALIGN_D2C_SW_MODE);

        pipe_.start(config_);

        printCameraIntrinsics();

        running_ = true;
        capture_thread_ = std::thread(&OrbbecCamera::captureLoop, this);
    }

    void stop() {
        if (!running_) return;

        running_ = false;

        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }

        try {
            pipe_.stop();
        } catch (...) {
        }
    }

    std::pair<int, int> get_depth_size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return {depth_width_, depth_height_};
    }

    std::pair<int, int> get_color_size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return {color_width_, color_height_};
    }

    bool is_rotate_180_enabled() const {
        return rotate_180_;
    }

    py::dict get_color_intrinsics() {
        auto cameraParam = pipe_.getCameraParam();
        auto intr = cameraParam.rgbIntrinsic;
        return makeIntrinsicsDict(
            intr.fx,
            intr.fy,
            intr.cx,
            intr.cy,
            intr.width,
            intr.height,
            rotate_180_
        );
    }

    py::dict get_depth_intrinsics() {
        auto cameraParam = pipe_.getCameraParam();
        auto intr = cameraParam.depthIntrinsic;
        return makeIntrinsicsDict(
            intr.fx,
            intr.fy,
            intr.cx,
            intr.cy,
            intr.width,
            intr.height,
            rotate_180_
        );
    }

    py::object get_color_frame() {
        std::lock_guard<std::mutex> lock(mtx_);

        if (color_bgr_.empty() || color_width_ <= 0 || color_height_ <= 0) {
            return py::none();
        }

        py::array_t<uint8_t> img({color_height_, color_width_, 3});
        std::memcpy(img.mutable_data(), color_bgr_.data(), color_bgr_.size());

        return img;
    }

    int get_depth(int x, int y) {
        std::lock_guard<std::mutex> lock(mtx_);

        if (depth_data_.empty()) return -1;
        if (x < 0 || y < 0 || x >= depth_width_ || y >= depth_height_) return -2;

        return static_cast<int>(depth_data_[y * depth_width_ + x]);
    }

    std::pair<int, int> get_depth_in_box(int x1, int y1, int x2, int y2) {
        std::lock_guard<std::mutex> lock(mtx_);

        if (depth_data_.empty()) return {-1, 0};

        if (x1 > x2) std::swap(x1, x2);
        if (y1 > y2) std::swap(y1, y2);

        x1 = std::max(0, std::min(x1, depth_width_ - 1));
        x2 = std::max(0, std::min(x2, depth_width_ - 1));
        y1 = std::max(0, std::min(y1, depth_height_ - 1));
        y2 = std::max(0, std::min(y2, depth_height_ - 1));

        int bw = x2 - x1 + 1;
        int bh = y2 - y1 + 1;

        int shrink_x = bw / 10;
        int shrink_y = bh / 10;

        x1 += shrink_x;
        x2 -= shrink_x;
        y1 += shrink_y;
        y2 -= shrink_y;

        if (x1 >= x2 || y1 >= y2) return {-2, 0};

        std::vector<int> valid;
        valid.reserve((x2 - x1 + 1) * (y2 - y1 + 1));

        const int MAX_DEPTH_MM = 5000;

        for (int y = y1; y <= y2; y++) {
            for (int x = x1; x <= x2; x++) {
                uint16_t d = depth_data_[y * depth_width_ + x];
                if (d > 0 && d < MAX_DEPTH_MM) {
                    valid.push_back(static_cast<int>(d));
                }
            }
        }

        if (valid.size() < 20) {
            return {-3, static_cast<int>(valid.size())};
        }

        size_t mid = valid.size() / 2;
        std::nth_element(valid.begin(), valid.begin() + mid, valid.end());

        return {valid[mid], static_cast<int>(valid.size())};
    }

private:
    void captureLoop() {
        while (running_) {
            auto frameset = pipe_.waitForFrames(100);
            if (!frameset) continue;

            auto depthFrame = frameset->depthFrame();
            auto colorFrame = frameset->colorFrame();

            std::vector<uint16_t> new_depth;
            int new_depth_w = 0;
            int new_depth_h = 0;

            std::vector<uint8_t> new_color_bgr;
            int new_color_w = 0;
            int new_color_h = 0;
            if (depthFrame) {
                extractDepthFrame(depthFrame, new_depth, new_depth_w, new_depth_h);
            }

            if (colorFrame) {
                cv::Mat bgr = decodeColorFrame(colorFrame);
                if (!bgr.empty()) {
                    bgr = rotateFrameIfNeeded(bgr);
                    new_color_w = bgr.cols;
                    new_color_h = bgr.rows;
                    new_color_bgr.assign(bgr.data, bgr.data + bgr.total() * bgr.elemSize());
                }
            }

            {
                std::lock_guard<std::mutex> lock(mtx_);

                if (!new_depth.empty()) {
                    depth_width_ = new_depth_w;
                    depth_height_ = new_depth_h;
                    depth_data_.swap(new_depth);
                }

                if (!new_color_bgr.empty()) {
                    color_width_ = new_color_w;
                    color_height_ = new_color_h;
                    color_bgr_.swap(new_color_bgr);
                }
            }
        }
    }

    void extractDepthFrame(const std::shared_ptr<ob::DepthFrame> &frame,
                           std::vector<uint16_t> &depth,
                           int &width,
                           int &height) {
        width = frame->width();
        height = frame->height();

        auto *depth_ptr = reinterpret_cast<uint16_t *>(frame->data());
        if (!depth_ptr || width <= 0 || height <= 0) {
            return;
        }

        cv::Mat raw(height, width, CV_16UC1, depth_ptr);
        cv::Mat prepared = rotateFrameIfNeeded(raw);
        const auto *prepared_ptr = reinterpret_cast<const uint16_t *>(prepared.data);
        depth.assign(prepared_ptr, prepared_ptr + width * height);
    }

    cv::Mat rotateFrameIfNeeded(const cv::Mat &frame) {
        if (!rotate_180_) {
            return frame;
        }

        cv::Mat rotated;
        cv::rotate(frame, rotated, cv::ROTATE_180);
        return rotated;
    }

    cv::Mat decodeColorFrame(const std::shared_ptr<ob::ColorFrame> &frame) {
        int width = frame->width();
        int height = frame->height();
        auto format = frame->format();
        void *data = frame->data();
        size_t data_size = frame->dataSize();

        if (!data || width <= 0 || height <= 0 || data_size == 0) {
            return cv::Mat();
        }

        cv::Mat bgr;

        if (format == OB_FORMAT_RGB) {
            cv::Mat rgb(height, width, CV_8UC3, data);
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        } else if (format == OB_FORMAT_BGR) {
            cv::Mat tmp(height, width, CV_8UC3, data);
            bgr = tmp.clone();
        } else if (format == OB_FORMAT_MJPG) {
            std::vector<uchar> jpg_data(
                reinterpret_cast<uchar *>(data),
                reinterpret_cast<uchar *>(data) + data_size
            );
            bgr = cv::imdecode(jpg_data, cv::IMREAD_COLOR);
        } else if (format == OB_FORMAT_YUYV) {
            cv::Mat yuyv(height, width, CV_8UC2, data);
            cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
        } else if (format == OB_FORMAT_UYVY) {
            cv::Mat uyvy(height, width, CV_8UC2, data);
            cv::cvtColor(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
        } else {
            return cv::Mat();
        }

        if (!bgr.empty() && !bgr.isContinuous()) {
            bgr = bgr.clone();
        }

        return bgr;
    }

    py::dict makeIntrinsicsDict(double fx,
                                double fy,
                                double cx,
                                double cy,
                                int width,
                                int height,
                                bool rotated_180) {
        double used_cx = cx;
        double used_cy = cy;

        /*
         * The SDK returns intrinsics for the original, unrotated image.
         * This class currently rotates both color and depth frames by 180 degrees
         * before returning them to Python.
         *
         * A pixel at (u, v) in the original image becomes:
         *     u_rot = width  - 1 - u
         *     v_rot = height - 1 - v
         *
         * The optical center must be transformed in the same way:
         *     cx_rot = width  - 1 - cx
         *     cy_rot = height - 1 - cy
         *
         * fx and fy do not change after a 180 degree rotation.
         *
         * If the camera is later mounted upright, set rotate_180_ to false.
         * Then the returned cx/cy will be the raw SDK values.
         */
        if (rotated_180) {
            used_cx = static_cast<double>(width) - 1.0 - cx;
            used_cy = static_cast<double>(height) - 1.0 - cy;
        }

        py::dict d;
        d["fx"] = fx;
        d["fy"] = fy;
        d["cx"] = used_cx;
        d["cy"] = used_cy;
        d["raw_cx"] = cx;
        d["raw_cy"] = cy;
        d["width"] = width;
        d["height"] = height;
        d["rotated_180"] = rotated_180;
        return d;
    }

    void printCameraIntrinsics() {
        auto color = get_color_intrinsics();
        auto depth = get_depth_intrinsics();

        std::cout << "[Color Intrinsics used by Python] "
                  << "fx=" << color["fx"].cast<double>()
                  << ", fy=" << color["fy"].cast<double>()
                  << ", cx=" << color["cx"].cast<double>()
                  << ", cy=" << color["cy"].cast<double>()
                  << ", raw_cx=" << color["raw_cx"].cast<double>()
                  << ", raw_cy=" << color["raw_cy"].cast<double>()
                  << ", width=" << color["width"].cast<int>()
                  << ", height=" << color["height"].cast<int>()
                  << ", rotated_180=" << color["rotated_180"].cast<bool>()
                  << std::endl;

        std::cout << "[Depth Intrinsics used by Python] "
                  << "fx=" << depth["fx"].cast<double>()
                  << ", fy=" << depth["fy"].cast<double>()
                  << ", cx=" << depth["cx"].cast<double>()
                  << ", cy=" << depth["cy"].cast<double>()
                  << ", raw_cx=" << depth["raw_cx"].cast<double>()
                  << ", raw_cy=" << depth["raw_cy"].cast<double>()
                  << ", width=" << depth["width"].cast<int>()
                  << ", height=" << depth["height"].cast<int>()
                  << ", rotated_180=" << depth["rotated_180"].cast<bool>()
                  << std::endl;
    }

private:
    ob::Pipeline pipe_;
    std::shared_ptr<ob::Config> config_;

    std::atomic<bool> running_;
    std::thread capture_thread_;
    std::mutex mtx_;

    std::vector<uint16_t> depth_data_;
    int depth_width_;
    int depth_height_;

    std::vector<uint8_t> color_bgr_;
    int color_width_;
    int color_height_;

    bool rotate_180_;
};
