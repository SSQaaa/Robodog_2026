#include <libobsensor/ObSensor.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

class OrbbecCamera {
public:
    OrbbecCamera()
        : running_(false),
          depth_width_(0),
          depth_height_(0),
          color_width_(0),
          color_height_(0) {}

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
/*
            if (depthFrame) {
                new_depth_w = depthFrame->width();
                new_depth_h = depthFrame->height();

                auto *depth_ptr = reinterpret_cast<uint16_t *>(depthFrame->data());
                if (depth_ptr && new_depth_w > 0 && new_depth_h > 0) {
                    new_depth.assign(depth_ptr, depth_ptr + new_depth_w * new_depth_h);
                }
            }
*/

            //////////////////////////////////  rotate180 ///////////////////////////
            if (depthFrame) {
                new_depth_w = depthFrame->width();
                new_depth_h = depthFrame->height();
            
                auto *depth_ptr = reinterpret_cast<uint16_t *>(depthFrame->data());
            
                if (depth_ptr && new_depth_w > 0 && new_depth_h > 0) {
                    cv::Mat depth_raw(new_depth_h, new_depth_w, CV_16UC1, depth_ptr);
                    cv::Mat depth_rotated;
            
                    // ? 深度图也旋转180°
                    cv::rotate(depth_raw, depth_rotated, cv::ROTATE_180);
            
                    new_depth.assign(
                        reinterpret_cast<uint16_t *>(depth_rotated.data),
                        reinterpret_cast<uint16_t *>(depth_rotated.data) + new_depth_w * new_depth_h
                    );
                }
            }

            if (colorFrame) {
                cv::Mat bgr = decodeColorFrame(colorFrame);
                if (!bgr.empty()) {
                    //////////////////////////////////  rotate180 ///////////////////////////
                    cv::rotate(bgr, bgr, cv::ROTATE_180);
                    
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
};