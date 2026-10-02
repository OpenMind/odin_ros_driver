/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Accumulates /odin1/cloud_slam (XYZRGB, odom frame) into a voxel-downsampled
// RGB map and writes it as a PCD file on shutdown or via the ~/save_pcd service.
// Enabled with: ros2 launch odin_ros_driver odin1_ros2.launch.py pcd:=true

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

class PcdMapSaverNode : public rclcpp::Node
{
public:
    PcdMapSaverNode()
        : Node("pcd_map_saver_node")
    {
        topic_ = this->declare_parameter<std::string>("cloud_topic", "/odin1/cloud_slam");
        leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.05);
        output_dir_ = this->declare_parameter<std::string>("output_dir", "");
        file_name_ = this->declare_parameter<std::string>("file_name", "");

        if (leaf_size_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(), "voxel_leaf_size must be > 0, using 0.05");
            leaf_size_ = 0.05;
        }
        if (output_dir_.empty()) {
            output_dir_ = default_output_dir();
        }
        if (file_name_.empty()) {
            file_name_ = "rgb_map_" + timestamp_string() + ".pcd";
        }

        auto qos = rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            topic_, qos,
            std::bind(&PcdMapSaverNode::cloudCallback, this, std::placeholders::_1));

        save_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "~/save_pcd",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                std::string path;
                res->success = save(path);
                res->message = res->success ? path : "save failed (map empty or write error)";
            });

        RCLCPP_INFO(this->get_logger(), "Collecting RGB map from %s (voxel %.3f m) -> %s/%s",
                    topic_.c_str(), leaf_size_, output_dir_.c_str(), file_name_.c_str());
    }

    // Streams the accumulated map to a binary PCD without copying it; returns false if empty or on write error.
    bool save(std::string& out_path)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (voxels_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No points received on %s, nothing to save", topic_.c_str());
            return false;
        }

        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        out_path = (std::filesystem::path(output_dir_) / file_name_).string();
        const std::string tmp_path = out_path + ".tmp";

        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        out << "# .PCD v0.7 - Point Cloud Data file format\n"
            << "VERSION 0.7\n"
            << "FIELDS x y z rgb\n"
            << "SIZE 4 4 4 4\n"
            << "TYPE F F F U\n"
            << "COUNT 1 1 1 1\n"
            << "WIDTH " << voxels_.size() << "\n"
            << "HEIGHT 1\n"
            << "VIEWPOINT 0 0 0 1 0 0 0\n"
            << "POINTS " << voxels_.size() << "\n"
            << "DATA binary\n";

        std::vector<PcdPoint> batch;
        batch.reserve(kWriteBatch);
        for (const auto& kv : voxels_) {
            const Voxel& v = kv.second;
            PcdPoint p;
            p.x = cellCenter(kv.first, 42, v.offset[0]);
            p.y = cellCenter(kv.first, 21, v.offset[1]);
            p.z = cellCenter(kv.first, 0, v.offset[2]);
            p.rgb = 0xff000000u | (uint32_t(v.rgb[0]) << 16) | (uint32_t(v.rgb[1]) << 8) | v.rgb[2];
            batch.push_back(p);
            if (batch.size() == kWriteBatch) {
                out.write(reinterpret_cast<const char*>(batch.data()), batch.size() * sizeof(PcdPoint));
                batch.clear();
            }
        }
        out.write(reinterpret_cast<const char*>(batch.data()), batch.size() * sizeof(PcdPoint));
        out.close();

        if (out) {
            std::filesystem::rename(tmp_path, out_path, ec);
        }
        if (!out || ec) {
            RCLCPP_ERROR(this->get_logger(), "Failed to write %s", out_path.c_str());
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Saved RGB map with %zu points to %s", voxels_.size(), out_path.c_str());
        return true;
    }

private:
    struct Voxel
    {
        uint8_t offset[3];
        uint8_t rgb[3];
        uint16_t count;
    };
    static_assert(sizeof(Voxel) == 8, "Voxel must stay 8 bytes");

    struct PcdPoint
    {
        float x, y, z;
        uint32_t rgb;
    };
    static_assert(sizeof(PcdPoint) == 16, "PCD point layout must match the header");

    static constexpr size_t kWriteBatch = 1 << 16;
    static constexpr int64_t kKeyOffset = 1 << 20;
    static constexpr uint64_t kKeyMask = (1ULL << 21) - 1;

    static void accumulate(uint8_t& mean, int sample, int n)
    {
        mean = static_cast<uint8_t>((mean * (n - 1) + sample + n / 2) / n);
    }

    float cellCenter(uint64_t key, int shift, uint8_t offset) const
    {
        const int64_t cell = static_cast<int64_t>((key >> shift) & kKeyMask) - kKeyOffset;
        return static_cast<float>((cell + (offset + 0.5) / 256.0) * leaf_size_);
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        bool has_rgb = false;
        for (const auto& f : msg->fields) {
            if (f.name == "rgb") has_rgb = true;
        }
        if (!has_rgb) {
            RCLCPP_WARN_ONCE(this->get_logger(), "%s has no rgb field, skipping", topic_.c_str());
            return;
        }

        sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> it_rgb(*msg, "rgb");
        const size_t n = static_cast<size_t>(msg->width) * msg->height;
        const double inv_leaf = 1.0 / leaf_size_;

        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z, ++it_rgb) {
            const float x = *it_x, y = *it_y, z = *it_z;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            uint32_t packed;
            const float rgb = *it_rgb;
            std::memcpy(&packed, &rgb, sizeof(packed));

            const double fx = x * inv_leaf, fy = y * inv_leaf, fz = z * inv_leaf;
            const double cx = std::floor(fx), cy = std::floor(fy), cz = std::floor(fz);
            Voxel& v = voxels_[voxelKey(cx, cy, cz)];
            if (v.count < std::numeric_limits<uint16_t>::max()) ++v.count;
            const int n = v.count;
            accumulate(v.offset[0], std::min(255, static_cast<int>((fx - cx) * 256)), n);
            accumulate(v.offset[1], std::min(255, static_cast<int>((fy - cy) * 256)), n);
            accumulate(v.offset[2], std::min(255, static_cast<int>((fz - cz) * 256)), n);
            accumulate(v.rgb[0], (packed >> 16) & 0xff, n);
            accumulate(v.rgb[1], (packed >> 8) & 0xff, n);
            accumulate(v.rgb[2], packed & 0xff, n);
        }

        if (++msg_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Map has %zu voxels after %zu clouds", voxels_.size(), msg_count_);
        }
    }

    // Packs 21 bits per axis (about +/-52 km at 5 cm) into one 64-bit key.
    static uint64_t voxelKey(double cx, double cy, double cz)
    {
        const uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(cx) + kKeyOffset) & kKeyMask;
        const uint64_t iy = static_cast<uint64_t>(static_cast<int64_t>(cy) + kKeyOffset) & kKeyMask;
        const uint64_t iz = static_cast<uint64_t>(static_cast<int64_t>(cz) + kKeyOffset) & kKeyMask;
        return (ix << 42) | (iy << 21) | iz;
    }

    // Same location as the .bin maps: {ws}/src/odin_ros_driver/map/pcd
    static std::string default_output_dir()
    {
        const char* prefix = std::getenv("COLCON_PREFIX_PATH");
        if (prefix) {
            std::string path(prefix);
            size_t pos = path.find("/install");
            if (pos != std::string::npos) {
                return path.substr(0, pos) + "/src/odin_ros_driver/map/pcd";
            }
        }
        const char* home = std::getenv("HOME");
        return std::string(home ? home : "/tmp") + "/odin_maps";
    }

    static std::string timestamp_string()
    {
        std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
        return buf;
    }

    std::string topic_;
    double leaf_size_;
    std::string output_dir_;
    std::string file_name_;
    size_t msg_count_ = 0;

    std::mutex mutex_;
    std::unordered_map<uint64_t, Voxel> voxels_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PcdMapSaverNode>();
    rclcpp::spin(node);

    // Ctrl+C stops spin; write the final map before exiting.
    std::string path;
    node->save(path);

    rclcpp::shutdown();
    return 0;
}
