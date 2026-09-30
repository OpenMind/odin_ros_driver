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

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <unordered_map>

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

    // Writes the accumulated map; returns false if empty or on write error.
    bool save(std::string& out_path)
    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cloud.reserve(voxels_.size());
            for (const auto& kv : voxels_) {
                const Voxel& v = kv.second;
                const double n = static_cast<double>(v.count);
                pcl::PointXYZRGB p;
                p.x = static_cast<float>(v.x / n);
                p.y = static_cast<float>(v.y / n);
                p.z = static_cast<float>(v.z / n);
                p.r = static_cast<uint8_t>(v.r / n);
                p.g = static_cast<uint8_t>(v.g / n);
                p.b = static_cast<uint8_t>(v.b / n);
                cloud.push_back(p);
            }
        }
        if (cloud.empty()) {
            RCLCPP_WARN(this->get_logger(), "No points received on %s, nothing to save", topic_.c_str());
            return false;
        }
        cloud.width = static_cast<uint32_t>(cloud.size());
        cloud.height = 1;
        cloud.is_dense = true;

        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        out_path = (std::filesystem::path(output_dir_) / file_name_).string();
        if (pcl::io::savePCDFileBinary(out_path, cloud) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to write %s", out_path.c_str());
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Saved RGB map with %zu points to %s", cloud.size(), out_path.c_str());
        return true;
    }

private:
    struct Voxel
    {
        double x = 0, y = 0, z = 0;
        uint64_t r = 0, g = 0, b = 0;
        uint32_t count = 0;
    };

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

            Voxel& v = voxels_[voxelKey(x * inv_leaf, y * inv_leaf, z * inv_leaf)];
            v.x += x; v.y += y; v.z += z;
            v.r += (packed >> 16) & 0xff;
            v.g += (packed >> 8) & 0xff;
            v.b += packed & 0xff;
            ++v.count;
        }

        if (++msg_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Map has %zu voxels after %zu clouds", voxels_.size(), msg_count_);
        }
    }

    // Packs 21 bits per axis (about +/-52 km at 5 cm) into one 64-bit key.
    static uint64_t voxelKey(double fx, double fy, double fz)
    {
        constexpr int64_t kOffset = 1 << 20;
        constexpr uint64_t kMask = (1ULL << 21) - 1;
        const uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(std::floor(fx)) + kOffset) & kMask;
        const uint64_t iy = static_cast<uint64_t>(static_cast<int64_t>(std::floor(fy)) + kOffset) & kMask;
        const uint64_t iz = static_cast<uint64_t>(static_cast<int64_t>(std::floor(fz)) + kOffset) & kMask;
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
