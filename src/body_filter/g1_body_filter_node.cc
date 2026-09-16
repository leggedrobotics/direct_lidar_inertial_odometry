// g1_body_filter_node.cc  (ROS 2 Humble)
//
// Self-filter for the G1: removes points that fall on/near the robot's own body from a
// PointCloud2. Replacement for the ROS 1 `sensor_filters` + `robot_body_filter` chain used in
// dlio.launch (robot_body_filter/RobotBodyFilterPointCloud2 with do_contains_test only), which is
// not available for ROS 2 Humble from apt.
//
// Semantics kept from that configuration:
//   * the body model is a dedicated URDF whose collision shapes are simple primitives
//     (boxes / spheres / cylinders, e.g. g1_description/g1_29dof_body_filter.urdf);
//   * only the links listed in `only_links` are used (all links with collisions if empty);
//   * inflation: extents * scale + padding, with per-link overrides;
//   * contains test only (no clipping, no shadow/ray test), cloud already deskewed to a single
//     stamp, link poses looked up at the cloud stamp (fallback: latest);
//   * optional MarkerArray of the inflated test shapes for RViz.
// Points are copied byte-wise, so all fields of the input cloud are preserved.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <urdf/model.h>

class G1BodyFilterNode : public rclcpp::Node
{
public:
  G1BodyFilterNode()
  : rclcpp::Node("body_filter"),
    tf_buffer_(this->get_clock())
  {
    // ---- parameters ----
    const std::string urdf_path = declare_parameter<std::string>("urdf_path", "");
    only_links_ = declare_parameter<std::vector<std::string>>("only_links", std::vector<std::string>{});
    const double scale   = declare_parameter<double>("inflation_scale", 1.0);
    const double padding = declare_parameter<double>("inflation_padding", 0.0);
    const auto pl_pad_links  = declare_parameter<std::vector<std::string>>("per_link_padding_links", std::vector<std::string>{});
    const auto pl_pad_values = declare_parameter<std::vector<double>>("per_link_padding_values", std::vector<double>{});
    const auto pl_sc_links   = declare_parameter<std::vector<std::string>>("per_link_scale_links", std::vector<std::string>{});
    const auto pl_sc_values  = declare_parameter<std::vector<double>>("per_link_scale_values", std::vector<double>{});
    tf_timeout_    = declare_parameter<double>("tf_timeout", 0.2);
    debug_markers_ = declare_parameter<bool>("debug_markers", false);

    if (pl_pad_links.size() != pl_pad_values.size() || pl_sc_links.size() != pl_sc_values.size()) {
      throw std::runtime_error("per_link_*_links and per_link_*_values must have the same length");
    }
    std::map<std::string, double> link_padding, link_scale;
    for (size_t i = 0; i < pl_pad_links.size(); ++i) link_padding[pl_pad_links[i]] = pl_pad_values[i];
    for (size_t i = 0; i < pl_sc_links.size(); ++i)  link_scale[pl_sc_links[i]]   = pl_sc_values[i];

    // ---- body model ----
    if (urdf_path.empty()) {
      throw std::runtime_error("parameter 'urdf_path' (body-filter URDF) is required");
    }
    urdf::Model model;
    if (!model.initFile(urdf_path)) {
      throw std::runtime_error("failed to parse URDF: " + urdf_path);
    }
    const std::set<std::string> wanted(only_links_.begin(), only_links_.end());
    std::set<std::string> found;
    for (const auto& kv : model.links_) {
      const auto& link = kv.second;
      if (!wanted.empty() && wanted.count(link->name) == 0) continue;
      const double s = link_scale.count(link->name)   ? link_scale[link->name]   : scale;
      const double p = link_padding.count(link->name) ? link_padding[link->name] : padding;
      for (const auto& col : link->collision_array) {
        if (!col || !col->geometry) continue;
        Shape sh;
        sh.link = link->name;
        const auto& o = col->origin;
        double qx, qy, qz, qw; o.rotation.getQuaternion(qx, qy, qz, qw);
        sh.link_T_shape = Eigen::Translation3d(o.position.x, o.position.y, o.position.z) *
                          Eigen::Quaterniond(qw, qx, qy, qz);
        switch (col->geometry->type) {
          case urdf::Geometry::BOX: {
            auto b = std::dynamic_pointer_cast<urdf::Box>(col->geometry);
            sh.type = Shape::BOX;
            sh.half = Eigen::Vector3d(b->dim.x, b->dim.y, b->dim.z) * 0.5 * s + Eigen::Vector3d::Constant(p);
            sh.bound_r = sh.half.norm();
            break;
          }
          case urdf::Geometry::SPHERE: {
            auto sp = std::dynamic_pointer_cast<urdf::Sphere>(col->geometry);
            sh.type = Shape::SPHERE;
            sh.radius = sp->radius * s + p;
            sh.bound_r = sh.radius;
            break;
          }
          case urdf::Geometry::CYLINDER: {
            auto cy = std::dynamic_pointer_cast<urdf::Cylinder>(col->geometry);
            sh.type = Shape::CYLINDER;
            sh.radius = cy->radius * s + p;
            sh.half_len = cy->length * 0.5 * s + p;
            sh.bound_r = std::sqrt(sh.radius * sh.radius + sh.half_len * sh.half_len);
            break;
          }
          default:
            RCLCPP_WARN(get_logger(), "[body_filter] link '%s': unsupported collision geometry (mesh?) skipped",
                        link->name.c_str());
            continue;
        }
        shapes_.push_back(sh);
        found.insert(link->name);
      }
    }
    for (const auto& l : wanted) {
      if (!found.count(l)) {
        RCLCPP_WARN(get_logger(), "[body_filter] only_links entry '%s' has no usable collision in %s",
                    l.c_str(), urdf_path.c_str());
      }
    }
    if (shapes_.empty()) {
      throw std::runtime_error("body filter has no shapes to test against");
    }
    for (const auto& sh : shapes_) links_.insert(sh.link);

    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);

    // topics are meant to be remapped from the launch file ("input" / "output")
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("output", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    if (debug_markers_) {
      marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/robot_model_for_contains_test", 1);
    }
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "input", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
        std::bind(&G1BodyFilterNode::onCloud, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "[body_filter] %zu shapes from %zu links (scale=%.2f padding=%.3f) urdf=%s",
                shapes_.size(), links_.size(), scale, padding, urdf_path.c_str());
  }

private:
  struct Shape {
    enum Type { BOX, SPHERE, CYLINDER } type{BOX};
    std::string link;
    Eigen::Isometry3d link_T_shape{Eigen::Isometry3d::Identity()};
    Eigen::Vector3d half{0, 0, 0};   // box half extents (inflated)
    double radius{0.0};              // sphere / cylinder radius (inflated)
    double half_len{0.0};            // cylinder half length (inflated)
    double bound_r{0.0};             // bounding-sphere radius for the quick reject
    // per-frame state (cloud frame)
    Eigen::Isometry3d shape_T_cloud{Eigen::Isometry3d::Identity()};
    Eigen::Vector3d center{0, 0, 0};
    bool active{false};
  };

  static bool contains(const Shape& sh, const Eigen::Vector3d& p_cloud)
  {
    const Eigen::Vector3d d = p_cloud - sh.center;
    if (d.squaredNorm() > sh.bound_r * sh.bound_r) return false;   // quick reject
    const Eigen::Vector3d q = sh.shape_T_cloud * p_cloud;
    switch (sh.type) {
      case Shape::BOX:
        return std::fabs(q.x()) <= sh.half.x() && std::fabs(q.y()) <= sh.half.y() && std::fabs(q.z()) <= sh.half.z();
      case Shape::SPHERE:
        return q.squaredNorm() <= sh.radius * sh.radius;
      case Shape::CYLINDER:
        return std::fabs(q.z()) <= sh.half_len && (q.x() * q.x() + q.y() * q.y()) <= sh.radius * sh.radius;
    }
    return false;
  }

  // Look up cloud_frame -> link for every link at `stamp`; returns number of active links.
  size_t updatePoses(const std::string& cloud_frame, const builtin_interfaces::msg::Time& stamp)
  {
    const tf2::TimePoint tp = tf2_ros::fromMsg(stamp);
    const tf2::Duration timeout = tf2::durationFromSec(tf_timeout_);
    std::map<std::string, Eigen::Isometry3d> cloud_T_link;
    for (const auto& link : links_) {
      geometry_msgs::msg::TransformStamped t;
      bool ok = false;
      try {
        t = tf_buffer_.lookupTransform(cloud_frame, link, tp, timeout);
        ok = true;
      } catch (const tf2::TransformException&) {
        try {
          t = tf_buffer_.lookupTransform(cloud_frame, link, tf2::TimePointZero, timeout);
          ok = true;
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "[body_filter] TF %s->%s at cloud stamp unavailable; using latest",
                               cloud_frame.c_str(), link.c_str());
        } catch (const tf2::TransformException& ex) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "[body_filter] TF %s->%s unavailable, link skipped: %s",
                               cloud_frame.c_str(), link.c_str(), ex.what());
        }
      }
      if (ok) cloud_T_link[link] = tf2::transformToEigen(t.transform);
    }
    size_t active = 0;
    for (auto& sh : shapes_) {
      auto it = cloud_T_link.find(sh.link);
      sh.active = (it != cloud_T_link.end());
      if (!sh.active) continue;
      const Eigen::Isometry3d cloud_T_shape = it->second * sh.link_T_shape;
      sh.center = cloud_T_shape.translation();
      sh.shape_T_cloud = cloud_T_shape.inverse();
      ++active;
    }
    return active;
  }

  void onCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    if (updatePoses(msg->header.frame_id, msg->header.stamp) == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[body_filter] no link poses available, passing cloud through unfiltered");
      pub_->publish(*msg);
      return;
    }

    // byte-wise copy of the surviving points (keeps every field of the input cloud)
    sensor_msgs::msg::PointCloud2 out;
    out.header = msg->header;
    out.fields = msg->fields;
    out.is_bigendian = msg->is_bigendian;
    out.point_step = msg->point_step;
    out.height = 1;
    out.is_dense = msg->is_dense;
    out.data.reserve(msg->data.size());

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x"), it_y(*msg, "y"), it_z(*msg, "z");
    const uint8_t* src = msg->data.data();
    const size_t n_in = static_cast<size_t>(msg->width) * msg->height;
    size_t kept = 0;
    for (size_t i = 0; i < n_in; ++i, ++it_x, ++it_y, ++it_z) {
      const Eigen::Vector3d p(*it_x, *it_y, *it_z);
      bool inside = false;
      if (std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z())) {
        for (const auto& sh : shapes_) {
          if (sh.active && contains(sh, p)) { inside = true; break; }
        }
      }
      if (inside) continue;
      const uint8_t* pt = src + i * msg->point_step;
      out.data.insert(out.data.end(), pt, pt + msg->point_step);
      ++kept;
    }
    out.width = static_cast<uint32_t>(kept);
    out.row_step = out.width * out.point_step;
    pub_->publish(out);

    if (debug_markers_ && marker_pub_ && marker_pub_->get_subscription_count() > 0) {
      publishMarkers(msg->header);
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                         "[body_filter] %zu -> %zu points (%zu removed) in %.2f ms", n_in, kept, n_in - kept, ms);
  }

  void publishMarkers(const std_msgs::msg::Header& header)
  {
    visualization_msgs::msg::MarkerArray arr;
    int id = 0;
    for (const auto& sh : shapes_) {
      visualization_msgs::msg::Marker m;
      m.header = header;
      m.ns = "body_filter/" + sh.link;
      m.id = id++;
      m.action = sh.active ? visualization_msgs::msg::Marker::ADD : visualization_msgs::msg::Marker::DELETE;
      const Eigen::Isometry3d cloud_T_shape = sh.shape_T_cloud.inverse();
      m.pose = tf2::toMsg(cloud_T_shape);
      switch (sh.type) {
        case Shape::BOX:
          m.type = visualization_msgs::msg::Marker::CUBE;
          m.scale.x = 2 * sh.half.x(); m.scale.y = 2 * sh.half.y(); m.scale.z = 2 * sh.half.z();
          break;
        case Shape::SPHERE:
          m.type = visualization_msgs::msg::Marker::SPHERE;
          m.scale.x = m.scale.y = m.scale.z = 2 * sh.radius;
          break;
        case Shape::CYLINDER:
          m.type = visualization_msgs::msg::Marker::CYLINDER;
          m.scale.x = m.scale.y = 2 * sh.radius; m.scale.z = 2 * sh.half_len;
          break;
      }
      m.color.r = 1.0f; m.color.g = 0.4f; m.color.b = 0.0f; m.color.a = 0.4f;
      arr.markers.push_back(m);
    }
    marker_pub_->publish(arr);
  }

  std::vector<std::string> only_links_;
  std::set<std::string> links_;
  std::vector<Shape> shapes_;
  double tf_timeout_{0.2};
  bool debug_markers_{false};

  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1BodyFilterNode>());
  rclcpp::shutdown();
  return 0;
}
