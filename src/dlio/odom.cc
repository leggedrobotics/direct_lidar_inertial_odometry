/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.h"
#include "dlio/utils.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <queue>

#include "rclcpp/qos.hpp"

namespace {

template <typename PublisherPtrT>
inline bool hasSubscribers(const PublisherPtrT& pub) {
  return pub && pub->get_subscription_count() > 0;
}

// Minimal scope guard so every early return out of processPointCloud() still
// runs the scan-timestamp bookkeeping. prev_scan_stamp must advance once per
// consumed scan; if any exit path forgets it, the next scan asks integrateImu()
// for an ever-growing interval that eventually falls out of the IMU ring buffer
// and can never recover.
template <typename F>
class ScopeExit {
 public:
  explicit ScopeExit(F fn) : fn_(std::move(fn)) {}
  ~ScopeExit() { fn_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  F fn_;
};

template <typename F>
ScopeExit<F> makeScopeExit(F fn) {
  return ScopeExit<F>(std::move(fn));
}

struct NormalScatterStats {
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  int valid_normals = 0;
  double total_weight = 0.0;
};

inline void logTranslationSpectrumAlways(const rclcpp::Logger& logger,
                                         const Eigen::Vector3d& evals,
                                         const Eigen::Matrix3d& evecs) {
  RCLCPP_INFO(
      logger,
      "Translation observability from scan normal spread: eigvals(sorted asc) = [%.3f %.3f %.3f]",
      evals(0), evals(1), evals(2));

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d v = evecs.col(i).normalized();
    RCLCPP_INFO(
        logger,
        "  dir[%d] = [%.6f %.6f %.6f]",
        i,
        v.x(), v.y(), v.z());
  }
}

inline std::array<bool, 3> classifyWeakDirections(const Eigen::Vector3d& evals,
                                                  const double abs_threshold) {
  std::array<bool, 3> weak{{false, false, false}};
  for (int i = 0; i < 3; ++i) {
    const double lambda_i = evals(i);
    weak[i] = lambda_i <= abs_threshold;
  }
  return weak;
}

inline void sortEigenpairsAscending(const Eigen::Vector3d& evals_in,
                                    const Eigen::Matrix3d& evecs_in,
                                    Eigen::Vector3d& evals_out,
                                    Eigen::Matrix3d& evecs_out) {
  std::array<int, 3> idx{{0, 1, 2}};
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return evals_in(a) < evals_in(b);
  });

  for (int k = 0; k < 3; ++k) {
    evals_out(k) = evals_in(idx[k]);
    evecs_out.col(k) = evecs_in.col(idx[k]).normalized();
  }
}

inline double surfacePlanarityWeight(const Eigen::Vector3d& evals) {
  constexpr double kEps = 1e-12;
  const double lambda_max = std::max(evals(2), kEps);
  const double weight = (evals(1) - evals(0)) / lambda_max;
  if (!std::isfinite(weight)) {
    return 0.0;
  }
  return std::clamp(weight, 0.0, 1.0);
}

inline NormalScatterStats buildNormalScatterMatrix(const nano_gicp::CovarianceList& covs) {
  NormalScatterStats stats;

  for (const auto& cov4 : covs) {
    const Eigen::Matrix3d cov =
        0.5 * (cov4.block<3, 3>(0, 0) + cov4.block<3, 3>(0, 0).transpose());
    if (!cov.allFinite()) {
      continue;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) {
      continue;
    }

    const Eigen::Vector3d evals = eig.eigenvalues();
    const double weight = surfacePlanarityWeight(evals);
    if (weight <= 1e-6) {
      continue;
    }

    Eigen::Vector3d normal = eig.eigenvectors().col(0);
    if (!normal.allFinite()) {
      continue;
    }

    normal.normalize();
    stats.scatter.noalias() += weight * (normal * normal.transpose());
    stats.total_weight += weight;
    ++stats.valid_normals;
  }

  return stats;
}

}  // namespace

dlio::OdomNode::OdomNode() : Node("dlio_odom_node") {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  this->first_external_odom_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;

  // Reliable transport with bounded history to absorb bursts before callback queuing.
  const size_t lidar_qos_depth = this->pointcloud_queue_size_;
  auto qosLiDAR = rclcpp::QoS(rclcpp::KeepLast(lidar_qos_depth))
              .reliability(rclcpp::ReliabilityPolicy::Reliable)
              .durability(rclcpp::DurabilityPolicy::Volatile);
  lidar_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud", qosLiDAR,
      std::bind(&dlio::OdomNode::callbackPointCloud, this, std::placeholders::_1),
      lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // SensorDataQoS default expanded explicitly:
  // history=keep_last, depth=5, reliability=best_effort, durability=volatile.
  auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(200))
                    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
                    .durability(rclcpp::DurabilityPolicy::Volatile);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  // SubscriptionOptions default callback_group is nullptr (node default group).
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", imu_qos,
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->external_odom_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto external_odom_sub_opt = rclcpp::SubscriptionOptions();
  external_odom_sub_opt.callback_group = this->external_odom_cb_group;
  this->external_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
      "external_odom", 100,
      std::bind(&dlio::OdomNode::callbackExternalOdom, this, std::placeholders::_1),
      external_odom_sub_opt);

  this->reset_srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "dlio/reset",
      std::bind(&dlio::OdomNode::resetService, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(),
      this->reset_srv_cb_group_);

  this->map_reset_client_ = this->create_client<std_srvs::srv::Trigger>("dlio/reset_map");

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path_map", 1);
  this->path_odom_pub = this->create_publisher<nav_msgs::msg::Path>("path_odom", 1);
  this->path_map_prop_pub = this->create_publisher<nav_msgs::msg::Path>("path_map_prop", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);

  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  this->odom_map_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose_inverted", reliable_qos);
  this->odom_baselink_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose", reliable_qos);

  auto best_effort_qos = rclcpp::QoS(100)
                             .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                             .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
                             .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", best_effort_qos);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", best_effort_qos);
  this->deskewed_not_transformed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_not_transformed", best_effort_qos);
  this->deskewed_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_and_transformed_to_map", best_effort_qos);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  // Markers use reliable transport to avoid stale RViz artifacts from dropped DELETE updates.
  rclcpp::QoS marker_qos(rclcpp::KeepLast(50));
  marker_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  marker_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  marker_qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  this->pub_lin_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/velocity_linear", marker_qos);
  this->pub_ang_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/velocity_angular", marker_qos);
  this->pub_corr_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/correction", marker_qos);
  this->pub_degen_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "markers/degeneracy_directions", marker_qos);
  this->degen_status_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "degenerate", marker_qos);
  this->corr_marker_points_.reserve(
      2U * static_cast<std::size_t>(std::max(1, this->viz_corr_max_segments_)));

  this->pointcloud_worker_ = std::thread([this]{ pointCloudWorkerLoop(); });
  this->pub_worker_ = std::thread([this]{ workerLoop(); });

  {
    std::lock_guard<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.reserve(this->kMaxKeyframes);
    this->keyframe_timestamps.reserve(this->kMaxKeyframes);
    this->keyframe_normals.reserve(this->kMaxKeyframes);
    this->keyframe_transformations.reserve(this->kMaxKeyframes);
  }

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);
  this->scan_state = this->state;
  this->scan_state_prior = this->state;
  this->scan_state_stamp_ = 0.0;
  this->scan_state_prior_stamp_ = 0.0;
  this->live_state_stamp_ = 0.0;
  this->scan_state_valid_ = false;
  this->scan_state_prior_valid_ = false;

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  if (!this->imu_calibrate_) {
    this->captureInitialImuBaseline(
        Eigen::Vector3f(0.f, 0.f, static_cast<float>(std::abs(this->gravity_))),
        this->state.q,
        this->state.b.accel,
        this->state.b.gyro);
  }

  this->degen_prev_trans_dirs_map_[0] = Eigen::Vector3d::UnitX();
  this->degen_prev_trans_dirs_map_[1] = Eigen::Vector3d::UnitY();
  this->degen_prev_trans_dirs_map_[2] = Eigen::Vector3d::UnitZ();

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed = 0.0;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_self_.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_self_.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_self_.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_self_.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_self_.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_self_.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);
  this->gicp_self_.setSearchMethodSource(temp, true);
  this->gicp_self_.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  const float crop_size = static_cast<float>(this->crop_size_);
  const float vf_res = static_cast<float>(this->vf_res_);
  this->crop.setMin(Eigen::Vector4f(-crop_size, -crop_size, -crop_size, 1.0f));
  this->crop.setMax(Eigen::Vector4f(crop_size, crop_size, crop_size, 1.0f));

  this->voxel.setLeafSize(vf_res, vf_res, vf_res);

  {
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    this->metrics.spaciousness.push_back(0.);
    this->metrics.density.push_back(static_cast<float>(this->gicp_max_corr_dist_));
    // Start with a neutral model-deviation scale so adaptive gating
    // has a stable value before the first completed registration.
    this->metrics.motion_deviation.push_back(static_cast<float>(this->gicp_max_corr_dist_));
  }

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while (file != nullptr && fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  if (file != nullptr) {
    fclose(file);
  }

}

inline bool hasWeakDirection(const std::array<bool, 3>& weak) {
  return weak[0] || weak[1] || weak[2];
}

inline void logTranslationDegeneracy(const rclcpp::Logger& logger,
                                     const Eigen::Vector3d& evals,
                                     const Eigen::Matrix3d& evecs,
                                     const std::array<bool, 3>& weak) {
RCLCPP_WARN(
    logger,
    "Translation degeneracy from scan normal spread detected. eigvals(sorted asc) = [%.3f %.3f %.3f]",
    evals(0), evals(1), evals(2));

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d v = evecs.col(i).normalized();
    RCLCPP_WARN(
        logger,
        "  dir[%d] = [%.6f %.6f %.6f]  weak=%s",
        i,
        v.x(), v.y(), v.z(),
        weak[i] ? "true" : "false");
  }
}

dlio::OdomNode::~OdomNode() {
  this->requestStop();

  if (pointcloud_worker_.joinable()) pointcloud_worker_.join();
  if (pub_worker_.joinable()) pub_worker_.join();
  if (submap_future.valid()) submap_future.wait();
  if (debug_future_.valid()) debug_future_.wait();

}

void dlio::OdomNode::requestStop() {
  const bool already_stopping = stop_.exchange(true, std::memory_order_relaxed);
  if (!already_stopping) {
    RCLCPP_INFO(this->get_logger(),
                "\033[38;5;214m[SHUTDOWN] Odom node stopping. Draining workers and exiting...\033[0m");
  }

  {
    std::lock_guard<std::mutex> lk(this->main_loop_running_mutex);
    this->main_loop_running = false;
  }
  {
    std::lock_guard<std::mutex> lk(this->q_mtx_);
    this->q_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(this->pc_q_mtx_);
    this->pc_q_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    this->reset_requested_ = false;
    if (this->reset_in_progress_.exchange(false)) {
      this->reset_succeeded_ = false;
      this->reset_status_message_ = "Shutdown requested.";
    } else if (this->reset_status_message_.empty()) {
      this->reset_status_message_ = "Shutdown requested.";
    }
  }

  pc_q_cv_.notify_all();
  q_cv_.notify_all();
  cv_imu_stamp.notify_all();
  submap_build_cv.notify_all();
  reset_done_cv_.notify_all();
}

bool dlio::OdomNode::shouldStop() {
  if (stop_.load(std::memory_order_relaxed)) {
    return true;
  }

  const auto context = this->get_node_base_interface()->get_context();
  return !context || !context->is_valid();
}

void dlio::OdomNode::captureInitialImuBaseline(
    const Eigen::Vector3f& gravity_vec,
    const Eigen::Quaternionf& gravity_align_q,
    const Eigen::Vector3f& accel_bias,
    const Eigen::Vector3f& gyro_bias) {
  if (this->initial_imu_baseline_.valid) {
    return;
  }

  Eigen::Vector3f gravity_vec_safe = gravity_vec;
  if (!gravity_vec_safe.allFinite() || gravity_vec_safe.norm() <= 1e-6f) {
    gravity_vec_safe = Eigen::Vector3f(0.f, 0.f, static_cast<float>(std::abs(this->gravity_)));
  }

  this->initial_imu_baseline_.gravity_vec = gravity_vec_safe;
  this->initial_imu_baseline_.gravity_norm = gravity_vec_safe.norm();
  this->initial_imu_baseline_.gravity_align_q = gravity_align_q.normalized();
  this->initial_imu_baseline_.accel_bias = accel_bias;
  this->initial_imu_baseline_.gyro_bias = gyro_bias;
  this->initial_imu_baseline_.valid = true;

  RCLCPP_INFO(
      this->get_logger(),
      "Captured initial IMU baseline: |g|=%.6f accel_bias=[%.6f %.6f %.6f] gyro_bias=[%.6f %.6f %.6f]",
      this->initial_imu_baseline_.gravity_norm,
      this->initial_imu_baseline_.accel_bias.x(),
      this->initial_imu_baseline_.accel_bias.y(),
      this->initial_imu_baseline_.accel_bias.z(),
      this->initial_imu_baseline_.gyro_bias.x(),
      this->initial_imu_baseline_.gyro_bias.y(),
      this->initial_imu_baseline_.gyro_bias.z());
}

bool dlio::OdomNode::beginPendingReset() {
  std::lock_guard<std::mutex> lock(this->reset_mutex_);
  if (this->shouldStop() || this->reset_in_progress_.load() || !this->reset_requested_) {
    return false;
  }

  this->reset_requested_ = false;
  this->reset_in_progress_.store(true);
  this->reset_succeeded_ = false;
  this->reset_status_message_.clear();
  return true;
}

void dlio::OdomNode::finishPendingReset(bool success, const std::string& message) {
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    if (this->shouldStop()) {
      success = false;
      this->reset_status_message_ = "Shutdown requested.";
    } else {
      this->reset_status_message_ = message;
    }
    this->reset_in_progress_.store(false);
    this->reset_succeeded_ = success;
  }
  this->reset_done_cv_.notify_all();
}

void dlio::OdomNode::requestMapReset(const std::string& origin) {
  if (this->shouldStop()) {
    return;
  }

  if (!this->map_reset_client_) {
    return;
  }

  if (!this->map_reset_client_->service_is_ready()) {
    RCLCPP_WARN(this->get_logger(),
                "[RESET:%s] dlio/reset_map service not ready — map was NOT cleared.",
                origin.c_str());
    return;
  }

  auto map_req = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto map_future = this->map_reset_client_->async_send_request(map_req);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!this->shouldStop()) {
    if (map_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      auto map_res = map_future.get();
      if (map_res->success) {
        RCLCPP_INFO(this->get_logger(),
                    "\033[33m[RESET:%s] Map reset confirmed: %s\033[0m",
                    origin.c_str(), map_res->message.c_str());
      } else {
        RCLCPP_WARN(this->get_logger(),
                    "[RESET:%s] Map reset returned failure: %s",
                    origin.c_str(), map_res->message.c_str());
      }
      return;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
      RCLCPP_WARN(this->get_logger(),
                  "[RESET:%s] Map reset service call timed out after 3 s.",
                  origin.c_str());
      return;
    }
  }
}

bool dlio::OdomNode::triggerInternalReset(const std::string& reason) {
  bool owns_reset_request = false;
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    if (!this->initial_imu_baseline_.valid) {
      RCLCPP_ERROR(this->get_logger(),
                   "[RESET] Internal reset rejected: initial IMU baseline has not been captured yet.");
      return false;
    }

    if (this->reset_in_progress_.load()) {
      RCLCPP_WARN(this->get_logger(),
                  "[RESET] Internal reset requested while another reset is already in progress. "
                  "Executing worker-side reset immediately.");
    } else {
      owns_reset_request = true;
      this->reset_requested_ = false;
      this->reset_in_progress_.store(true);
      this->reset_succeeded_ = false;
      this->reset_status_message_.clear();
    }
  }

  this->degen_consecutive_hits_ = 0;

  RCLCPP_ERROR(this->get_logger(),
               "\033[31m[RESET] Internal self-reset triggered: %s\033[0m",
               reason.c_str());

  this->performReset();
  if (owns_reset_request) {
    this->requestMapReset("self");
  }
  return true;
}

void dlio::OdomNode::performReset() {
  // Called exclusively from pointCloudWorkerLoop() at a safe scan boundary.
  // The service thread is blocked on reset_done_cv_ for the duration of this function.
  RCLCPP_INFO(this->get_logger(),
              "\033[33m[RESET] ============ BEGIN SYSTEM RESET ============\033[0m");

  // -----------------------------------------------------------------------
  // Gap 3 fix (step 1/2): release main_loop_running BEFORE waiting on the
  // submap future, so pauseSubmapBuildIfNeeded() can unblock.
  // -----------------------------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(this->main_loop_running_mutex);
    this->main_loop_running = false;
  }
  this->submap_build_cv.notify_all();
  RCLCPP_INFO(this->get_logger(), "[RESET] main_loop_running cleared, submap_build_cv notified.");

  // -----------------------------------------------------------------------
  // Gap 3 fix (step 2/2): now safe to wait for the async submap build.
  // -----------------------------------------------------------------------
  if (this->submap_future.valid() &&
      this->submap_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    RCLCPP_INFO(this->get_logger(), "[RESET] Waiting for in-flight submap build to finish...");
    this->submap_future.wait();
    RCLCPP_INFO(this->get_logger(), "[RESET] Submap build joined.");
  }

  // -----------------------------------------------------------------------
  // Gap 4 fix: flush the publish queue so the pub_worker_ doesn't emit
  // stale odometry after the state is cleared.
  // -----------------------------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(this->q_mtx_);
    const std::size_t dropped = this->q_.size();
    this->q_.clear();
    if (dropped > 0) {
      RCLCPP_INFO(this->get_logger(),
                  "[RESET] Flushed %zu stale publish job(s) from publish queue.", dropped);
    }
  }
  this->q_cv_.notify_all();

  // -----------------------------------------------------------------------
  // IMU buffer and IMU-tracking state
  // -----------------------------------------------------------------------
  {
    // Prevent an in-flight IMU callback from repopulating timing state while
    // the reset clears it.
    std::lock_guard<std::mutex> imu_callback_lock(this->mtx_imu_callback_);
    {
      std::lock_guard<std::mutex> lk(this->mtx_imu);
      this->imu_buffer.clear();
    }
    this->imu_input_timestamps_.resetSequence();
    this->imu_integration_timestamps_.resetSequence();
    this->first_imu_stamp             = 0.0;
    this->imu_transform_ang_vel_prev_ = Eigen::Vector3f::Zero();
    this->first_imu_received          = false;
  }
  {
    std::unique_lock<std::mutex> lock(this->mtx_external_odom);
    this->first_external_odom_received = false;
    this->externalOdomPose.p    = Eigen::Vector3f::Zero();
    this->externalOdomPose.q    = Eigen::Quaternionf::Identity();
    this->prevExternalOdomPose  = this->externalOdomPose;
  }
  RCLCPP_INFO(this->get_logger(), "[RESET] IMU buffer cleared, IMU stamps zeroed, external odom state reset.");

  // -----------------------------------------------------------------------
  // Pose / velocity / bias — restored from the immutable startup baseline.
  // Mirror the IMU calibration initialization path, but reuse the saved
  // gravity vector and saved biases instead of estimating them again.
  // -----------------------------------------------------------------------
  const float gravity_norm = (this->initial_imu_baseline_.gravity_norm > 1e-6f)
      ? this->initial_imu_baseline_.gravity_norm
      : static_cast<float>(std::abs(this->gravity_));
  const float gravity_scalar =
      (this->gravity_ < 0.0) ? -gravity_norm : gravity_norm;

  Eigen::Vector3f reset_gravity_vec = this->initial_imu_baseline_.gravity_vec;
  if (!reset_gravity_vec.allFinite() || reset_gravity_vec.norm() <= 1e-6f) {
    reset_gravity_vec = Eigen::Vector3f(0.f, 0.f, std::abs(gravity_scalar));
  } else {
    reset_gravity_vec = reset_gravity_vec.normalized() * gravity_norm;
  }

  Eigen::Quaternionf reset_gravity_align_q = this->initial_imu_baseline_.gravity_align_q;
  if (this->gravity_align_) {
    reset_gravity_align_q = Eigen::Quaternionf::FromTwoVectors(
        reset_gravity_vec, Eigen::Vector3f(0.f, 0.f, gravity_scalar));
  }
  reset_gravity_align_q.normalize();
  this->gravity_ = gravity_scalar;

  {
    std::lock_guard<std::mutex> lk(this->geo.mtx);
    this->state.p               = Eigen::Vector3f::Zero();
    this->state.q               = reset_gravity_align_q;
    this->state.v.lin.b         = Eigen::Vector3f::Zero();
    this->state.v.lin.w         = Eigen::Vector3f::Zero();
    this->state.v.ang.b         = Eigen::Vector3f::Zero();
    this->state.v.ang.w         = Eigen::Vector3f::Zero();
    this->state.b.accel         = this->initial_imu_baseline_.accel_bias;
    this->state.b.gyro          = this->initial_imu_baseline_.gyro_bias;
    this->scan_state            = this->state;
    this->scan_state_prior      = this->state;
    this->scan_state_stamp_     = 0.0;
    this->scan_state_prior_stamp_ = 0.0;
    this->live_state_stamp_     = 0.0;
    this->scan_state_valid_     = false;
    this->scan_state_prior_valid_ = false;
    this->geo.first_opt_done    = false;
    this->geo.prev_p            = Eigen::Vector3f::Zero();
    this->geo.prev_q            = reset_gravity_align_q;
    this->geo.prev_vel          = Eigen::Vector3f::Zero();
    this->geo.dp                = 0.0;
    this->geo.dq_deg            = 0.0;
  }
  this->lidarPose.p = Eigen::Vector3f::Zero();
  this->lidarPose.q = reset_gravity_align_q;
  this->T       = Eigen::Matrix4f::Identity();
  this->T.block<3,3>(0,0) = reset_gravity_align_q.toRotationMatrix();
  this->T_prior = this->T;
  this->T_corr  = Eigen::Matrix4f::Identity();
  {
    std::lock_guard<std::mutex> lk(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest    = Eigen::Matrix4f::Identity();
    this->has_T_map_odom_latest = false;
  }
  RCLCPP_INFO(this->get_logger(),
              "[RESET] State restored from baseline: |g|=%.4f  "
              "accel_bias=[%.4f %.4f %.4f]  gyro_bias=[%.4f %.4f %.4f]",
              this->initial_imu_baseline_.gravity_norm,
              this->initial_imu_baseline_.accel_bias.x(),
              this->initial_imu_baseline_.accel_bias.y(),
              this->initial_imu_baseline_.accel_bias.z(),
              this->initial_imu_baseline_.gyro_bias.x(),
              this->initial_imu_baseline_.gyro_bias.y(),
              this->initial_imu_baseline_.gyro_bias.z());

  // -----------------------------------------------------------------------
  // Keyframes and submap
  // -----------------------------------------------------------------------
  {
    std::lock_guard<decltype(this->keyframes_mutex)> lk(this->keyframes_mutex);
    this->keyframes.clear();
    this->keyframe_timestamps.clear();
    this->keyframe_normals.clear();
    this->keyframe_transformations.clear();
  }
  this->num_processed_keyframes = 0;
  this->keyframe_convex.clear();
  this->keyframe_concave.clear();
  this->submap_kf_idx_curr.clear();
  this->submap_kf_idx_prev.clear();
  this->submap_cloud   = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_normals = nullptr;
  this->submap_kdtree  = nullptr;
  this->submap_hasChanged   = true;
  this->new_submap_is_ready = false;
  this->gicp.clearSource();
  this->gicp.clearTarget();
  this->gicp_temp.clearTarget();
  this->gicp_self_.clearSource();
  this->gicp_self_.clearTarget();
  RCLCPP_INFO(this->get_logger(), "[RESET] Keyframes, submap, and GICP caches cleared.");

  // -----------------------------------------------------------------------
  // Scan clouds
  // -----------------------------------------------------------------------
  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan  = std::make_shared<const pcl::PointCloud<PointType>>();

  // -----------------------------------------------------------------------
  // Timestamps, trajectory, and computation metrics
  // -----------------------------------------------------------------------
  this->first_scan_stamp = 0.0;
  this->prev_scan_stamp  = 0.0;
  this->scan_stamp       = 0.0;
  this->resetPointCloudTiming();
  this->elapsed_time     = 0.0;
  this->length_traversed = 0.0;
  this->trajectory.clear();
  {
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    this->comp_times.clear();
  }
  {
    std::lock_guard<std::mutex> lk(this->g_metrics_mutex);
    this->metrics.spaciousness.clear();
    this->metrics.density.clear();
    this->metrics.motion_deviation.clear();
    this->metrics.spaciousness.push_back(0.f);
    this->metrics.density.push_back(static_cast<float>(this->gicp_max_corr_dist_));
    this->metrics.motion_deviation.push_back(static_cast<float>(this->gicp_max_corr_dist_));
  }
  this->spaciousness_lpf_initialized_ = false;
  this->density_lpf_initialized_      = false;
  RCLCPP_INFO(this->get_logger(), "[RESET] Timestamps, trajectory, and metrics cleared.");

  // -----------------------------------------------------------------------
  // Re-initialisation flags
  // Biases already restored above; skip re-calibration.
  // -----------------------------------------------------------------------
  this->dlio_initialized   = false;
  this->first_valid_scan   = false;
  this->imu_calibrated     = true;   // keep baseline biases, no re-calib
  RCLCPP_INFO(this->get_logger(),
              "[RESET] Flags reset: dlio_initialized=false, first_valid_scan=false, "
              "imu_calibrated=true (biases retained from baseline).");

  // -----------------------------------------------------------------------
  // Visualisation state
  // -----------------------------------------------------------------------
  this->path_poses_.clear();
  this->path_odom_poses_.clear();
  this->path_map_prop_poses_.clear();
  this->kf_pose_ros.poses.clear();
  this->corr_marker_points_.clear();
  this->degen_info_.valid              = false;
  this->degen_consecutive_hits_        = 0;
  this->degen_recovery_start_stamp_    = 0.0;
  this->gicp_freeze_trials_latched_    = false;
  this->gicp_rematch_trials_latched_   = false;
  this->degen_prev_dirs_initialized_   = false;
  this->degen_prev_trans_dirs_map_[0]  = Eigen::Vector3d::UnitX();
  this->degen_prev_trans_dirs_map_[1]  = Eigen::Vector3d::UnitY();
  this->degen_prev_trans_dirs_map_[2]  = Eigen::Vector3d::UnitZ();

  // Publish empty paths so RViz clears immediately.
  const rclcpp::Time now = this->now();
  {
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp    = now;
    empty_path.header.frame_id = this->odom_frame;
    if (hasSubscribers(this->path_pub))          this->path_pub->publish(empty_path);
    if (hasSubscribers(this->path_odom_pub))     this->path_odom_pub->publish(empty_path);
    empty_path.header.frame_id = "dlio_map";
    if (hasSubscribers(this->path_map_prop_pub)) this->path_map_prop_pub->publish(empty_path);
  }
  {
    geometry_msgs::msg::PoseArray empty_poses;
    empty_poses.header.stamp    = now;
    empty_poses.header.frame_id = "dlio_map";
    if (hasSubscribers(this->kf_pose_pub)) this->kf_pose_pub->publish(empty_poses);
  }
  // Delete correction line marker.
  if (hasSubscribers(this->pub_corr_marker_)) {
    visualization_msgs::msg::Marker del;
    del.header.stamp    = now;
    del.header.frame_id = "dlio_map";
    del.ns     = "correction_lines";
    del.id     = 2;
    del.action = visualization_msgs::msg::Marker::DELETE;
    this->pub_corr_marker_->publish(del);
  }
  // Delete degeneracy markers.
  if (hasSubscribers(this->pub_degen_marker_)) {
    visualization_msgs::msg::MarkerArray del_arr;
    for (int i = 0; i < 3; ++i) {
      visualization_msgs::msg::Marker m;
      m.header.stamp    = now;
      m.header.frame_id = "dlio_map";
      m.ns     = "degeneracy_translation";
      m.id     = i;
      m.action = visualization_msgs::msg::Marker::DELETE;
      del_arr.markers.push_back(m);
    }
    this->pub_degen_marker_->publish(del_arr);
  }
  RCLCPP_INFO(this->get_logger(),
              "[RESET] Visualisation state cleared and empty paths/markers published.");

  // -----------------------------------------------------------------------
  // Scan geometry gate: drop incoming scans until one has enough 3D structure
  // in its local surface normals, ensuring the first registration after reset
  // is not started from a nearly planar scene.
  // -----------------------------------------------------------------------
  if (this->restart_gate_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "\033[33m[RESET] Geometry gate active "
                "(min_normal_scatter_eigenvalue=%.1f). Waiting for a non-degenerate scan...\033[0m",
                this->restart_gate_min_eigenvalue_);

    int dropped = 0;
    while (!this->shouldStop()) {
      PointCloudJob gate_job;
      {
        std::unique_lock<std::mutex> lk(this->pc_q_mtx_);
        this->pc_q_cv_.wait(lk, [this]{
          return this->shouldStop() || !this->pc_q_.empty();
        });
        if (this->shouldStop()) break;
        gate_job = std::move(this->pc_q_.front());
        this->pc_q_.pop_front();
      }

      if (gate_job.cloud_msg && !gate_job.ring_range_filtered) {
        this->filterPointCloudByRingRange(*gate_job.cloud_msg);
        gate_job.ring_range_filtered = true;
      }
      if (this->scanPassesGeometryGate(gate_job.cloud_msg)) {
        RCLCPP_INFO(this->get_logger(),
                    "\033[32m[RESET] Geometry gate passed after dropping %d scan(s). "
                    "Returning scan to queue for normal processing.\033[0m", dropped);
        {
          std::lock_guard<std::mutex> lk(this->pc_q_mtx_);
          this->pc_q_.push_front(std::move(gate_job));
        }
        this->pc_q_cv_.notify_one();
        break;
      }

      ++dropped;
      RCLCPP_WARN(this->get_logger(),
                  "[RESET] Geometry gate: scan #%d dropped (degenerate).", dropped);
    }

    if (this->shouldStop()) {
      this->finishPendingReset(false, "Node stopped during geometry gate.");
      return;
    }
  }

  // -----------------------------------------------------------------------
  // Signal completion to the waiting service thread.
  // -----------------------------------------------------------------------
  this->estimator_halted_.store(false);
  RCLCPP_INFO(this->get_logger(),
              "\033[32m[RESET] ============ ODOM RESET COMPLETE ============\033[0m");
  this->finishPendingReset(true, "Odom reset complete.");
}

bool dlio::OdomNode::scanPassesGeometryGate(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  // The gate and analyzeDegeneracyFromCurrentScan() feed the same
  // buildNormalScatterMatrix() and compare its smallest eigenvalue against a
  // threshold, so both must weight points the same way or the two thresholds
  // are in different units. The detector runs on this->gicp, which is never
  // configured and therefore uses nano_gicp's default PLANE. PLANE replaces
  // every point covariance with singular values (1, 1, 1e-3), so each point
  // contributes a full unit vote; NONE keeps the raw sample covariance, whose
  // planarity weight averaged 0.39 on the hiking2025 data. Running the gate
  // with NONE made restart_gate_min_eigenvalue ~2.5x stricter than the
  // detector that had just declared the scene recovered, which stalled one
  // restart for 787 scans (~79 s). Match the detector.
  this->gicp_self_.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);

  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc, *cloud);
  if (cloud->empty()) {
    return false;
  }

  // Remove points inside the ego-vehicle crop box.
  pcl::CropBox<PointType> crop_gate;
  const float crop_size = static_cast<float>(this->crop_size_);
  crop_gate.setNegative(true);
  crop_gate.setMin(Eigen::Vector4f(-crop_size, -crop_size, -crop_size, 1.0f));
  crop_gate.setMax(Eigen::Vector4f(crop_size, crop_size, crop_size, 1.0f));
  crop_gate.setInputCloud(cloud);
  crop_gate.filter(*cloud);

  // Voxel-downsample to keep the covariance computation fast.
  if (this->vf_use_) {
    pcl::VoxelGrid<PointType> vg;
    const float vf_res = static_cast<float>(this->vf_res_);
    vg.setLeafSize(vf_res, vf_res, vf_res);
    vg.setInputCloud(cloud);
    vg.filter(*cloud);
  }

  if (static_cast<int> (cloud->size()) < this->gicp_min_num_points_) {
    RCLCPP_WARN(this->get_logger(),
                "[GATE] Scan has only %zu points after filtering (min %d) — dropped.",
                cloud->size(), this->gicp_min_num_points_);
    return false;
  }

  this->gicp_self_.setInputSource(cloud);
  if (!this->gicp_self_.calculateSourceCovariances()) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Covariance computation failed — scan dropped.");
    return false;
  }

  const auto covs = this->gicp_self_.getSourceCovariances();
  if (!covs || covs->size() != cloud->size()) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Covariance cache is invalid — scan dropped.");
    return false;
  }

  const NormalScatterStats normal_stats = buildNormalScatterMatrix(*covs);

  if (normal_stats.valid_normals < 3 ||
      normal_stats.total_weight <= 0.0 ||
      !normal_stats.scatter.allFinite()) {
    RCLCPP_WARN(this->get_logger(),
                "[GATE] Normal diversity is ill-defined (valid_normals=%d, total_weight=%.1f) — scan dropped.",
                normal_stats.valid_normals, normal_stats.total_weight);
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(normal_stats.scatter);
  if (eig.info() != Eigen::Success) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Normal-scatter eigendecomposition failed — scan dropped.");
    return false;
  }

  const double min_eval = eig.eigenvalues().minCoeff();
  const bool passes = (min_eval >= this->restart_gate_min_eigenvalue_);

  RCLCPP_INFO(this->get_logger(),
              "[GATE] Scan normal diversity: eigvals=[%.1f %.1f %.1f]"
              "  valid_normals=%d"
              "  total_weight=%.1f"
              "  min=\033[38;5;214m%.1f\033[0m"
              "  threshold=\033[38;5;214m%.1f\033[0m"
              "  %s",
              eig.eigenvalues()(0), eig.eigenvalues()(1), eig.eigenvalues()(2),
              normal_stats.valid_normals, normal_stats.total_weight,
              min_eval, this->restart_gate_min_eigenvalue_,
              passes ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");


  return passes;
}

bool dlio::OdomNode::stateExceedsDegeneracyBounds(
    const State& candidate,
    std::string& reason) const {
  if (!candidate.p.allFinite() ||
      !candidate.q.coeffs().allFinite() ||
      candidate.q.norm() <= 1e-6f ||
      !candidate.v.lin.w.allFinite() ||
      !candidate.v.ang.b.allFinite() ||
      !candidate.v.ang.w.allFinite() ||
      !candidate.b.accel.allFinite() ||
      !candidate.b.gyro.allFinite()) {
    reason = "state contains a non-finite position, attitude, velocity, or bias";
    return true;
  }

  const double linear_speed = static_cast<double>(candidate.v.lin.w.norm());
  const double angular_speed = std::max(
      static_cast<double>(candidate.v.ang.b.norm()),
      static_cast<double>(candidate.v.ang.w.norm()));
  const double accel_bias = static_cast<double>(candidate.b.accel.cwiseAbs().maxCoeff());
  const double gyro_bias = static_cast<double>(candidate.b.gyro.cwiseAbs().maxCoeff());

  if (linear_speed >= this->degen_max_linear_speed_) {
    reason = "linear speed " + std::to_string(linear_speed) +
             " m/s reached/exceeds " + std::to_string(this->degen_max_linear_speed_) + " m/s";
    return true;
  }
  if (angular_speed >= this->degen_max_angular_speed_) {
    reason = "angular speed " + std::to_string(angular_speed) +
             " rad/s reached/exceeds " + std::to_string(this->degen_max_angular_speed_) + " rad/s";
    return true;
  }
  if (accel_bias >= this->degen_max_accel_bias_) {
    reason = "absolute accelerometer bias " + std::to_string(accel_bias) +
             " m/s^2 reached/exceeds " + std::to_string(this->degen_max_accel_bias_) + " m/s^2";
    return true;
  }
  if (gyro_bias >= this->degen_max_gyro_bias_) {
    reason = "absolute gyroscope bias " + std::to_string(gyro_bias) +
             " rad/s reached/exceeds " + std::to_string(this->degen_max_gyro_bias_) + " rad/s";
    return true;
  }

  return false;
}

// Scene-only degeneracy: does this scan constrain translation in all three
// directions? Depends purely on the current scan, so it keeps re-evaluating
// while the estimator is halted.
bool dlio::OdomNode::currentScanGeometryIsDegenerate(std::string& reason) const {
  if (!this->use_degeneracy_) {
    return false;
  }

  if (!this->degen_info_.valid) {
    reason = "scan degeneracy analysis is invalid";
    return true;
  }

  if (hasWeakDirection(this->degen_info_.weak_trans)) {
    reason = "scan normal-spread spectrum has a weak translation direction";
    return true;
  }

  return false;
}

bool dlio::OdomNode::currentScanIsDegenerate(
    const State& candidate,
    std::string& reason) const {
  if (this->currentScanGeometryIsDegenerate(reason)) {
    return true;
  }
  if (!this->use_degeneracy_) {
    return false;
  }

  return this->stateExceedsDegeneracyBounds(candidate, reason);
}

void dlio::OdomNode::enterDegenerateHalt(const std::string& reason) {
  const bool was_halted = this->estimator_halted_.exchange(true);

  // Do not allow jobs already computed before the halt to publish stale poses.
  {
    std::lock_guard<std::mutex> lock(this->q_mtx_);
    this->q_.clear();
  }
  this->q_cv_.notify_all();

  if (!was_halted) {
    RCLCPP_ERROR(
        this->get_logger(),
        "\033[1;31m[DEGEN] Estimator halted; pose/TF publication disabled: %s\033[0m",
        reason.c_str());
  }
}

// Advance the deskew anchor to the scan that was just consumed. Monotonic and
// idempotent, so it is safe to call from a scope guard on top of the explicit
// assignments already made by initializeInputTarget() and processPointCloud().
void dlio::OdomNode::advancePrevScanStamp() {
  if (!this->first_valid_scan) {
    // performReset() clears prev_scan_stamp/scan_stamp on purpose; the
    // re-initialisation path owns the first post-reset stamp.
    return;
  }
  if (!std::isfinite(this->scan_stamp) || this->scan_stamp <= 0.0) {
    return;
  }
  if (this->scan_stamp > this->prev_scan_stamp) {
    this->prev_scan_stamp = this->scan_stamp;
  }
}

// Oldest measurement still held by the IMU ring buffer, or -1 when empty.
// integrateImu() cannot serve a start time older than this.
double dlio::OdomNode::oldestImuStamp() {
  std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
  if (this->imu_buffer.empty()) {
    return -1.0;
  }
  return this->imu_buffer.back().stamp;
}

void dlio::OdomNode::processHaltedScan(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {
  if (!pc || !this->use_degeneracy_) {
    return;
  }

  // Continue evaluating incoming scans without running GICP or publishing.
  // getScanFromROS()/preprocessPoints() also provide the same scan-time IMU
  // prior used by the normal estimator, so speed checks remain live while the
  // estimator is halted.
  this->getScanFromROS(pc);
  if (!this->original_scan || this->original_scan->empty()) {
    return;
  }

  // A halted estimator must not call preprocessPoints() here. That path runs
  // full per-point deskew using prev_scan_stamp. If getNextPose() halted the
  // estimator, prev_scan_stamp is intentionally the last accepted scan and
  // can be far behind the current bag time; repeatedly trying that interval
  // would produce the same deskew failure for every recovery scan.
  //
  // For the health check, a rigidly transformed, non-deskewed cloud is enough:
  // normal-spread degeneracy is invariant to the missing per-point motion
  // compensation, and no estimator state is advanced while halted.
  this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();
  this->T_prior = this->T;
  auto halted_scan = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(
      *this->original_scan, *halted_scan,
      this->T * this->extrinsics.baselink2lidar_T);
  this->deskewed_scan = halted_scan;
  this->deskew_status = false;
  this->deskew_size = 0;

  if (this->vf_use_) {
    auto filtered_scan = std::make_shared<pcl::PointCloud<PointType>>();
    this->voxel.setInputCloud(halted_scan);
    this->voxel.filter(*filtered_scan);
    this->current_scan = filtered_scan;
  } else {
    this->current_scan = halted_scan;
  }

  // Keep the deskew anchor moving even while the estimator is stopped, so the
  // first scan accepted after a restart asks for a one-period IMU interval
  // rather than the whole halt. performReset() clears it before that scan.
  if (this->scan_stamp > this->prev_scan_stamp) {
    this->prev_scan_stamp = this->scan_stamp;
  }

  if (!this->current_scan ||
      static_cast<int>(this->current_scan->points.size()) <= this->gicp_min_num_points_) {
    this->degen_recovery_start_stamp_ = 0.0;
    return;
  }

  this->setInputSource();
  this->analyzeDegeneracyFromCurrentScan(this->T_prior);
  this->publishDegeneracyMarkers(this->scan_header_stamp);

  // Recovery is gated on scan geometry ONLY, never on the state bounds.
  // IMU propagation and the observer are both stopped while halted, so
  // this->state is frozen at whatever value triggered the halt. If the halt
  // came from a state bound (speed/bias), testing that frozen value here can
  // never pass: it reports the identical number for every subsequent scan and
  // the estimator stays halted forever. Observed as 659 consecutive
  // "linear speed 10.496372 m/s" holds on the hiking2025 bag.
  // The state does not need defending here anyway -- performReset() discards
  // it and restores the IMU baseline, and the geometry gate then decides
  // which scan restarts the estimator.
  std::string reason;
  const bool degenerate = this->currentScanGeometryIsDegenerate(reason);

  this->publishDegeneracyStatus(degenerate);
  if (degenerate) {
    this->degen_recovery_start_stamp_ = 0.0;
    ++this->degen_consecutive_hits_;
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "[DEGEN] Recovery hold reset by scan at %.9f: %s",
        this->scan_stamp, reason.c_str());
    return;
  }

  this->degen_consecutive_hits_ = 0;
  if (this->degen_recovery_start_stamp_ <= 0.0) {
    this->degen_recovery_start_stamp_ = this->scan_stamp;
    RCLCPP_INFO(
        this->get_logger(),
        "[DEGEN] Non-degenerate recovery window started at %.9f; requiring %.3f s.",
        this->degen_recovery_start_stamp_, this->degen_recovery_time_);
  }

  const double stable_duration = this->scan_stamp - this->degen_recovery_start_stamp_;
  if (stable_duration < this->degen_recovery_time_) {
    return;
  }

  // performReset() owns the restart boundary and its geometry gate. Requeue
  // this healthy scan so the gate can consume it and normal processing resumes
  // only after reset completion.
  {
    std::lock_guard<std::mutex> lock(this->pc_q_mtx_);
    this->pc_q_.push_front(PointCloudJob{pc, true});
  }
  this->pc_q_cv_.notify_one();

  RCLCPP_INFO(
      this->get_logger(),
      "[DEGEN] %.3f s without degeneracy detected. Attempting estimator restart.",
      stable_duration);
  if (!this->triggerInternalReset("degeneracy cleared for the configured recovery window")) {
    std::lock_guard<std::mutex> lock(this->pc_q_mtx_);
    if (!this->pc_q_.empty() && this->pc_q_.front().cloud_msg == pc) {
      this->pc_q_.pop_front();
    }
  }
}

void dlio::OdomNode::resetService(
    std::shared_ptr<std_srvs::srv::Trigger::Request>,  // NOLINT(performance-unnecessary-value-param)
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {  // NOLINT(performance-unnecessary-value-param)
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);

    if (this->shouldStop()) {
      res->success = false;
      res->message = "Reset rejected: node is shutting down.";
      return;
    }

    if (!this->initial_imu_baseline_.valid) {
      res->success = false;
      res->message = "Reset rejected: initial IMU baseline has not been captured yet.";
      return;
    }

    if (this->reset_in_progress_.load()) {
      res->success = false;
      res->message = "Reset rejected: another reset is already in progress.";
      return;
    }

    if (this->reset_requested_) {
      res->success = false;
      res->message = "Reset rejected: a reset request is already pending.";
      return;
    }

    this->reset_requested_ = true;
  }

  if (!this->beginPendingReset()) {
    res->success = false;
    res->message = this->shouldStop()
        ? "Reset rejected: node is shutting down."
        : "Reset rejected: failed to transition reset request into the pending state.";
    return;
  }

  RCLCPP_INFO(this->get_logger(), "\033[33m[RESET] Request accepted. Notifying worker thread...\033[0m");

  // Wake both worker waits so they see reset_in_progress_ immediately.
  this->pc_q_cv_.notify_all();
  this->cv_imu_stamp.notify_all();

  // Block until the worker thread calls finishPendingReset().
  {
    std::unique_lock<std::mutex> lock(this->reset_mutex_);
    this->reset_done_cv_.wait(lock, [this]{
      return this->shouldStop() || !this->reset_in_progress_.load();
    });
  }

  if (this->shouldStop()) {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = false;
    res->message = this->reset_status_message_.empty()
        ? "Reset aborted: node is shutting down."
        : this->reset_status_message_;
    return;
  }

  RCLCPP_INFO(this->get_logger(), "\033[33m[RESET] Odom reset done. Triggering map reset...\033[0m");
  this->requestMapReset("service");

  if (this->shouldStop()) {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = false;
    res->message = this->reset_status_message_.empty()
        ? "Reset aborted: node is shutting down."
        : this->reset_status_message_;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = this->reset_succeeded_;
    res->message = this->reset_status_message_;
  }

  RCLCPP_INFO(this->get_logger(),
              "\033[32m[RESET] Full system reset complete. success=%s msg=\"%s\"\033[0m",
              res->success ? "true" : "false", res->message.c_str());
}

void dlio::OdomNode::enqueuePublish(
    pcl::PointCloud<PointType>::ConstPtr cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    double scanStamp,
    const Eigen::Vector3f& state_p_scan,
    const Eigen::Quaternionf& state_q_scan,
    const Eigen::Vector3f& state_vlin_b_scan,
    const Eigen::Vector3f& state_vang_b_scan) {

  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    return;
  }

  PubJob job;
  job.cloud = std::move(cloud);
  job.T_cloud = T_cloud;
  job.T_all = T_all;
  job.scan_header_stamp = this->scan_header_stamp;
  job.scanStamp = scanStamp;

  // Store the odom-state snapshot from the same scan-time reference as T_all/T_cloud.
  job.state_p_scan = state_p_scan;
  job.state_q_scan = state_q_scan.normalized();
  job.state_vlin_b_scan = state_vlin_b_scan;
  job.state_vang_b_scan = state_vang_b_scan;

  {
    std::lock_guard<std::mutex> lk(q_mtx_);
    if (!this->estimator_halted_.load(std::memory_order_relaxed)) {
      q_.push_back(std::move(job));
    }
  }
  q_cv_.notify_one();
}

void dlio::OdomNode::workerLoop() {
  while (!this->shouldStop()) {
    PubJob job;
    {
      std::unique_lock<std::mutex> lk(q_mtx_);
      q_cv_.wait(lk, [this]{ return this->shouldStop() || !q_.empty(); });
      if (this->shouldStop()) break;
      job = std::move(q_.front());
      q_.pop_front();
    }

    if (this->estimator_halted_.load(std::memory_order_relaxed)) {
      continue;
    }

    publishToROS(job.cloud,
                 job.T_cloud,
                 job.T_all,
                 job.scanStamp,
                 job.state_p_scan,
                 job.state_q_scan,
                 job.state_vlin_b_scan,
                 job.state_vang_b_scan);
  }
}
void dlio::OdomNode::enqueuePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {
  {
    std::lock_guard<std::mutex> lk(pc_q_mtx_);

    // Keep queue bounded; if overloaded, drop the oldest scan and keep recent measurements.
    while (pc_q_.size() >= this->pointcloud_queue_size_) {
      pc_q_.pop_front();
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Pointcloud queue full. Dropping oldest scan.");
    }

    pc_q_.push_back(PointCloudJob{pc, false});
  }
  pc_q_cv_.notify_one();
}

std::size_t dlio::OdomNode::filterPointCloudByRingRange(
    sensor_msgs::msg::PointCloud2& pc) {
  if (!this->ring_range_filter_enabled_ || pc.height == 0 || pc.width == 0) {
    return 0;
  }

  const dlio::RingRangeFilterResult result =
      dlio::filterPointCloudByRingRange(pc, this->ring_range_squared_);
  if (!result.supported_layout) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Ring range filter requires little-endian FLOAT32 x/y/z and UINT16 ring fields; "
        "leaving this cloud unfiltered.");
    return 0;
  }
  if (!this->ring_range_filter_reported_) {
    RCLCPP_INFO(
        this->get_logger(),
        "Per-ring LiDAR range filter active: input=%zu, removed=%zu, output=%zu points.",
        result.input_points, result.removedPoints(), result.output_points);
    this->ring_range_filter_reported_ = true;
  }
  return result.removedPoints();
}

void dlio::OdomNode::pointCloudWorkerLoop() {
  auto required_imu_time_from_cloud =
      [](const sensor_msgs::msg::PointCloud2::SharedPtr& pc) -> double {
        if (!pc) {
          return 0.0;
        }

        const double header_time = rclcpp::Time(pc->header.stamp).seconds();

        bool has_timestamp = false;
        bool has_t = false;
        bool has_time = false;
        for (const auto& field : pc->fields) {
          if (field.name == "timestamp") {
            has_timestamp = true;
          } else if (field.name == "t") {
            has_t = true;
          } else if (field.name == "time") {
            has_time = true;
          }
        }

        // No per-point timing field: fall back to header stamp.
        if (!has_timestamp && !has_t && !has_time) {
          return header_time;
        }

        pcl::PointCloud<PointType> cloud;
        pcl::fromROSMsg(*pc, cloud);
        if (cloud.empty()) {
          return header_time;
        }

        // RoboSense / Hesai / Livox-style timestamp field
        if (has_timestamp) {
          bool have_first = false;
          bool use_ns = false;

          for (const auto& pt : cloud.points) {
            const double t = pt.timestamp;
            if (!std::isfinite(t)) {
              continue;
            }
            use_ns = (t > 1e14);  // same convention already used in your sensor detection
            have_first = true;
            break;
          }

          if (!have_first) {
            return header_time;
          }

          double scan_end = -std::numeric_limits<double>::infinity();
          for (const auto& pt : cloud.points) {
            double t = pt.timestamp;
            if (!std::isfinite(t)) {
              continue;
            }
            if (use_ns) {
              t *= 1e-9;
            }
            scan_end = std::max(scan_end, t);
          }

          return std::isfinite(scan_end) ? scan_end : header_time;
        }

        // Ouster-style relative nanoseconds
        if (has_t) {
          double max_rel_ns = 0.0;
          for (const auto& pt : cloud.points) {
            const double t = pt.t;
            if (!std::isfinite(t)) {
              continue;
            }
            max_rel_ns = std::max(max_rel_ns, t);
          }
          return header_time + 1e-9 * max_rel_ns;
        }

        // Velodyne-style relative seconds
        double max_rel_s = 0.0;
        for (const auto& pt : cloud.points) {
          const double t = pt.time;
          if (!std::isfinite(t)) {
            continue;
          }
          max_rel_s = std::max(max_rel_s, t);
        }
        return header_time + max_rel_s;
      };

  while (!this->shouldStop()) {
    PointCloudJob job;
    {
      std::unique_lock<std::mutex> lk(pc_q_mtx_);
      // Gap 2 fix: predicate now also wakes on reset_in_progress_ so the worker
      // is not left blocked while the service thread waits for it.
      pc_q_cv_.wait(lk, [this]{
        return this->shouldStop()
            || this->reset_in_progress_.load()
            || !pc_q_.empty();
      });

      if (this->shouldStop()) {
        break;
      }

      // Gap 2 fix: if a reset arrived, drain the queue (while we hold the lock)
      // and jump to performReset() before touching any scan data.
      if (this->reset_in_progress_.load()) {
        const std::size_t dropped = pc_q_.size();
        pc_q_.clear();
        lk.unlock();
        if (dropped > 0) {
          RCLCPP_INFO(this->get_logger(),
                      "\033[33m[RESET] Worker: drained %zu queued scan(s) before reset.\033[0m",
                      dropped);
        }
        this->performReset();
        continue;
      }

      job = std::move(pc_q_.front());
      pc_q_.pop_front();
    }

    if (job.cloud_msg && !job.ring_range_filtered) {
      this->filterPointCloudByRingRange(*job.cloud_msg);
      job.ring_range_filtered = true;
    }

    // Wait here, before any pointcloud processing begins.
    const double required_imu_time = required_imu_time_from_cloud(job.cloud_msg);

    {
      std::unique_lock<decltype(this->mtx_imu)> imu_lock(this->mtx_imu);
      // Gap 2 fix: also wake on reset_in_progress_ so clearing imu_buffer
      // during reset doesn't leave this wait stuck forever.
      this->cv_imu_stamp.wait(imu_lock, [this, required_imu_time]{
        return this->shouldStop()
            || this->reset_in_progress_.load()
            || (!this->imu_buffer.empty() &&
                this->imu_buffer.front().stamp >= required_imu_time);
      });
    }

    if (this->shouldStop()) {
      break;
    }

    // Reset arrived while we were waiting for IMU: drop the dequeued scan and reset.
    if (this->reset_in_progress_.load()) {
      RCLCPP_INFO(this->get_logger(),
                  "\033[33m[RESET] Worker: reset detected after IMU wait — dropping current scan.\033[0m");
      this->performReset();
      continue;
    }

    this->processPointCloud(job.cloud_msg);
  }
}

void dlio::OdomNode::getParams() {

  // Version
  dlio::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  dlio::declare_param(this, "frames/odom", this->odom_frame, "dlio_odom");
  dlio::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  dlio::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  dlio::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  dlio::declare_param(this, "pointcloud/deskew", this->deskew_, true);
  dlio::declare_param(this, "pointcloud/queueSize", this->pointcloud_queue_size_, 5);
  if (this->pointcloud_queue_size_ < 1) {
    RCLCPP_WARN(this->get_logger(), "pointcloud/queueSize must be >= 1. Falling back to 1.");
    this->pointcloud_queue_size_ = 1;
  }

  // Input timing diagnostic. The default accepts a 100 ms LiDAR period with
  // 20 ms of jitter, while preserving the incoming cloud unchanged.
  dlio::declare_param(this, "pointcloud/timing/enabled",
                      this->pointcloud_timing_enabled_, true);
  dlio::declare_param(this, "pointcloud/timing/expectedPeriod",
                      this->pointcloud_expected_period_, 0.1);
  dlio::declare_param(this, "pointcloud/timing/tolerance",
                      this->pointcloud_period_tolerance_, 0.02);
  if (this->pointcloud_expected_period_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "pointcloud/timing/expectedPeriod must be > 0. Falling back to 0.1 s.");
    this->pointcloud_expected_period_ = 0.1;
  }
  if (this->pointcloud_period_tolerance_ < 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "pointcloud/timing/tolerance must be >= 0. Falling back to 0.02 s.");
    this->pointcloud_period_tolerance_ = 0.02;
  }

  // Gravity
  dlio::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  dlio::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  dlio::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  dlio::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  dlio::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  dlio::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  dlio::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  dlio::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Per-ring maximum range filter. The default table is indexed by the
  // zero-based ring values used in the PointCloud2 message.
  dlio::declare_param(this, "odom/preprocessing/ringRangeFilter/enabled",
                      this->ring_range_filter_enabled_, false);
  std::vector<double> ring_ranges_m;
  dlio::declare_param(this, "odom/preprocessing/ringRangeFilter/maxRanges",
                      ring_ranges_m, dlio::defaultRingRangesMeters());
  try {
    this->ring_range_squared_ = dlio::squaredRingRanges(ring_ranges_m);
  } catch (const std::invalid_argument& error) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid odom/preprocessing/ringRangeFilter/maxRanges: %s. "
                 "Disabling ring range filtering.",
                 error.what());
    this->ring_range_filter_enabled_ = false;
  }

  // Voxel Grid Filter
  dlio::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  dlio::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  dlio::declare_param(this, "adaptive", this->adaptive_params_, true);
  dlio::declare_param(this, "adaptive/spaciousness/min",    this->adaptive_sp_min_,          0.5f);
  dlio::declare_param(this, "adaptive/spaciousness/max",    this->adaptive_sp_max_,          5.0f);
  dlio::declare_param(this, "adaptive/density/factor_min",  this->adaptive_den_factor_min_,  0.5f);
  dlio::declare_param(this, "adaptive/density/factor_max",  this->adaptive_den_factor_max_,  2.0f);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(static_cast<float>(baselink2imu_t[0]),
                    static_cast<float>(baselink2imu_t[1]),
                    static_cast<float>(baselink2imu_t[2]));
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  dlio::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(static_cast<float>(baselink2lidar_t[0]),
                    static_cast<float>(baselink2lidar_t[1]),
                    static_cast<float>(baselink2lidar_t[2]));
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  dlio::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  dlio::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  dlio::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  dlio::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);
  dlio::declare_param(this, "odom/deskew/maxLookback", this->deskew_max_lookback_, 0.5);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  dlio::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  dlio::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  dlio::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  dlio::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  dlio::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel = Eigen::Vector3f(static_cast<float>(prior_accel_bias[0]),
                                          static_cast<float>(prior_accel_bias[1]),
                                          static_cast<float>(prior_accel_bias[2]));
    this->state.b.gyro = Eigen::Vector3f(static_cast<float>(prior_gyro_bias[0]),
                                         static_cast<float>(prior_gyro_bias[1]),
                                         static_cast<float>(prior_gyro_bias[2]));
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  dlio::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  dlio::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  dlio::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  dlio::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  dlio::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);
  dlio::declare_param(this, "odom/gicp/freezeTrialTriggerTranslation",
                      this->gicp_freeze_trial_trigger_translation_, 0.0);

  // Geometric Observer
  // --- Orientation layer (Eq. 3): Kq ≈ 2*c1 (attitude rate), Kgb ≈ c2 (gyro-bias rate) ---
  dlio::declare_param(this, "odom/geo/Kq",              this->geo_Kq_,              6.7);  // ~2/τq, τq≈0.30 s
  dlio::declare_param(this, "odom/geo/Kgb",             this->geo_Kgb_,             2.0);  // ~0.3–1.0 * c1; pick ~0.5*c1

  // --- Translation layer (Eq. 15): Kp≈ω_n^2, Kv≈2ζω_n, Kab≈K1 ---
  dlio::declare_param(this, "odom/geo/Kp",              this->geo_Kp_,              2.25); // ω_n≈1.5 s^-1  => ω_n^2
  dlio::declare_param(this, "odom/geo/Kv",              this->geo_Kv_,              3.0);  // 2 ζ ω_n with ζ≈1
  dlio::declare_param(this, "odom/geo/Kab",             this->geo_Kab_,             0.10); // conservative accel-bias adaption

  // --- Bias anti-windup clamps (pick from your IMU datasheet ranges) ---
  dlio::declare_param(this, "odom/geo/abias_max",       this->geo_abias_max_,       1.5);  // [m/s^2]
  dlio::declare_param(this, "odom/geo/gbias_max",       this->geo_gbias_max_,       0.30); // [rad/s]

  // Visualization (velocity markers)
  dlio::declare_param(this, "odom/debug/enabled",            this->debug_enabled_,        false);

  dlio::declare_param(this, "viz/vel_marker/enabled",        this->viz_vel_markers_,      true);
  dlio::declare_param(this, "viz/vel_marker/scale_lin",      this->viz_lin_gain_,         0.5);   // arrow length gain
  dlio::declare_param(this, "viz/vel_marker/ang/radius_gain",this->viz_ang_radius_gain_,  0.20);
  dlio::declare_param(this, "viz/vel_marker/ang/r_min",      this->viz_ang_radius_min_,   0.10);
  dlio::declare_param(this, "viz/vel_marker/ang/r_max",      this->viz_ang_radius_max_,   0.50);
  dlio::declare_param(this, "viz/vel_marker/thickness",      this->viz_disc_thickness_,   0.03);
  dlio::declare_param(this, "viz/vel_marker/lifetime",       this->viz_marker_lifetime_,  0.10);
  dlio::declare_param(this, "viz/corr_marker/enabled",       this->viz_corr_marker_,      true);
  dlio::declare_param(this, "viz/corr_marker/scale",         this->viz_corr_gain_,        1.0);
  dlio::declare_param(this, "viz/corr_marker/line_width",    this->viz_corr_line_width_,  0.008);
  dlio::declare_param(this, "viz/corr_marker/max_segments",  this->viz_corr_max_segments_,2000);
  dlio::declare_param(this, "viz/corr_marker/lifetime",      this->viz_corr_lifetime_,    0.0);

  // Translation-only degeneracy analysis from the raw scan normal-spread
  // matrix. Weak directions are flagged by absolute thresholds on its
  // eigenvalues.
  dlio::declare_param(this, "odom/gicp/degeneracy/enabled",
                      this->use_degeneracy_, true);
  dlio::declare_param(this, "odom/gicp/degeneracy/trans_eig_abs_threshold",
                      this->degen_trans_eig_abs_thresh_, 200.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/recovery_time",
                      this->degen_recovery_time_, 3.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/max_linear_speed",
                      this->degen_max_linear_speed_, 8.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/max_angular_speed",
                      this->degen_max_angular_speed_, 4.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/max_accel_bias",
                      this->degen_max_accel_bias_, 5.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/max_gyro_bias",
                      this->degen_max_gyro_bias_, 0.5);

  // Restart geometry gate: after a reset, incoming scans are checked via
  // local-normal diversity before the system re-initializes. Scans that fail
  // (weakest normal-scatter eigenvalue below threshold) are dropped.
  dlio::declare_param(this, "odom/restart/geometry_gate/enabled",
                      this->restart_gate_enabled_, true);
  dlio::declare_param(this, "odom/restart/geometry_gate/min_eigenvalue",
                      this->restart_gate_min_eigenvalue_, 50.0);

  dlio::declare_param(this, "viz/degeneracy_marker/enabled", this->viz_degen_marker_, true);
  dlio::declare_param(this, "viz/degeneracy_marker/trans_scale", this->viz_degen_trans_scale_, 0.75);
  dlio::declare_param(this, "viz/degeneracy_marker/shaft_diameter", this->viz_degen_shaft_diam_, 0.03);
  dlio::declare_param(this, "viz/degeneracy_marker/head_diameter", this->viz_degen_head_diam_, 0.06);
  dlio::declare_param(this, "viz/degeneracy_marker/head_length", this->viz_degen_head_len_, 0.10);
  dlio::declare_param(this, "viz/degeneracy_marker/lifetime", this->viz_degen_lifetime_, 0.0);

  if (this->viz_corr_gain_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/scale must be >= 0. Clamping to 0.");
    this->viz_corr_gain_ = 0.0;
  }
  if (this->viz_corr_line_width_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/line_width must be > 0. Clamping to 0.01.");
    this->viz_corr_line_width_ = 0.01;
  }
  if (this->viz_corr_max_segments_ < 1) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/max_segments must be >= 1. Clamping to 1.");
    this->viz_corr_max_segments_ = 1;
  }
  if (this->viz_corr_lifetime_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/lifetime must be >= 0. Clamping to 0.");
    this->viz_corr_lifetime_ = 0.0;
  }

  if (this->degen_trans_eig_abs_thresh_ < 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/trans_eig_abs_threshold must be >= 0. Clamping to 0.");
    this->degen_trans_eig_abs_thresh_ = 0.0;
  }
  if (this->deskew_max_lookback_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/deskew/maxLookback must be > 0. Clamping to 0.5 s.");
    this->deskew_max_lookback_ = 0.5;
  }
  if (this->degen_recovery_time_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/recovery_time must be > 0. Clamping to 3.0 s.");
    this->degen_recovery_time_ = 3.0;
  }
  if (this->degen_max_linear_speed_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/max_linear_speed must be > 0. Clamping to 8.0 m/s.");
    this->degen_max_linear_speed_ = 8.0;
  }
  if (this->degen_max_angular_speed_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/max_angular_speed must be > 0. Clamping to 4.0 rad/s.");
    this->degen_max_angular_speed_ = 4.0;
  }
  if (this->degen_max_accel_bias_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/max_accel_bias must be > 0. Clamping to 5.0 m/s^2.");
    this->degen_max_accel_bias_ = 5.0;
  }
  if (this->degen_max_gyro_bias_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/max_gyro_bias must be > 0. Clamping to 0.5 rad/s.");
    this->degen_max_gyro_bias_ = 0.5;
  }
  if (this->viz_degen_trans_scale_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/trans_scale must be > 0. Clamping to 0.75.");
    this->viz_degen_trans_scale_ = 0.75;
  }
  if (this->viz_degen_shaft_diam_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/shaft_diameter must be > 0. Clamping to 0.03.");
    this->viz_degen_shaft_diam_ = 0.03;
  }
  if (this->viz_degen_head_diam_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/head_diameter must be > 0. Clamping to 0.06.");
    this->viz_degen_head_diam_ = 0.06;
  }
  if (this->viz_degen_head_len_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/head_length must be > 0. Clamping to 0.10.");
    this->viz_degen_head_len_ = 0.10;
  }
  if (this->viz_degen_lifetime_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/lifetime must be >= 0. Clamping to 0.");
    this->viz_degen_lifetime_ = 0.0;
  }

}
void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << '\n'
            << "+-------------------------------------------------------------------+\n";
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << '\n';
  std::cout << "+-------------------------------------------------------------------+\n";

}

void dlio::OdomNode::publishToROS(
    const pcl::PointCloud<PointType>::ConstPtr& cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    const double scanStamp,
    const Eigen::Vector3f& state_p_scan,
    const Eigen::Quaternionf& state_q_scan,
    const Eigen::Vector3f& state_vlin_b_scan,
    const Eigen::Vector3f& state_vang_b_scan)
{
  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    return;
  }

  // Build an exact scan-time timestamp once and use it everywhere below.
  const uint64_t nsec = static_cast<uint64_t>(scanStamp * 1e9);
  builtin_interfaces::msg::Time scan_stamp_msg;
  scan_stamp_msg.sec = static_cast<int32_t>(nsec / 1000000000ULL);
  scan_stamp_msg.nanosec = static_cast<uint32_t>(nsec % 1000000000ULL);
  const rclcpp::Time scan_time(scan_stamp_msg);

  // ---------------------------------------------------------------------------
  // dlio_map <-> base_link at scan time
  //
  // T_all is the scan-time transform dlio_map -> base_link.
  // Your TF convention in this node publishes the inverse direction:
  //   parent = base_link, child = dlio_map
  // ---------------------------------------------------------------------------
  const Eigen::Vector3f p_mb = T_all.block<3,1>(0,3);
  Eigen::Quaternionf q_mb(T_all.block<3,3>(0,0));
  q_mb.normalize();

  const Eigen::Quaternionf q_bm = q_mb.conjugate();
  const Eigen::Vector3f p_bm = -(q_bm._transformVector(p_mb));

  // map_pose: pose of dlio_map in base_link, time-aligned to the scan.
  if (hasSubscribers(this->odom_map_pub)) {
    nav_msgs::msg::Odometry odom_map;
    odom_map.header.stamp = scan_stamp_msg;
    odom_map.header.frame_id = this->baselink_frame;
    odom_map.child_frame_id = "dlio_map";

    odom_map.pose.pose.position.x = p_bm.x();
    odom_map.pose.pose.position.y = p_bm.y();
    odom_map.pose.pose.position.z = p_bm.z();
    odom_map.pose.pose.orientation.w = q_bm.w();
    odom_map.pose.pose.orientation.x = q_bm.x();
    odom_map.pose.pose.orientation.y = q_bm.y();
    odom_map.pose.pose.orientation.z = q_bm.z();

    // Twist of dlio_map w.r.t. base_link, expressed in child frame (dlio_map).
    // Keep this derived from the same scan-time odom snapshot used below so the
    // published pose/twist/cloud/TF are self-consistent.
    const Eigen::Vector3f v_mb_m = q_mb._transformVector(state_vlin_b_scan);
    const Eigen::Vector3f w_mb_m = q_mb._transformVector(state_vang_b_scan);
    const Eigen::Vector3f v_bm_m = -(v_mb_m + p_mb.cross(w_mb_m));
    const Eigen::Vector3f w_bm_m = -w_mb_m;

    odom_map.twist.twist.linear.x  = v_bm_m.x();
    odom_map.twist.twist.linear.y  = v_bm_m.y();
    odom_map.twist.twist.linear.z  = v_bm_m.z();
    odom_map.twist.twist.angular.x = w_bm_m.x();
    odom_map.twist.twist.angular.y = w_bm_m.y();
    odom_map.twist.twist.angular.z = w_bm_m.z();

    this->odom_map_pub->publish(odom_map);
  }

  // base_pose: pose of base_link in dlio_map, time-aligned to the scan.
  if (hasSubscribers(this->odom_baselink_pub)) {
    nav_msgs::msg::Odometry odom_baselink;
    odom_baselink.header.stamp    = scan_stamp_msg;
    odom_baselink.header.frame_id = "dlio_map";
    odom_baselink.child_frame_id  = this->baselink_frame;

    odom_baselink.pose.pose.position.x    = p_mb.x();
    odom_baselink.pose.pose.position.y    = p_mb.y();
    odom_baselink.pose.pose.position.z    = p_mb.z();
    odom_baselink.pose.pose.orientation.w = q_mb.w();
    odom_baselink.pose.pose.orientation.x = q_mb.x();
    odom_baselink.pose.pose.orientation.y = q_mb.y();
    odom_baselink.pose.pose.orientation.z = q_mb.z();

    // Twist of base_link w.r.t. dlio_map, expressed in child frame (base_link).
    odom_baselink.twist.twist.linear.x  = state_vlin_b_scan.x();
    odom_baselink.twist.twist.linear.y  = state_vlin_b_scan.y();
    odom_baselink.twist.twist.linear.z  = state_vlin_b_scan.z();
    odom_baselink.twist.twist.angular.x = state_vang_b_scan.x();
    odom_baselink.twist.twist.angular.y = state_vang_b_scan.y();
    odom_baselink.twist.twist.angular.z = state_vang_b_scan.z();

    this->odom_baselink_pub->publish(odom_baselink);
  }

  // ---------------------------------------------------------------------------
  // Scan-time path in dlio_map
  // ---------------------------------------------------------------------------
  this->path_ros.header.stamp = scan_stamp_msg;
  this->path_ros.header.frame_id = "dlio_map";

  geometry_msgs::msg::PoseStamped path_pose;
  path_pose.header.stamp = scan_stamp_msg;
  path_pose.header.frame_id = "dlio_map";
  path_pose.pose.position.x = p_mb.x();
  path_pose.pose.position.y = p_mb.y();
  path_pose.pose.position.z = p_mb.z();
  path_pose.pose.orientation.w = q_mb.w();
  path_pose.pose.orientation.x = q_mb.x();
  path_pose.pose.orientation.y = q_mb.y();
  path_pose.pose.orientation.z = q_mb.z();

  constexpr size_t kMaxPath = 1500;
  if (this->path_poses_.size() >= kMaxPath) {
    this->path_poses_.pop_front();
  }
  this->path_poses_.push_back(std::move(path_pose));
  if (hasSubscribers(this->path_pub)) {
    this->path_ros.poses.assign(this->path_poses_.begin(), this->path_poses_.end());
    this->path_pub->publish(this->path_ros);
  }

  if (this->viz_corr_marker_ && this->pub_corr_marker_ &&
      hasSubscribers(this->pub_corr_marker_)) {
    this->publishCorrectionMarker(scan_time, T_cloud, T_all);
  }

  // ---------------------------------------------------------------------------
  // dlio_odom <-> base_link at the SAME scan-time corrected state snapshot
  //
  // state_q_scan/state_p_scan represent dlio_odom -> base_link at scan time.
  // The cloud published in dlio_odom must use this exact same snapshot.
  // ---------------------------------------------------------------------------
  const Eigen::Quaternionf q_ob = state_q_scan.normalized();
  const Eigen::Quaternionf q_bo = q_ob.conjugate();
  const Eigen::Vector3f p_bo = -(q_bo._transformVector(state_p_scan));

  Eigen::Matrix4f T_bl_odom = Eigen::Matrix4f::Identity();
  T_bl_odom.block<3,3>(0,0) = q_bo.toRotationMatrix();
  T_bl_odom.block<3,1>(0,3) = p_bo;

  // map -> odom at the exact scan-time corrected snapshot.
  const Eigen::Matrix4f T_map_odom = T_all * T_bl_odom;

  // Keep the latest map->odom for IMU-rate propagated map visualization.
  {
    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest = T_map_odom;
    this->has_T_map_odom_latest = true;
  }

  // ---------------------------------------------------------------------------
  // TFs at scan time
  //
  // Publish only base_link->dlio_map here.
  // base_link->dlio_odom is published exclusively in publishPoseSnapshot().
  // ---------------------------------------------------------------------------
  std::vector<geometry_msgs::msg::TransformStamped> tfs;
  tfs.reserve(1);

  geometry_msgs::msg::TransformStamped tf_bl_map;
  tf_bl_map.header.stamp = scan_stamp_msg;
  tf_bl_map.header.frame_id = this->baselink_frame;
  tf_bl_map.child_frame_id = "dlio_map";
  tf_bl_map.transform.translation.x = p_bm.x();
  tf_bl_map.transform.translation.y = p_bm.y();
  tf_bl_map.transform.translation.z = p_bm.z();
  tf_bl_map.transform.rotation.w = q_bm.w();
  tf_bl_map.transform.rotation.x = q_bm.x();
  tf_bl_map.transform.rotation.y = q_bm.y();
  tf_bl_map.transform.rotation.z = q_bm.z();
  tfs.emplace_back(std::move(tf_bl_map));

  this->br->sendTransform(tfs);

  // Publish clouds using the same scan-time map<->odom relation and the exact
  // same timestamp used for TF lookup.
  this->publishCloud(cloud, T_cloud, T_all, T_map_odom, scan_time);
}

static inline void prepare_xyz_msg(sensor_msgs::msg::PointCloud2& msg,
                                   const std::string& frame_id,
                                   const rclcpp::Time& stamp,
                                   size_t n) {
  msg.header.frame_id = frame_id;

  // precise conversion rclcpp::Time -> builtin_interfaces::msg::Time
  const int64_t nsec = stamp.nanoseconds();
  msg.header.stamp.sec     = static_cast<int32_t>(nsec / 1000000000LL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(nsec % 1000000000LL);

  msg.height = 1;
  msg.width  = static_cast<uint32_t>(n);
  msg.is_bigendian = false;
  msg.is_dense = true;

  sensor_msgs::PointCloud2Modifier mod(msg);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(n);
}

void dlio::OdomNode::publishCloud(
    const pcl::PointCloud<PointType>::ConstPtr& cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_odom,
    const rclcpp::Time& cloud_stamp)
{
  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    return;
  }

  if (this->wait_until_move_ && this->length_traversed < 0.1) {
    return;
  }

  if (!cloud) {
    return;
  }

  const size_t n = cloud->size();
  if (n == 0) {
    return;
  }

  // Cloud input is already in dlio_map after deskew / registration pipeline.
  // We publish three views of the same scan:
  //   1) deskewed                 : in dlio_odom
  //   2) deskewed_not_transformed : in base_link
  //   3) deskewed_and_transformed_to_map : in dlio_map
  // Each view is only computed and published when its topic has a subscriber.
  const bool want_odom = hasSubscribers(this->deskewed_pub);
  const bool want_base = hasSubscribers(this->deskewed_not_transformed_pub);
  const bool want_map  = hasSubscribers(this->deskewed_map_pub);

  if (!want_odom && !want_base && !want_map) {
    return;
  }

  if (want_odom) {
    const Eigen::Matrix4f T_odom_map = T_map_odom.inverse();
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, this->odom_frame, cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_odom = T_odom_map * (T_cloud * Eigen::Vector4f(p.x, p.y, p.z, 1.f));
      *x = v_odom.x(); *y = v_odom.y(); *z = v_odom.z();
      all_finite = all_finite && std::isfinite(v_odom.x()) && std::isfinite(v_odom.y()) && std::isfinite(v_odom.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_pub->publish(msg);
  }

  if (want_base) {
    const Eigen::Matrix4f T_revert = T_all.inverse() * T_cloud;
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, this->baselink_frame, cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_base = T_revert * Eigen::Vector4f(p.x, p.y, p.z, 1.f);
      *x = v_base.x(); *y = v_base.y(); *z = v_base.z();
      all_finite = all_finite && std::isfinite(v_base.x()) && std::isfinite(v_base.y()) && std::isfinite(v_base.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_not_transformed_pub->publish(msg);
  }

  if (want_map) {
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, "dlio_map", cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_map = T_cloud * Eigen::Vector4f(p.x, p.y, p.z, 1.f);
      *x = v_map.x(); *y = v_map.y(); *z = v_map.z();
      all_finite = all_finite && std::isfinite(v_map.x()) && std::isfinite(v_map.y()) && std::isfinite(v_map.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_map_pub->publish(msg);
  }
}

void dlio::OdomNode::publishKeyframe(
    std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf,
    rclcpp::Time timestamp) {

  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    return;
  }

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Trim PoseArray to avoid unbounded RViz payload
  if (this->kf_pose_ros.poses.size() > 30) {
    const auto trim_count =
        static_cast<decltype(this->kf_pose_ros.poses)::difference_type>(
            this->kf_pose_ros.poses.size() - 30U);
    this->kf_pose_ros.poses.erase(
      this->kf_pose_ros.poses.begin(),
      this->kf_pose_ros.poses.begin() + trim_count
    );
  }

  // Keyframes are stored/published after being transformed into the map frame.
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = "dlio_map";
  if (hasSubscribers(this->kf_pose_pub)) {
    this->kf_pose_pub->publish(this->kf_pose_ros);
  }

  if (hasSubscribers(this->kf_cloud_pub)) {
    auto publish_kf_cloud = [&]() {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = "dlio_map";
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    };
    if (this->vf_use_) {
      if (kf.second->points.size() ==
          static_cast<decltype(kf.second->points.size())>(kf.second->width) *
              static_cast<decltype(kf.second->points.size())>(kf.second->height)) {
        publish_kf_cloud();
      }
    } else {
      publish_kf_cloud();
    }
  }
}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  // std::vector<int> idx;
  // original_scan_->is_dense = false;
  // pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

  // automatically detect sensor type
  if (this->sensor == dlio::SensorType::UNKNOWN) {
    for (auto &field : pc->fields) {
      if (field.name == "t") {
        this->sensor = dlio::SensorType::OUSTER;
        break;
      } else if (field.name == "time") {
        this->sensor = dlio::SensorType::VELODYNE;
        break;
      } else if (field.name == "timestamp" && !original_scan_->points.empty() && original_scan_->points[0].timestamp < 1e14) {
        this->sensor = dlio::SensorType::HESAI;
        // ROBOSENSE IS ALSO HERE
        break;
      } else if (field.name == "timestamp" && !original_scan_->points.empty() && original_scan_->points[0].timestamp > 1e14) {
        this->sensor = dlio::SensorType::LIVOX;
        break;
      }
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

}

void dlio::OdomNode::preprocessPoints() {

  if (!this->original_scan || this->original_scan->empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
    this->current_scan = this->deskewed_scan;
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data or external odometry is present
    if (!this->first_valid_scan) {
      bool imu_ready = false;
      {
        std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
        imu_ready = !this->imu_buffer.empty() &&
                    this->imu_buffer.front().stamp >= this->scan_stamp &&
                    this->imu_buffer.back().stamp < this->scan_stamp;
      }
      bool ext_odom_ready = this->first_external_odom_received;

      if (!imu_ready && !ext_odom_ready) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      (void)this->predictScanState(this->scan_stamp);

      State scan_anchor;
      {
        std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
        scan_anchor = this->scan_state;
      }
      bool have_imu = false;
      {
        std::lock_guard<decltype(this->mtx_imu)> imu_lock(this->mtx_imu);
        have_imu = !this->imu_buffer.empty();
      }

      if (have_imu) {
        // IMU prior: integrate IMU between scans
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
        frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                    scan_anchor.v.lin.w, {this->scan_stamp}, scan_anchor.b);

        if (frames.size() > 0) {
          this->T_prior = frames.back();
        } else {
          this->T_prior = this->T;
        }
      } else if (this->first_external_odom_received) {
        // External odom prior: delta pose from ground truth gives GICP a good initial guess
        std::unique_lock<std::mutex> lock(this->mtx_external_odom);
        Eigen::Vector3f dp = this->externalOdomPose.p - this->prevExternalOdomPose.p;
        Eigen::Quaternionf dq = this->prevExternalOdomPose.q.inverse() * this->externalOdomPose.q;
        lock.unlock();

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = dq.toRotationMatrix();
        delta.block<3,1>(0,3) = dp;

        this->T_prior = delta * this->T;
      } else {
        this->T_prior = this->T;
      }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::deskewPointcloud() {

  // Establish a valid scan_stamp up front. Several early returns below leave
  // without reaching the per-point timestamp extraction; if scan_stamp kept the
  // previous scan's value, prev_scan_stamp would freeze and every later sweep
  // would query a progressively staler IMU interval.
  this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

  if (!this->original_scan || this->original_scan->empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  auto deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  deskewed_scan_->points.resize(this->original_scan->points.size());
  deskewed_scan_->width  = static_cast<uint32_t>(deskewed_scan_->points.size());
  deskewed_scan_->height = 1;

  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + static_cast<double>(pt.value().t) * 1e-9; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };

  } else if (this->sensor == dlio::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [](boost::range::index_value<PointType&, long> pt)
      { return static_cast<double>(pt.value().timestamp) * 1e-9; };
  
    } else {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    const auto begin_it = points_unique_timestamps.begin();
    if (begin_it == points_unique_timestamps.end()) {
      this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
      this->deskew_status = false;
      this->deskew_size = 0;
      return;
    }
    offset = sweep_ref_time - extract_point_time(*begin_it);
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(static_cast<int>(it->index()));
  }

  if (timestamps.empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  unique_time_indices.push_back(static_cast<int>(deskewed_scan_->points.size()));

  // RCLCPP_INFO_THROTTLE(
  //     this->get_logger(), *this->get_clock(), 1000,
  //     "[deskew dbg] header=%.9f first_raw=%.9f last_raw=%.9f first_adj=%.9f last_adj=%.9f "
  //     "header-first_raw=%.3f ms header-last_raw=%.3f ms span=%.3f ms offset=%.3f ms n_unique=%zu",
  //     sweep_ref_time,
  //     first_point_time_raw,
  //     last_point_time_raw,
  //     timestamps.front(),
  //     timestamps.back(),
  //     1e3 * (sweep_ref_time - first_point_time_raw),
  //     1e3 * (sweep_ref_time - last_point_time_raw),
  //     1e3 * (timestamps.back() - timestamps.front()),
  //     1e3 * offset,
  //     timestamps.size());

  // int median_pt_index = timestamps.size() / 2;
  // this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point
  this->scan_stamp = timestamps[0];

  // if (this->prev_scan_stamp > 0.0) {
  //   RCLCPP_INFO_THROTTLE(
  //       this->get_logger(), *this->get_clock(), 1000,
  //       "[deskew interval dbg] prev_first=%.9f curr_first=%.9f curr_last=%.9f "
  //       "scan_span=%.3f ms query_span=%.3f ms",
  //       this->prev_scan_stamp,
  //       timestamps.front(),
  //       timestamps.back(),
  //       1e3 * (timestamps.back() - timestamps.front()),
  //       1e3 * (timestamps.back() - this->prev_scan_stamp));
  // }

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    bool imu_ready = false;
    {
      std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
      imu_ready = !this->imu_buffer.empty() &&
                  this->imu_buffer.front().stamp >= timestamps.back() &&
                  this->imu_buffer.back().stamp < timestamps.front();
    }

    if (!imu_ready) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  (void)this->predictScanState(this->scan_stamp);
  State scan_anchor;
  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    scan_anchor = this->scan_state;
  }

  // IMU prior & deskewing for second scan onwards.
  //
  // The integration normally starts at prev_scan_stamp, anchored on the last
  // registered LiDAR pose. That anchor is only usable while prev_scan_stamp is
  // recent and still covered by the IMU ring buffer. After a degeneracy halt,
  // a restart, or any dropped scan, it can be far behind -- integrateImu() then
  // returns an empty vector and, because the anchor never moves on its own,
  // every subsequent sweep fails the same way. Detect that case and re-anchor
  // the sweep at its own start using the predicted scan state instead.
  const double oldest_imu_stamp = this->oldestImuStamp();
  double integration_start = this->prev_scan_stamp;
  Eigen::Quaternionf anchor_q = this->lidarPose.q;
  Eigen::Vector3f anchor_p = this->lidarPose.p;

  const char* reanchor_reason = nullptr;
  if (!std::isfinite(integration_start) || integration_start <= 0.0) {
    reanchor_reason = "anchor not initialized";
  } else if (integration_start > timestamps.front()) {
    reanchor_reason = "anchor is newer than the sweep start";
  } else if (timestamps.front() - integration_start > this->deskew_max_lookback_) {
    reanchor_reason = "anchor is older than the maximum deskew lookback";
  } else if (oldest_imu_stamp >= 0.0 && integration_start <= oldest_imu_stamp) {
    reanchor_reason = "anchor predates the oldest buffered IMU sample";
  }

  if (reanchor_reason != nullptr) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Deskew re-anchored at the sweep start (%s): prev_scan_stamp=%.6f "
      "sweep=[%.6f, %.6f] oldest_imu=%.6f.",
      reanchor_reason, this->prev_scan_stamp,
      timestamps.front(), timestamps.back(), oldest_imu_stamp);

    integration_start = timestamps.front();
    anchor_q = scan_anchor.q;
    anchor_p = scan_anchor.p;
  }

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(integration_start, anchor_q, anchor_p,
                              scan_anchor.v.lin.w, timestamps, scan_anchor.b);
  this->deskew_size = static_cast<int>(frames.size()); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Deskew unavailable: integrateImu returned %zu/%zu frames "
      "(start=%.6f sweep=[%.6f, %.6f] oldest_imu=%.6f). "
      "Using T_prior without per-point compensation.",
      frames.size(), timestamps.size(), integration_start,
      timestamps.front(), timestamps.back(), oldest_imu_stamp);

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  // this->T_prior = frames[median_pt_index];
  this->T_prior = frames[0];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    this->scan_state = this->state;
    this->scan_state.p = this->lidarPose.p;
    this->scan_state.q = this->lidarPose.q.normalized();
    this->scan_state.v.lin.b =
        this->scan_state.q.toRotationMatrix().transpose() * this->scan_state.v.lin.w;
    this->scan_state_stamp_ = this->scan_stamp;
    this->scan_state_valid_ = true;
    this->scan_state_prior = this->scan_state;
    this->scan_state_prior_stamp_ = this->scan_stamp;
    this->scan_state_prior_valid_ = true;
    this->state = this->scan_state;
    this->live_state_stamp_ = this->scan_stamp;
    this->geo.prev_p = this->scan_state.p;
    this->geo.prev_q = this->scan_state.q;
    this->geo.prev_vel = this->scan_state.v.lin.w;
  }

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

}

void dlio::OdomNode::setInputSource() {
  // Source = current deskewed/filtered scan in world frame.
  // NanoGICP builds a source k-d tree and source covariances from this cloud.
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  bool imu_ready = this->first_imu_received && this->imu_calibrated;
  bool ext_odom_ready = this->first_external_odom_received;

  if (!imu_ready && !ext_odom_ready) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << '\n' << " DLIO initialized! (via "
            << (ext_odom_ready ? "external odometry" : "IMU") << ")\n";

}

// ROS 2 Jazzy subscription callbacks accept SharedPtr by value; const ref is not a supported callback signature here.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
void dlio::OdomNode::callbackPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr pc) {
  this->checkPointCloudTiming(pc);

  // Keep callback lightweight to avoid blocking DDS receive threads.
  this->enqueuePointCloud(pc);

  if (!this->debug_enabled_) {
    constexpr std::size_t kRateWindowSize = 20;
    this->pc_rate_window_.push_back(std::chrono::steady_clock::now());
    if (this->pc_rate_window_.size() > kRateWindowSize) {
      this->pc_rate_window_.pop_front();
    }
    if (this->pc_rate_window_.size() >= 2) {
      const auto now = this->pc_rate_window_.back();
      const double since_print = std::chrono::duration<double>(
          now - this->pc_rate_last_print_).count();
      if (since_print >= 2.0) {
        const double span = std::chrono::duration<double>(
            now - this->pc_rate_window_.front()).count();
        const double rate = static_cast<double>(this->pc_rate_window_.size() - 1) / span;
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        printf("\033[32m[%02d:%02d:%02d] [DLIO] Pointcloud rate: %.2f Hz\033[0m\n",
               tm.tm_hour, tm.tm_min, tm.tm_sec, rate);
        this->pc_rate_last_print_ = now;
      }
    }
  }
}

void dlio::OdomNode::checkPointCloudTiming(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {
  if (!this->pointcloud_timing_enabled_ || !pc) {
    return;
  }

  const std::int64_t stamp_ns = rclcpp::Time(pc->header.stamp).nanoseconds();
  std::lock_guard<std::mutex> lock(this->pointcloud_timing_mutex_);

  if (!this->pointcloud_timing_has_previous_) {
    this->pointcloud_previous_stamp_ns_ = stamp_ns;
    this->pointcloud_timing_has_previous_ = true;
    return;
  }

  const double period = static_cast<double>(stamp_ns - this->pointcloud_previous_stamp_ns_) / 1e9;
  const double lower_bound = this->pointcloud_expected_period_ - this->pointcloud_period_tolerance_;
  const double upper_bound = this->pointcloud_expected_period_ + this->pointcloud_period_tolerance_;
  const bool unexpected_period = period < lower_bound || period > upper_bound;

  if (unexpected_period) {
    const char* classification = period <= 0.0 ? "non-increasing" :
        (period > upper_bound ? "delayed" : "too fast");
    RCLCPP_ERROR(
        this->get_logger(),
        "\033[31m[POINTCLOUD TIMING] %s header interval: %.3f ms "
        "(expected %.3f +/- %.3f ms); previous=%.9f, current=%.9f\033[0m",
        classification,
        period * 1e3,
        this->pointcloud_expected_period_ * 1e3,
        this->pointcloud_period_tolerance_ * 1e3,
        static_cast<double>(this->pointcloud_previous_stamp_ns_) / 1e9,
        static_cast<double>(stamp_ns) / 1e9);
  }

  // Advance on every sample so a single late/early cloud does not cause all
  // subsequent warnings to be measured against an old header.
  this->pointcloud_previous_stamp_ns_ = stamp_ns;
}

void dlio::OdomNode::resetPointCloudTiming() {
  std::lock_guard<std::mutex> lock(this->pointcloud_timing_mutex_);
  this->pointcloud_timing_has_previous_ = false;
  this->pointcloud_previous_stamp_ns_ = 0;
}

void dlio::OdomNode::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    this->processHaltedScan(pc);
    return;
  }

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  // Use steady_clock, not this->now(), which is the ROS node clock.
  // When use_sim_time=true the ROS clock is driven by /clock messages and
  // does not advance during computation, so now()-then would be ~0.
  const auto then = std::chrono::steady_clock::now();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Single owner of the deskew anchor: whichever way this function returns, the
  // scan that was just consumed becomes the new prev_scan_stamp. Without this
  // the early-return paths below (too few points, degenerate initial scan,
  // rejected pose) leave the anchor behind, and the gap it has to cover grows
  // with every dropped scan until it no longer fits the IMU ring buffer.
  const auto scan_stamp_guard = makeScopeExit([this]() { this->advancePrevScanStamp(); });

  if (!this->original_scan || this->original_scan->empty()) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  if (!this->current_scan || this->current_scan->points.size() <= this->gicp_min_num_points_) {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  // Compute Metrics
  this->computeMetrics();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    if (this->use_degeneracy_) {
      this->analyzeDegeneracyFromCurrentScan(this->T_prior);
      this->publishDegeneracyMarkers(this->scan_header_stamp);

      State initial_candidate;
      {
        std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
        initial_candidate = this->scan_state_prior_valid_ ? this->scan_state_prior : this->state;
      }

      std::string reason;
      const bool initial_degenerate = this->currentScanIsDegenerate(initial_candidate, reason);
      this->publishDegeneracyStatus(initial_degenerate);
      if (initial_degenerate) {
        ++this->degen_consecutive_hits_;
        this->enterDegenerateHalt("initial scan: " + reason);
        lock.lock();
        this->main_loop_running = false;
        lock.unlock();
        this->submap_build_cv.notify_one();
        return;
      }
    }

    this->initializeInputTarget();
    if (!this->use_degeneracy_) {
      this->publishDegeneracyStatus(false);
    }
    State initial_scan_state;
    {
      std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
      initial_scan_state = this->scan_state;
    }
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, initial_scan_state);
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  if (!this->getNextPose()) {
    // scan_stamp_guard advances the deskew anchor past this rejected scan, so a
    // halt cannot leave recovery asking for a multi-second stale IMU interval.
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  // Capture a scan-time odom snapshot immediately after the LiDAR update.
  // This snapshot must travel with the scan so that map<->odom for the published cloud
  // is computed from the same timestamped state as T_all/T_cloud.
  Eigen::Vector3f state_p_scan, state_vlin_b_scan, state_vang_b_scan;
  Eigen::Quaternionf state_q_scan;
  State scan_state_snapshot;
  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    scan_state_snapshot = this->scan_state;
    state_p_scan = scan_state_snapshot.p;
    state_q_scan = scan_state_snapshot.q.normalized();
    state_vlin_b_scan = scan_state_snapshot.v.lin.b;
    state_vang_b_scan = scan_state_snapshot.v.ang.b;
  }
  // Update latest map->odom at scan time so IMU-rate map propagation can use it immediately.
  {
    const Eigen::Quaternionf q_bo_scan = state_q_scan.conjugate();
    const Eigen::Vector3f p_bo_scan = -(q_bo_scan._transformVector(state_p_scan));
    Eigen::Matrix4f T_bl_odom_scan = Eigen::Matrix4f::Identity();
    T_bl_odom_scan.block<3,3>(0,0) = q_bo_scan.toRotationMatrix();
    T_bl_odom_scan.block<3,1>(0,3) = p_bo_scan;
    const Eigen::Matrix4f T_map_odom_scan = this->T * T_bl_odom_scan;

    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest = T_map_odom_scan;
    this->has_T_map_odom_latest = true;
  }

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, scan_state_snapshot);
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Incremental distance (keep cumulative exact, do not recompute over entire trajectory)
  if (!this->trajectory.empty()) {
    const Eigen::Vector3f& prev = this->trajectory.back().first;
    const double l = (scan_state_snapshot.p - prev).norm();
    if (l >= 0.1) this->length_traversed += l;
  }

  // Keep trajectory only for recent visualization/debug
  this->trajectory.emplace_back(scan_state_snapshot.p, scan_state_snapshot.q);
  if (this->trajectory.size() > 1600) {
    const auto trim_count = static_cast<std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>>::difference_type>(
        this->trajectory.size() - 1600);
    this->trajectory.erase(this->trajectory.begin(), this->trajectory.begin() + trim_count);
  }

  // Update time stamps
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }

  this->enqueuePublish(std::move(published_cloud),
                       this->T_corr,
                       this->T,
                       this->scan_stamp,
                       state_p_scan,
                       state_q_scan,
                       state_vlin_b_scan,
                       state_vang_b_scan);

  // Update computation time statistics: rolling 2-second window keyed by scan_stamp.
  {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count();
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    this->comp_times.push_back({this->scan_stamp, elapsed});
    const double cutoff = this->scan_stamp - 2.0;
    while (!this->comp_times.empty() && this->comp_times.front().first < cutoff) {
      this->comp_times.pop_front();
    }
  }

  // this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements and publish custom DLIO message
  if (this->debug_enabled_) {
    if (!this->debug_future_.valid() ||
        this->debug_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      this->debug_future_ = std::async(std::launch::async, &dlio::OdomNode::debug, this);
    }
  }

  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    this->geo.first_opt_done = true;
  }
}

// ROS 2 Jazzy subscription callbacks accept SharedPtr by value; const ref is not a supported callback signature here.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
void dlio::OdomNode::callbackImu(sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  // The IMU callback group is mutually exclusive, but reset runs on the point
  // cloud worker. Serialize their access to timestamp and transform history.
  std::lock_guard<std::mutex> imu_callback_lock(this->mtx_imu_callback_);

  const rclcpp::Time raw_stamp(imu_raw->header.stamp);
  const auto input_timing = this->imu_input_timestamps_.observe(raw_stamp.nanoseconds());
  if (!input_timing.accepted) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Dropping non-monotonic IMU sample: stamp=%lld ns, previous=%lld ns "
        "(lifetime drops=%llu).",
        static_cast<long long>(raw_stamp.nanoseconds()),
        static_cast<long long>(input_timing.previous_stamp_ns),
        static_cast<unsigned long long>(input_timing.rejected_count));
    return;
  }

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu(imu_raw, input_timing.dt_seconds);
  this->imu_stamp = imu->header.stamp;
  const double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  const Eigen::Vector3f ang_vel(static_cast<float>(imu->angular_velocity.x),
                                static_cast<float>(imu->angular_velocity.y),
                                static_cast<float>(imu->angular_velocity.z));

  const Eigen::Vector3f lin_accel(static_cast<float>(imu->linear_acceleration.x),
                                  static_cast<float>(imu->linear_acceleration.y),
                                  static_cast<float>(imu->linear_acceleration.z));

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      this->imu_calib_samples_++;

      this->imu_calib_gyro_sum_[0] += ang_vel[0];
      this->imu_calib_gyro_sum_[1] += ang_vel[1];
      this->imu_calib_gyro_sum_[2] += ang_vel[2];

      this->imu_calib_accel_sum_[0] += lin_accel[0];
      this->imu_calib_accel_sum_[1] += lin_accel[1];
      this->imu_calib_accel_sum_[2] += lin_accel[2];

      if (!this->imu_calib_printed_) {
        std::cout << '\n' << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        this->imu_calib_printed_ = true;
      }

    } else {

      std::cout << "done\n\n";

      const float sample_count = static_cast<float>(std::max(this->imu_calib_samples_, 1));
      const Eigen::Vector3f gyro_avg = this->imu_calib_gyro_sum_ / sample_count;
      const Eigen::Vector3f accel_avg = this->imu_calib_accel_sum_ / sample_count;

      Eigen::Vector3f grav_vec(0.0f, 0.0f, static_cast<float>(this->gravity_));

      if (this->gravity_align_) {
        Eigen::Vector3f accel_bias;
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          accel_bias = this->state.b.accel;
        }

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - accel_bias).normalized() * static_cast<float>(std::abs(this->gravity_));
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(
            grav_vec, Eigen::Vector3f(0.0f, 0.0f, static_cast<float>(this->gravity_)));

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:\n";
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << '\n';
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << '\n';
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << '\n';
        std::cout << '\n';

        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          // set gravity aligned orientation
          this->state.q = grav_q;
          this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
          this->lidarPose.q = this->state.q;
        }
      }

      if (this->calibrate_accel_) {
        Eigen::Vector3f accel_bias = accel_avg - grav_vec;
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          this->state.b.accel = accel_bias;
        }

        // subtract gravity from avg accel to get bias
        std::cout << " Accel biases [xyz]: " << to_string_with_precision(accel_bias[0], 8) << ", "
                                             << to_string_with_precision(accel_bias[1], 8) << ", "
                                             << to_string_with_precision(accel_bias[2], 8) << '\n';
      }

      if (this->calibrate_gyro_) {
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          this->state.b.gyro = gyro_avg;
        }

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(gyro_avg[0], 8) << ", "
                                             << to_string_with_precision(gyro_avg[1], 8) << ", "
                                             << to_string_with_precision(gyro_avg[2], 8) << '\n';
      }

      this->imu_calib_samples_ = 0;
      this->imu_calib_gyro_sum_.setZero();
      this->imu_calib_accel_sum_.setZero();
      this->imu_calib_printed_ = false;

      Eigen::Quaternionf initial_gravity_align_q = Eigen::Quaternionf::Identity();
      Eigen::Vector3f initial_accel_bias = Eigen::Vector3f::Zero();
      Eigen::Vector3f initial_gyro_bias = Eigen::Vector3f::Zero();
      {
        std::lock_guard<std::mutex> lock(this->geo.mtx);
        initial_gravity_align_q = this->state.q;
        initial_accel_bias = this->state.b.accel;
        initial_gyro_bias = this->state.b.gyro;
      }
      this->captureInitialImuBaseline(
          grav_vec,
          initial_gravity_align_q,
          initial_accel_bias,
          initial_gyro_bias);

      this->imu_calibrated = true;
      this->imu_integration_timestamps_.resetSequence();
      (void)this->imu_integration_timestamps_.observe(raw_stamp.nanoseconds());

    }

  } else {

    const auto integration_timing =
        this->imu_integration_timestamps_.observe(raw_stamp.nanoseconds());
    if (!integration_timing.accepted) {
      // input_timing already enforces strict ordering. Reaching this branch
      // indicates inconsistent internal reset/calibration state.
      RCLCPP_ERROR(this->get_logger(),
                   "IMU integration timestamp tracker rejected an accepted input sample.");
      return;
    }

    const double dt = integration_timing.dt_seconds.value_or(0.0);
    // this->imu_rates.push_back( 1./dt );

    // Store transformed/calibrated measurements with bias still present.
    // Bias is applied explicitly by each consumer at its state timestamp.
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->imu_meas.lin_accel = this->imu_accel_sm_ * lin_accel;
    this->imu_meas.ang_vel = ang_vel;

    // Store bias-uncorrected measurements for deterministic scan integration and replay.
    {
      std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
      this->imu_buffer.push_front(this->imu_meas);
    }

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (!this->estimator_halted_.load(std::memory_order_relaxed) &&
        integration_timing.dt_seconds.has_value() && this->propagateState(this->imu_meas)) {
      this->publishPoseSnapshot();
    }

  }

}

void dlio::OdomNode::publishPoseSnapshot() {
  if (this->estimator_halted_.load(std::memory_order_relaxed)) {
    return;
  }

  Eigen::Vector3f p = Eigen::Vector3f::Zero();
  Eigen::Vector3f vlin_b = Eigen::Vector3f::Zero();
  Eigen::Vector3f vang_b = Eigen::Vector3f::Zero();
  Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
  rclcpp::Time stamp(0, 0);

  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    p      = this->state.p;
    q      = this->state.q;
    vlin_b = this->state.v.lin.b;
    vang_b = this->state.v.ang.b;
    stamp  = this->imu_stamp;
  }

  q.normalize();

  if (this->use_degeneracy_) {
    State live_snapshot;
    live_snapshot.p = p;
    live_snapshot.q = q;
    live_snapshot.v.lin.w = q.toRotationMatrix() * vlin_b;
    live_snapshot.v.lin.b = vlin_b;
    live_snapshot.v.ang.b = vang_b;
    live_snapshot.v.ang.w = q.toRotationMatrix() * vang_b;
    {
      std::lock_guard<std::mutex> lock(this->geo.mtx);
      live_snapshot.b = this->state.b;
    }
    std::string reason;
    if (this->stateExceedsDegeneracyBounds(live_snapshot, reason)) {
      this->enterDegenerateHalt("IMU propagation: " + reason);
      this->publishDegeneracyStatus(true);
      return;
    }
  }

  // Build and publish Odometry (dlio_odom -> base_link)
  // twist is expressed in child frame (base_link), so use body-frame velocities directly.
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = stamp;
  odom_msg.header.frame_id = this->odom_frame;
  odom_msg.child_frame_id  = this->baselink_frame;

  odom_msg.pose.pose.position.x = p.x();
  odom_msg.pose.pose.position.y = p.y();
  odom_msg.pose.pose.position.z = p.z();
  odom_msg.pose.pose.orientation.w = q.w();
  odom_msg.pose.pose.orientation.x = q.x();
  odom_msg.pose.pose.orientation.y = q.y();
  odom_msg.pose.pose.orientation.z = q.z();

  odom_msg.twist.twist.linear.x  = vlin_b.x();
  odom_msg.twist.twist.linear.y  = vlin_b.y();
  odom_msg.twist.twist.linear.z  = vlin_b.z();
  odom_msg.twist.twist.angular.x = vang_b.x();
  odom_msg.twist.twist.angular.y = vang_b.y();
  odom_msg.twist.twist.angular.z = vang_b.z();

  if (hasSubscribers(this->odom_pub)) {
    this->odom_pub->publish(odom_msg);
  }

  // Build and publish PoseStamped (dlio_odom -> base_link)
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = stamp;
  pose_msg.header.frame_id = this->odom_frame;
  pose_msg.pose.position.x = p.x();
  pose_msg.pose.position.y = p.y();
  pose_msg.pose.position.z = p.z();
  pose_msg.pose.orientation.w = q.w();
  pose_msg.pose.orientation.x = q.x();
  pose_msg.pose.orientation.y = q.y();
  pose_msg.pose.orientation.z = q.z();

  if (hasSubscribers(this->pose_pub)) {
    this->pose_pub->publish(pose_msg);
  }

  // Path: base_link pose in dlio_odom at IMU propagation rate.
  this->path_odom_ros.header.stamp = stamp;
  this->path_odom_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped pose_odom;
  pose_odom.header.stamp = stamp;
  pose_odom.header.frame_id = this->odom_frame;
  pose_odom.pose.position.x = p.x();
  pose_odom.pose.position.y = p.y();
  pose_odom.pose.position.z = p.z();
  pose_odom.pose.orientation.w = q.w();
  pose_odom.pose.orientation.x = q.x();
  pose_odom.pose.orientation.y = q.y();
  pose_odom.pose.orientation.z = q.z();

  constexpr size_t kMaxOdomPath = 10000;
  if (this->path_odom_poses_.size() >= kMaxOdomPath) {
    this->path_odom_poses_.pop_front();
  }
  this->path_odom_poses_.push_back(std::move(pose_odom));
  if (hasSubscribers(this->path_odom_pub)) {
    this->path_odom_ros.poses.assign(this->path_odom_poses_.begin(), this->path_odom_poses_.end());
    this->path_odom_pub->publish(this->path_odom_ros);
  }

  // IMU-rate propagated base_link trajectory in dlio_map using latest scan-time map->odom.
  Eigen::Matrix4f T_map_odom = Eigen::Matrix4f::Identity();
  bool has_T_map_odom = false;
  {
    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    has_T_map_odom = this->has_T_map_odom_latest;
    if (has_T_map_odom) {
      T_map_odom = this->T_map_odom_latest;
    }
  }
  if (has_T_map_odom) {
    Eigen::Matrix4f T_odom_base = Eigen::Matrix4f::Identity();
    T_odom_base.block<3,3>(0,0) = q.toRotationMatrix();
    T_odom_base.block<3,1>(0,3) = p;

    const Eigen::Matrix4f T_map_base = T_map_odom * T_odom_base;
    const Eigen::Vector3f p_mb_prop = T_map_base.block<3,1>(0,3);
    Eigen::Quaternionf q_mb_prop(T_map_base.block<3,3>(0,0));
    q_mb_prop.normalize();

    this->path_map_prop_ros.header.stamp = stamp;
    this->path_map_prop_ros.header.frame_id = "dlio_map";

    geometry_msgs::msg::PoseStamped pose_map_prop;
    pose_map_prop.header.stamp = stamp;
    pose_map_prop.header.frame_id = "dlio_map";
    pose_map_prop.pose.position.x = p_mb_prop.x();
    pose_map_prop.pose.position.y = p_mb_prop.y();
    pose_map_prop.pose.position.z = p_mb_prop.z();
    pose_map_prop.pose.orientation.w = q_mb_prop.w();
    pose_map_prop.pose.orientation.x = q_mb_prop.x();
    pose_map_prop.pose.orientation.y = q_mb_prop.y();
    pose_map_prop.pose.orientation.z = q_mb_prop.z();

    constexpr size_t kMaxMapPropPath = 10000;
    if (this->path_map_prop_poses_.size() >= kMaxMapPropPath) {
      this->path_map_prop_poses_.pop_front();
    }
    this->path_map_prop_poses_.push_back(std::move(pose_map_prop));
    if (hasSubscribers(this->path_map_prop_pub)) {
      this->path_map_prop_ros.poses.assign(this->path_map_prop_poses_.begin(), this->path_map_prop_poses_.end());
      this->path_map_prop_pub->publish(this->path_map_prop_ros);
    }
  }

  // TF: base_link -> dlio_odom (inverted state)
  const Eigen::Quaternionf q_inv = q.conjugate();
  const Eigen::Vector3f p_inv = -(q_inv._transformVector(p));

  geometry_msgs::msg::TransformStamped tf_bl_odom;
  tf_bl_odom.header.stamp = stamp;
  tf_bl_odom.header.frame_id = this->baselink_frame;
  tf_bl_odom.child_frame_id  = this->odom_frame;
  tf_bl_odom.transform.translation.x = p_inv.x();
  tf_bl_odom.transform.translation.y = p_inv.y();
  tf_bl_odom.transform.translation.z = p_inv.z();
  tf_bl_odom.transform.rotation.w = q_inv.w();
  tf_bl_odom.transform.rotation.x = q_inv.x();
  tf_bl_odom.transform.rotation.y = q_inv.y();
  tf_bl_odom.transform.rotation.z = q_inv.z();
  br->sendTransform(tf_bl_odom);

  this->publishVelocityMarkers(stamp, vlin_b, vang_b);
}

void dlio::OdomNode::publishCorrectionMarker(
    const rclcpp::Time& stamp,
    const Eigen::Ref<const Eigen::Matrix4f>& T_corr,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all) {
  if (!this->viz_corr_marker_) {
    return;
  }
  if (!this->pub_corr_marker_) {
    return;
  }

  if (!hasSubscribers(this->pub_corr_marker_)) {
    return;
  }

  // Recover T_prior from T_all = T_corr * T_prior without a full 4x4 inverse.
  const Eigen::Matrix3f R_corr = T_corr.block<3,3>(0,0);
  const Eigen::Vector3f t_corr = T_corr.block<3,1>(0,3);
  const Eigen::Vector3f p_all  = T_all.block<3,1>(0,3);
  const Eigen::Vector3f p_prior = R_corr.transpose() * (p_all - t_corr);
  const Eigen::Vector3f corr_vec = p_all - p_prior;

  visualization_msgs::msg::Marker marker;
  this->createCorrectionMarker("dlio_map", stamp, p_prior, corr_vec, marker);
  this->pub_corr_marker_->publish(marker);
}

void dlio::OdomNode::analyzeDegeneracyFromCurrentScan(
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_base) {
  const auto covs = this->gicp.getSourceCovariances();
  if (!covs || covs->empty()) {
    this->degen_info_.valid = false;
    return;
  }

  const NormalScatterStats normal_stats = buildNormalScatterMatrix(*covs);
  if (normal_stats.valid_normals < 3 ||
      normal_stats.total_weight <= 0.0 ||
      !normal_stats.scatter.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  const Eigen::Matrix3d normal_spread = normal_stats.scatter;
  if (!normal_spread.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  this->degen_info_.p_map_base = T_map_base.block<3,1>(0,3).cast<double>();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans(normal_spread);
  if (eig_trans.info() != Eigen::Success) {
    this->degen_info_.valid = false;
    return;
  }

  Eigen::Vector3d trans_evals_sorted = Eigen::Vector3d::Zero();
  Eigen::Matrix3d trans_evecs_sorted = Eigen::Matrix3d::Identity();

  sortEigenpairsAscending(
      eig_trans.eigenvalues(), eig_trans.eigenvectors(),
      trans_evals_sorted, trans_evecs_sorted);

  for (int k = 0; k < 3; ++k) {
    trans_evecs_sorted.col(k).normalize();
  }

  this->degen_info_.valid = true;
  this->degen_info_.eigvals_trans_dec = trans_evals_sorted;
  this->degen_info_.eigvecs_trans_map = trans_evecs_sorted;
  this->degen_info_.weak_trans = classifyWeakDirections(
      this->degen_info_.eigvals_trans_dec,
      this->degen_trans_eig_abs_thresh_);

  constexpr double eps = 1e-12;
  const double trans_min = std::max(this->degen_info_.eigvals_trans_dec(0), eps);
  const double trans_max = std::max(this->degen_info_.eigvals_trans_dec(2), eps);
  this->degen_info_.trans_condition = trans_max / trans_min;

  if (hasWeakDirection(this->degen_info_.weak_trans)) {
    logTranslationDegeneracy(
        this->get_logger(),
        this->degen_info_.eigvals_trans_dec,
        this->degen_info_.eigvecs_trans_map,
        this->degen_info_.weak_trans);
  } 
  // else {
  //   logTranslationSpectrumAlways(
  //       this->get_logger(),
  //       this->degen_info_.eigvals_trans_dec,
  //       this->degen_info_.eigvecs_trans_map);
  // }
}

void dlio::OdomNode::publishDegeneracyMarkers(const rclcpp::Time& stamp) {
  if (!this->viz_degen_marker_ || !this->pub_degen_marker_) {
    return;
  }
  if (!hasSubscribers(this->pub_degen_marker_)) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.reserve(3);

  auto pushDelete = [&](const int id, const std::string& ns) {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = "dlio_map";
    m.ns = ns;
    m.id = id;
    m.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.push_back(std::move(m));
  };

  if (!this->degen_info_.valid) {
    for (int i = 0; i < 3; ++i) {
      pushDelete(i, "degeneracy_translation");
    }
    if (hasSubscribers(this->pub_degen_marker_)) {
      this->pub_degen_marker_->publish(marker_array);
    }
    return;
  }

  const double eps = 1e-12;

  const Eigen::Vector3d origin = this->degen_info_.p_map_base;
  const double trans_lambda_max = std::max(this->degen_info_.eigvals_trans_dec.maxCoeff(), eps);

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto pushArrow = [&](const int id,
                       const std::string& ns,
                       const Eigen::Vector3d& dir_map,
                       const double length,
                       const Eigen::Vector3f& color) {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = "dlio_map";
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.lifetime = rclcpp::Duration::from_seconds(this->viz_degen_lifetime_);
    m.scale.x = this->viz_degen_shaft_diam_;
    m.scale.y = this->viz_degen_head_diam_;
    m.scale.z = this->viz_degen_head_len_;
    m.color.a = 1.0;
    m.color.r = color.x();
    m.color.g = color.y();
    m.color.b = color.z();

    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = 0.0;
    m.pose.orientation.w = 1.0;

    geometry_msgs::msg::Point p0;
    p0.x = origin.x();
    p0.y = origin.y();
    p0.z = origin.z();

    geometry_msgs::msg::Point p1;
    p1.x = origin.x() + length * dir_map.x();
    p1.y = origin.y() + length * dir_map.y();
    p1.z = origin.z() + length * dir_map.z();

    m.points.push_back(p0);
    m.points.push_back(p1);
    marker_array.markers.push_back(std::move(m));
  };

  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d dir_trans = this->degen_info_.eigvecs_trans_map.col(i).normalized();
    if (!dir_trans.allFinite()) {
      pushDelete(i, "degeneracy_translation");
      continue;
    }

    // Eigenvectors are sign-ambiguous. Keep temporal sign continuity to avoid RViz flicker.
    if (this->degen_prev_dirs_initialized_ && dir_trans.dot(this->degen_prev_trans_dirs_map_[i]) < 0.0) {
      dir_trans = -dir_trans;
    }
    this->degen_prev_trans_dirs_map_[i] = dir_trans;

    if (this->degen_info_.weak_trans[i]) {
      // Visual severity: weaker curvature (smaller lambda / lambda_max) => longer arrow.
      const double ratio = this->degen_info_.eigvals_trans_dec(i) / trans_lambda_max;
      const double severity = std::min(1.0, std::max(0.0, 1.0 - ratio));
      const double length = this->viz_degen_trans_scale_ * (0.35 + 0.65 * severity);
      const bool is_min_mode = (i == 0);
      pushArrow(i, "degeneracy_translation", dir_trans, length,
                Eigen::Vector3f(1.00f, is_min_mode ? 0.10f : 0.55f, 0.10f));
    } else {
      pushDelete(i, "degeneracy_translation");
    }
  }

  this->degen_prev_dirs_initialized_ = true;
  if (hasSubscribers(this->pub_degen_marker_)) {
    this->pub_degen_marker_->publish(marker_array);
  }
}

void dlio::OdomNode::publishDegeneracyStatus(const bool degenerate) {
  if (!hasSubscribers(this->degen_status_pub_)) {
    return;
  }

  std_msgs::msg::Bool status;
  status.data = degenerate;
  this->degen_status_pub_->publish(status);
}

void dlio::OdomNode::createCorrectionMarker(
    const std::string& frame_id,
    const rclcpp::Time& stamp,
    const Eigen::Vector3f& start_m,
    const Eigen::Vector3f& corr_vec_m,
    visualization_msgs::msg::Marker& marker) {
  const Eigen::Vector3f end_m = start_m + static_cast<float>(this->viz_corr_gain_) * corr_vec_m;

  geometry_msgs::msg::Point p0, p1;
  p0.x = start_m.x();
  p0.y = start_m.y();
  p0.z = start_m.z();
  p1.x = end_m.x();
  p1.y = end_m.y();
  p1.z = end_m.z();

  this->corr_marker_points_.push_back(p0);
  this->corr_marker_points_.push_back(p1);
  const size_t max_points = static_cast<size_t>(2 * std::max(1, this->viz_corr_max_segments_));
  if (this->corr_marker_points_.size() > max_points) {
    const auto trim_count =
        static_cast<std::vector<geometry_msgs::msg::Point>::difference_type>(
            this->corr_marker_points_.size() - max_points);
    this->corr_marker_points_.erase(
      this->corr_marker_points_.begin(),
      this->corr_marker_points_.begin() + trim_count);
  }

  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "correction_lines";
  marker.id = 2;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.lifetime = rclcpp::Duration::from_seconds(this->viz_corr_lifetime_);
  marker.scale.x = this->viz_corr_line_width_;

  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.35;
  marker.color.b = 0.0;

  marker.points = this->corr_marker_points_;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
}

void dlio::OdomNode::publishVelocityMarkers(const rclcpp::Time& stamp,
                                            const Eigen::Vector3f& vlin_b,
                                            const Eigen::Vector3f& vang_b) {
  if (!viz_vel_markers_) return;

  visualization_msgs::msg::Marker m_lin, m_ang;
  
  createLinVelocityMarker(this->baselink_frame, stamp, vlin_b, m_lin);
  createAngularVelocityMarker(this->baselink_frame, stamp, vang_b, m_ang);

  if (hasSubscribers(this->pub_lin_vel_marker_)) {
    this->pub_lin_vel_marker_->publish(m_lin);
  }
  if (hasSubscribers(this->pub_ang_vel_marker_)) {
    this->pub_ang_vel_marker_->publish(m_ang);
  }
}

void dlio::OdomNode::createLinVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                                             const Eigen::Vector3f& v_b,
                                             visualization_msgs::msg::Marker& marker) {
  // Arrow
  marker.header.frame_id = frame_id;
  marker.header.stamp    = stamp;
  marker.id   = 0;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Scale and Color
  marker.scale.x = 0.1;  // shaft diameter
  marker.scale.y = 0.2;  // head diameter
  marker.scale.z = 0.2;  // head length
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  // Define Arrow through start and end point
  geometry_msgs::msg::Point startPoint, endPoint;
  startPoint.x = 0.0;  // origin
  startPoint.y = 0.0;  // origin
  startPoint.z = 0.0;  // 0 meter above origin
  endPoint.x = startPoint.x + static_cast<double>(v_b.x());
  endPoint.y = startPoint.y + static_cast<double>(v_b.y());
  endPoint.z = startPoint.z + static_cast<double>(v_b.z());
  marker.points.clear();
  marker.points.push_back(startPoint);
  marker.points.push_back(endPoint);

  // Quaternion for orientation
  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  marker.pose.orientation.x = q.x();
  marker.pose.orientation.y = q.y();
  marker.pose.orientation.z = q.z();
  marker.pose.orientation.w = q.w();
}

void dlio::OdomNode::createAngularVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                                                 const Eigen::Vector3f& w_b,
                                                 visualization_msgs::msg::Marker& marker) {
  // Cylinder to visualize angular velocity as a disc/ring oriented along rotation axis
  marker.header.frame_id = frame_id;
  marker.header.stamp    = stamp;
  marker.ns   = "angular_velocity";
  marker.id   = 1;
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Angular velocity magnitude
  const double angularMagnitude = static_cast<double>(w_b.norm());

  if (angularMagnitude > 1e-6) {
    // Scale based on angular velocity magnitude
    const double baseRadius = std::min(std::max(angularMagnitude * 0.2, 0.1), 0.5);
    marker.scale.x = baseRadius * 2.0;  // diameter in x
    marker.scale.y = baseRadius * 2.0;  // diameter in y
    marker.scale.z = 0.02;              // thin disc height

    // Color: blue for angular velocity with alpha based on magnitude
    marker.color.a = static_cast<float>(std::min(angularMagnitude * 0.5 + 0.3, 1.0));
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
  } else {
    // No significant angular velocity - make marker invisible
    marker.scale.x = 0.0;
    marker.scale.y = 0.0;
    marker.scale.z = 0.0;
    marker.color.a = 0.0;
  }

  // Set lifetime
  marker.lifetime = rclcpp::Duration::from_seconds(0.1);

  // Position at current pose position
  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;

  // Orient the disc perpendicular to the angular velocity vector (rotation axis)
  if (angularMagnitude > 1e-6) {
    Eigen::Vector3d rotationAxis = w_b.cast<double>().normalized();

    // Create a rotation that aligns the cylinder's z-axis with the rotation axis
    // Default cylinder orientation is along z-axis
    Eigen::Vector3d zAxis(0.0, 0.0, 1.0);

    // Calculate rotation to align z-axis with rotation axis
    Eigen::Quaterniond orientation;
    const double dot = rotationAxis.dot(zAxis);
    if (dot > 0.9999) {
      // Already aligned
      orientation = Eigen::Quaterniond::Identity();
    } else if (dot < -0.9999) {
      // Opposite direction - rotate 180 degrees around x-axis
      orientation = Eigen::Quaterniond(0.0, 1.0, 0.0, 0.0);
    } else {
      // General case - use cross product to find rotation axis
      Eigen::Vector3d rotAxis = zAxis.cross(rotationAxis).normalized();
      const double c = std::max(-1.0, std::min(1.0, zAxis.dot(rotationAxis)));
      const double angle = std::acos(c);
      orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotAxis));
    }

    marker.pose.orientation.x = orientation.x();
    marker.pose.orientation.y = orientation.y();
    marker.pose.orientation.z = orientation.z();
    marker.pose.orientation.w = orientation.w();
  } else {
    // Default orientation
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
  }
}

bool dlio::OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  const Eigen::Matrix4f T_corr_guess = Eigen::Matrix4f::Identity();

  bool live_degenerate = false;
  if (this->use_degeneracy_) {
    // Use the current scan's normal spread rather than registration linearization.
    this->analyzeDegeneracyFromCurrentScan(this->T_prior);
    this->publishDegeneracyMarkers(this->scan_header_stamp);

    State scan_candidate;
    State live_candidate;
    {
      std::lock_guard<std::mutex> lock(this->geo.mtx);
      scan_candidate = this->scan_state_prior_valid_ ? this->scan_state_prior : this->state;
      live_candidate = this->state;
    }

    std::string degeneracy_reason;
    live_degenerate = this->currentScanIsDegenerate(scan_candidate, degeneracy_reason);
    if (!live_degenerate) {
      std::string live_reason;
      if (this->stateExceedsDegeneracyBounds(live_candidate, live_reason)) {
        live_degenerate = true;
        degeneracy_reason = "live " + live_reason;
      }
    }

    if (live_degenerate) {
      this->gicp_rematch_trials_latched_ = true;
      this->gicp_freeze_trials_latched_ = false;
      ++this->degen_consecutive_hits_;
      RCLCPP_WARN(this->get_logger(),
                  "[DEGEN] Scan rejected immediately (%d consecutive detections): %s",
                  this->degen_consecutive_hits_, degeneracy_reason.c_str());
      this->enterDegenerateHalt(degeneracy_reason);
      this->publishDegeneracyStatus(true);
      return false;
    } else {
      this->degen_consecutive_hits_ = 0;
    }
  } else {
    this->degen_consecutive_hits_ = 0;
  }

  this->publishDegeneracyStatus(live_degenerate);

  // Start with nearest-neighbor rematching. If a well-constrained scan produces
  // an oversized correction, latch fixed trial correspondences. Once weak
  // geometry is observed, latch rematching instead; fixed correspondences can
  // drag the estimate along an unobservable direction.
  this->gicp.setFreezeTrialCorrespondences(
      this->gicp_freeze_trials_latched_ &&
      !this->gicp_rematch_trials_latched_);
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned, T_corr_guess);

  this->T_corr = this->gicp.getFinalTransformation();
  this->T = this->T_corr * this->T_prior;
  this->gicp_hasConverged = this->gicp.hasConverged();

  if (!live_degenerate && !this->gicp_rematch_trials_latched_ &&
      !this->gicp_freeze_trials_latched_ &&
      this->gicp_freeze_trial_trigger_translation_ > 0.0) {
    const Eigen::Vector3f dynamic_correction =
        this->T.block<3, 1>(0, 3) - this->T_prior.block<3, 1>(0, 3);
    if (!this->T.allFinite() ||
        dynamic_correction.norm() >
            static_cast<float>(this->gicp_freeze_trial_trigger_translation_)) {
      this->gicp_freeze_trials_latched_ = true;
      this->gicp.setFreezeTrialCorrespondences(true);
      this->gicp.align(*aligned, T_corr_guess);
      this->T_corr = this->gicp.getFinalTransformation();
      this->T = this->T_corr * this->T_prior;
      this->gicp_hasConverged = this->gicp.hasConverged();
    }
  }

  if (!this->T.allFinite()) {
    RCLCPP_ERROR(this->get_logger(), "GICP returned a non-finite pose; retaining the IMU prior.");
    this->T = this->T_prior;
    this->T_corr = Eigen::Matrix4f::Identity();
    this->gicp_hasConverged = false;
  }

  // this->computeMotionDeviation();

  // Update next global pose
  // Both source and target clouds are in the global frame now, so transformation is global
  this->propagateGICP();

  // Geometric observer update using LiDAR registration result
  this->updateState();

  // The observer/GICP update itself can drive speed or bias beyond safe bounds.
  // Check the corrected state before processPointCloud can enqueue or publish it.
  if (this->use_degeneracy_) {
    State corrected_state;
    {
      std::lock_guard<std::mutex> lock(this->geo.mtx);
      corrected_state = this->state;
    }
    std::string reason;
    if (this->stateExceedsDegeneracyBounds(corrected_state, reason)) {
      this->enterDegenerateHalt("post-update: " + reason);
      this->publishDegeneracyStatus(true);
      return false;
    }
  }

  return true;
}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  static thread_local boost::circular_buffer<ImuMeas> imu_snapshot;

  {
    std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);

    // pointCloudWorkerLoop() is now responsible for waiting.
    // This function should only snapshot and search.
    if (this->stop_.load(std::memory_order_relaxed) ||
        this->imu_buffer.empty() ||
        this->imu_buffer.front().stamp < end_time) {
      return false;
    }

    imu_snapshot = this->imu_buffer;
  }

  if (imu_snapshot.empty()) {
    return false;
  }

  auto imu_it = imu_snapshot.begin();
  auto last_imu_it = imu_it;
  ++imu_it;

  while (imu_it != imu_snapshot.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    ++imu_it;
  }

  while (imu_it != imu_snapshot.end() && imu_it->stamp >= start_time) {
    ++imu_it;
  }

  if (imu_it == imu_snapshot.end()) {
    return false;
  }
  ++imu_it;

  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

bool dlio::OdomNode::predictScanState(double target_time) {
  State anchor;
  double anchor_stamp = 0.0;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    if (!this->scan_state_valid_) {
      this->scan_state_prior = this->state;
      this->scan_state_prior_stamp_ = target_time;
      this->scan_state_prior_valid_ = true;
      return false;
    }
    anchor = this->scan_state;
    anchor_stamp = this->scan_state_stamp_;
  }

  State predicted;
  const auto frames = this->integrateImu(
      anchor_stamp, anchor.q, anchor.p, anchor.v.lin.w, {target_time},
      anchor.b, &predicted);
  const bool success = frames.size() == 1;

  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    this->scan_state_prior = success ? predicted : anchor;
    this->scan_state_prior_stamp_ = target_time;
    this->scan_state_prior_valid_ = true;
  }

  if (!success) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Unable to propagate observer state from %.9f to scan time %.9f; "
        "using the previous corrected scan state as the observer prior.",
        anchor_stamp, target_time);
  }
  return success;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps,
                             const ImuBias& bias, State* state_at_first_timestamp) {
  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return {};
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return {};
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  const double dt = f2.dt;
  if (dt <= 0.0) {
    return {};
  }
  const float dtf = static_cast<float>(dt);

  // Time between first IMU sample and start_time
  const double idt = start_time - f1.stamp;
  const float idtf = static_cast<float>(idt);
  const float gravity = static_cast<float>(this->gravity_);

  // Angular acceleration between first two IMU samples
  const Eigen::Vector3f omega1 = f1.ang_vel - bias.gyro;
  const Eigen::Vector3f omega2 = f2.ang_vel - bias.gyro;
  Eigen::Vector3f alpha_dt = omega2 - omega1;
  Eigen::Vector3f alpha = alpha_dt / dtf;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(omega1 + 0.5f * alpha * idtf);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5f * ( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idtf,
    q_init.x() + 0.5f * ( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idtf,
    q_init.y() + 0.5f * ( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idtf,
    q_init.z() + 0.5f * ( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idtf
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = omega1 + 0.5f * alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5f * ( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dtf,
    q_init.x() + 0.5f * ( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dtf,
    q_init.y() + 0.5f * ( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dtf,
    q_init.z() + 0.5f * ( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dtf
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel - bias.accel);
  a1[2] -= gravity;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel - bias.accel);
  a2[2] -= gravity;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dtf;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1 * idtf + 0.5f * j * idtf * idtf;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init * idtf + 0.5f * a1 * idtf * idtf + (1.0f / 6.0f) * j * idtf * idtf * idtf;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, bias,
                                    state_at_first_timestamp, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(const Eigen::Quaternionf& q_init, const Eigen::Vector3f& p_init,
                                     const Eigen::Vector3f& v_init, const std::vector<double>& sorted_timestamps,
                                     const ImuBias& bias, State* state_at_first_timestamp,
                                     const boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                                     const boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel - bias.accel);
  const float gravity = static_cast<float>(this->gravity_);
  a[2] -= gravity;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    const double dt = f.dt;
    if (dt <= 0.0) {
      prev_imu_it = imu_it;
      continue;
    }
    const float dtf = static_cast<float>(dt);

    // Angular acceleration
    const Eigen::Vector3f omega0 = f0.ang_vel - bias.gyro;
    const Eigen::Vector3f omega1 = f.ang_vel - bias.gyro;
    Eigen::Vector3f alpha_dt = omega1 - omega0;
    Eigen::Vector3f alpha = alpha_dt / dtf;

    // Average angular velocity
    Eigen::Vector3f omega = omega0 + 0.5f * alpha_dt;

    const Eigen::Quaternionf q0 = q;

    // Orientation at current IMU sample
    q = Eigen::Quaternionf (
      q0.w() - 0.5f * ( q0.x()*omega[0] + q0.y()*omega[1] + q0.z()*omega[2] ) * dtf,
      q0.x() + 0.5f * ( q0.w()*omega[0] - q0.z()*omega[1] + q0.y()*omega[2] ) * dtf,
      q0.y() + 0.5f * ( q0.z()*omega[0] + q0.w()*omega[1] - q0.x()*omega[2] ) * dtf,
      q0.z() + 0.5f * ( q0.x()*omega[1] - q0.y()*omega[0] + q0.w()*omega[2] ) * dtf
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel - bias.accel);
    a[2] -= gravity;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dtf;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      const double idt = *stamp_it - f0.stamp;
      const float idtf = static_cast<float>(idt);

      // Average angular velocity
      Eigen::Vector3f omega_i = omega0 + 0.5f * alpha * idtf;

      // Orientation at interpolated timestamp (must start from q0, not q)
      Eigen::Quaternionf q_i (
        q0.w() - 0.5f * ( q0.x()*omega_i[0] + q0.y()*omega_i[1] + q0.z()*omega_i[2] ) * idtf,
        q0.x() + 0.5f * ( q0.w()*omega_i[0] - q0.z()*omega_i[1] + q0.y()*omega_i[2] ) * idtf,
        q0.y() + 0.5f * ( q0.z()*omega_i[0] + q0.w()*omega_i[1] - q0.x()*omega_i[2] ) * idtf,
        q0.z() + 0.5f * ( q0.x()*omega_i[1] - q0.y()*omega_i[0] + q0.w()*omega_i[2] ) * idtf
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v * idtf + 0.5f * a0 * idtf * idtf + (1.0f / 6.0f) * j * idtf * idtf * idtf;

      if (state_at_first_timestamp != nullptr && imu_se3.empty()) {
        const Eigen::Vector3f v_i = v + a0 * idtf + 0.5f * j * idtf * idtf;
        state_at_first_timestamp->p = p_i;
        state_at_first_timestamp->q = q_i;
        state_at_first_timestamp->v.lin.w = v_i;
        state_at_first_timestamp->v.lin.b = q_i.toRotationMatrix().transpose() * v_i;
        state_at_first_timestamp->v.ang.b = omega_i;
        state_at_first_timestamp->v.ang.w = q_i.toRotationMatrix() * omega_i;
        state_at_first_timestamp->b = bias;
      }

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      ++stamp_it;
    }

    // Position
    p += v * dtf + 0.5f * a0 * dtf * dtf + (1.0f / 6.0f) * j_dt * dtf * dtf;

    // Velocity
    v += a0 * dtf + 0.5f * j_dt * dtf;

    prev_imu_it = imu_it;

  }
  return imu_se3;
}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  const float norm = std::sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

bool dlio::OdomNode::propagateState(const ImuMeas& imu) {
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  if (!this->geo.first_opt_done) {
    return false;
  }

  const double dt = this->live_state_stamp_ > 0.0
      ? imu.stamp - this->live_state_stamp_
      : imu.dt;
  if (dt <= 0.0) {
    return false;
  }
  const float dtf = static_cast<float>(dt);
  const float half_dt_sq = 0.5f * dtf * dtf;

  const Eigen::Vector3f f_b = imu.lin_accel - this->state.b.accel;

  // Rotation and gravity (world frame)
  const Eigen::Matrix3f Rwb = this->state.q.toRotationMatrix();
  const Eigen::Vector3f g_w(0.0f, 0.0f, static_cast<float>(this->gravity_));

  // World-frame acceleration
  const Eigen::Vector3f a_w = Rwb * f_b - g_w;

  // Integrate p, v (world frame)
  this->state.p      += this->state.v.lin.w * dtf + a_w * half_dt_sq;
  this->state.v.lin.w += a_w * dtf;
  this->state.v.lin.b  = Rwb.transpose() * this->state.v.lin.w;

  // Integrate attitude with measured (bias-corrected) omega (body frame)
  const Eigen::Vector3f omega_b = this->state.v.ang.b = imu.ang_vel - this->state.b.gyro;
  const Eigen::Quaternionf omega_q(0.f, omega_b.x(), omega_b.y(), omega_b.z());
  Eigen::Quaternionf qdot = (this->state.q * omega_q);
  this->state.q.coeffs() += 0.5f * dtf * qdot.coeffs();
  this->state.q.normalize();

  // Update angular velocity in world for later use
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
  this->live_state_stamp_ = imu.stamp;
  return true;
}

void dlio::OdomNode::updateState() {

  // Freeze callback-side buffering/propagation while correcting the scan-time
  // state and replaying to the newest buffered IMU sample. Lock order matches
  // callbackImu(): callback serialization first, observer state second.
  std::lock_guard<std::mutex> imu_callback_lock(this->mtx_imu_callback_);
  std::lock_guard<std::mutex> state_lock(this->geo.mtx);

  if (!this->scan_state_prior_valid_ ||
      std::abs(this->scan_state_prior_stamp_ - this->scan_stamp) > 1e-6) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Missing scan-time observer prior: scan=%.9f prior=%.9f valid=%s.",
        this->scan_stamp, this->scan_state_prior_stamp_,
        this->scan_state_prior_valid_ ? "true" : "false");
    this->scan_state_prior = this->scan_state_valid_ ? this->scan_state : this->state;
    this->scan_state_prior_stamp_ = this->scan_stamp;
    this->scan_state_prior_valid_ = true;
  }

  State corrected = this->scan_state_prior;

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;

  double dt = this->scan_stamp - this->prev_scan_stamp;
  const double dt_max =
      this->pointcloud_expected_period_ + this->pointcloud_period_tolerance_;
  if (!std::isfinite(dt) || dt <= 0.0 || dt > dt_max) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Observer step out of range (scan=%.6f prev_scan=%.6f dt=%.6f s, max %.6f s); "
        "using the nominal LiDAR period %.6f s.",
        this->scan_stamp, this->prev_scan_stamp, dt, dt_max,
        this->pointcloud_expected_period_);
    dt = this->pointcloud_expected_period_;
  }
  const float dtf = static_cast<float>(dt);

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = corrected.q.normalized();
  const Eigen::Quaternionf qhat_conj = qhat.conjugate();

  // Constuct error quaternion
  qe = qhat_conj * qin;

  float sgn = 1.0f;
  if (qe.w() < 0) {
    sgn = -1.0f;
  }

  // Construct quaternion correction
  qcorr.w() = 1.0f - std::abs(qe.w());
  qcorr.vec() = sgn * qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - corrected.p;
  Eigen::Vector3f err_body;

  err_body = qhat_conj._transformVector(err);

  const float abias_max = static_cast<float>(this->geo_abias_max_);
  const float gbias_max = static_cast<float>(this->geo_gbias_max_);
  const float kab = static_cast<float>(this->geo_Kab_);
  const float kgb = static_cast<float>(this->geo_Kgb_);
  const float kp = static_cast<float>(this->geo_Kp_);
  const float kv = static_cast<float>(this->geo_Kv_);
  const float kq = static_cast<float>(this->geo_Kq_);

  // Update accel bias
  corrected.b.accel -= dtf * kab * err_body;
  corrected.b.accel = corrected.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  corrected.b.gyro[0] -= dtf * kgb * qe.w() * qe.x();
  corrected.b.gyro[1] -= dtf * kgb * qe.w() * qe.y();
  corrected.b.gyro[2] -= dtf * kgb * qe.w() * qe.z();
  corrected.b.gyro = corrected.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  corrected.p += dtf * kp * err;
  corrected.v.lin.w += dtf * kv * err;

  corrected.q.w() += dtf * kq * qcorr.w();
  corrected.q.x() += dtf * kq * qcorr.x();
  corrected.q.y() += dtf * kq * qcorr.y();
  corrected.q.z() += dtf * kq * qcorr.z();
  corrected.q.normalize();

  // Recompute body-frame velocity to match the corrected world velocity and orientation.
  corrected.v.lin.b = corrected.q.toRotationMatrix().transpose() * corrected.v.lin.w;
  corrected.v.ang.w = corrected.q.toRotationMatrix() * corrected.v.ang.b;

  // Save the corrected observer anchor at the LiDAR scan reference time.
  this->scan_state = corrected;
  this->scan_state_stamp_ = this->scan_stamp;
  this->scan_state_valid_ = true;
  this->scan_state_prior_valid_ = false;
  this->geo.prev_p = corrected.p;
  this->geo.prev_q = corrected.q;
  this->geo.prev_vel = corrected.v.lin.w;

  // Rebuild the live state at the latest received IMU timestamp using the new
  // scan-time correction and bias. callbackImu is frozen, so no measurement
  // can be both included here and propagated again after the locks are released.
  double latest_imu_stamp = -1.0;
  {
    std::lock_guard<decltype(this->mtx_imu)> imu_lock(this->mtx_imu);
    if (!this->imu_buffer.empty()) {
      latest_imu_stamp = this->imu_buffer.front().stamp;
    }
  }

  this->state = corrected;
  this->live_state_stamp_ = this->scan_stamp;
  if (latest_imu_stamp > this->scan_stamp) {
    State replayed;
    const auto replay_frames = this->integrateImu(
        this->scan_stamp, corrected.q, corrected.p, corrected.v.lin.w,
        {latest_imu_stamp}, corrected.b, &replayed);
    if (replay_frames.size() == 1) {
      this->state = replayed;
      this->live_state_stamp_ = latest_imu_stamp;
    } else {
      RCLCPP_ERROR(
          this->get_logger(),
          "IMU replay failed after scan correction: scan=%.9f latest_imu=%.9f.",
          this->scan_stamp, latest_imu_stamp);
    }
  }
  this->geo.first_opt_done = true;

}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::transformImu(
    const sensor_msgs::msg::Imu::SharedPtr& imu_raw,
    const std::optional<double>& dt) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  const bool have_prev_transform = dt.has_value();
  const float dtf = have_prev_transform ? static_cast<float>(*dt) : 1.0f;

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(static_cast<float>(imu_raw->angular_velocity.x),
                          static_cast<float>(imu_raw->angular_velocity.y),
                          static_cast<float>(imu_raw->angular_velocity.z));

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  const Eigen::Vector3f ang_vel_cg_prev = have_prev_transform ? this->imu_transform_ang_vel_prev_ : ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(static_cast<float>(imu_raw->linear_acceleration.x),
                            static_cast<float>(imu_raw->linear_acceleration.y),
                            static_cast<float>(imu_raw->linear_acceleration.z));

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dtf).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  this->imu_transform_ang_vel_prev_ = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness() {

  if (!this->original_scan || this->original_scan->empty()) {
    return;
  }

  // compute range of points
  std::vector<float> ds;
  ds.reserve(this->original_scan->points.size());

  for (const auto& point : this->original_scan->points) {
    const float d = std::sqrt(point.x * point.x + point.y * point.y);
    ds.push_back(d);
  }

  if (ds.empty()) {
    return;
  }

  // median
  const auto median_index =
      static_cast<std::vector<float>::difference_type>(ds.size() / 2U);
  std::nth_element(ds.begin(), ds.begin() + median_index, ds.end());
  float median_curr = ds[static_cast<std::size_t>(median_index)];
  if (!this->spaciousness_lpf_initialized_) {
    this->spaciousness_lpf_prev_ = median_curr;
    this->spaciousness_lpf_initialized_ = true;
  }
  float median_lpf = 0.95f * this->spaciousness_lpf_prev_ + 0.05f * median_curr;
  this->spaciousness_lpf_prev_ = median_lpf;

  std::lock_guard<std::mutex> lock(g_metrics_mutex);

  // push
  this->metrics.spaciousness.push_back( median_lpf );
  if (this->metrics.spaciousness.size() > 400) {
    this->metrics.spaciousness.pop_front();
  }

  if (this->metrics.density.size() > 400) {
    this->metrics.density.pop_front();
  }

}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  if (!this->density_lpf_initialized_) {
    this->density_lpf_prev_ = density;
    this->density_lpf_initialized_ = true;
  }
  float density_lpf = 0.95f * this->density_lpf_prev_ + 0.05f * density;
  this->density_lpf_prev_ = density_lpf;

  std::lock_guard<std::mutex> lock(g_metrics_mutex);
  this->metrics.density.push_back( density_lpf );

}

void dlio::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::updateKeyframes() {
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  if (this->keyframes.empty()) {
    this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
    this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.emplace_back(this->T_corr);
    return;
  }

  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;
  int num_nearby = 0;

  for (const auto& k : this->keyframes) {
    float dx = this->lidarPose.p[0] - k.first.first[0];
    float dy = this->lidarPose.p[1] - k.first.first[1];
    float dz = this->lidarPose.p[2] - k.first.first[2];
    float delta_d = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5f) ++num_nearby;
    if (delta_d < closest_d) { closest_d = delta_d; closest_idx = keyframes_idx; }
    ++keyframes_idx;
  }

  const Eigen::Vector3f&    closest_pose   = this->keyframes[closest_idx].first.first;
  const Eigen::Quaternionf& closest_pose_r = this->keyframes[closest_idx].first.second;

  const float dd = closest_d;

  Eigen::Quaternionf dq;
  if (this->lidarPose.q.dot(closest_pose_r) < 0.f) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w()*=-1.f; lq.x()*=-1.f; lq.y()*=-1.f; lq.z()*=-1.f;
    dq = this->lidarPose.q * lq.inverse();
  } else {
    dq = this->lidarPose.q * closest_pose_r.inverse();
  }

  const double theta_rad = 2.0 * std::atan2(std::sqrt(dq.x()*dq.x()+dq.y()*dq.y()+dq.z()*dq.z()), dq.w());
  const double theta_deg = theta_rad * (180.0 / M_PI);

  bool newKeyframe = false;
  if (std::abs(dd) > this->keyframe_thresh_dist_ || std::abs(theta_deg) > this->keyframe_thresh_rot_) newKeyframe = true;
  if (std::abs(dd) <= this->keyframe_thresh_dist_) newKeyframe = false;
  if (std::abs(dd) <= this->keyframe_thresh_dist_ && std::abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) newKeyframe = true;

  if (newKeyframe) {
    if (this->keyframes.size() >= kMaxKeyframes) {
      const std::size_t removed = this->keyframes.size() - (kMaxKeyframes - 1);

      const auto removed_diff =
          static_cast<std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                                            pcl::PointCloud<PointType>::ConstPtr>>::difference_type>(removed);
      const auto removed_timestamps_diff =
          static_cast<std::vector<rclcpp::Time>::difference_type>(removed);
      const auto removed_normals_diff =
          static_cast<std::vector<std::shared_ptr<const nano_gicp::CovarianceList>>::difference_type>(removed);
      const auto removed_transforms_diff =
          static_cast<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>::difference_type>(removed);

      this->keyframes.erase(this->keyframes.begin(), this->keyframes.begin() + removed_diff);
      this->keyframe_timestamps.erase(this->keyframe_timestamps.begin(), this->keyframe_timestamps.begin() + removed_timestamps_diff);
      this->keyframe_normals.erase(this->keyframe_normals.begin(), this->keyframe_normals.begin() + removed_normals_diff);
      this->keyframe_transformations.erase(this->keyframe_transformations.begin(), this->keyframe_transformations.begin() + removed_transforms_diff);

      this->onKeyframesTrim(removed);   // <<< keep all index-based state consistent

      if (removed >= 16) { // only when we dropped a chunk
        keyframes.shrink_to_fit();
        keyframe_timestamps.shrink_to_fit();
        keyframe_normals.shrink_to_fit();
        keyframe_transformations.shrink_to_fit();
      }
    }

    this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
    this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.emplace_back(this->T_corr);
  }

}

void dlio::OdomNode::onKeyframesTrim(std::size_t removed) {
  if (removed == 0) return;

  // num_processed_keyframes tracks how many keyframes have been transformed/published
  if (this->num_processed_keyframes <= removed) this->num_processed_keyframes = 0;
  else                                          this->num_processed_keyframes -= static_cast<int>(removed);

  auto shift_down = [removed](std::vector<int>& idxs) {
    const int r = static_cast<int>(removed);
    int w = 0;
    for (int i = 0; i < idxs.size(); ++i) {
      const int v = idxs[i] - r;
      if (v >= 0) idxs[w++] = v;    // keep only still-valid indices
    }
    idxs.resize(w);
  };

  shift_down(this->submap_kf_idx_prev);
  shift_down(this->submap_kf_idx_curr);
  shift_down(this->keyframe_convex);
  shift_down(this->keyframe_concave);
}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < this->adaptive_sp_min_) { sp = this->adaptive_sp_min_; }
  if (sp > this->adaptive_sp_max_) { sp = this->adaptive_sp_max_; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();
  const float den_min = this->adaptive_den_factor_min_ * static_cast<float>(this->gicp_max_corr_dist_);
  const float den_max = this->adaptive_den_factor_max_ * static_cast<float>(this->gicp_max_corr_dist_);

  if (den < den_min) { den = den_min; }
  if (den > den_max) { den = den_max; }

  if (sp < this->adaptive_sp_max_) { den = den_min; }
  if (sp > this->adaptive_sp_max_) { den = den_max; }

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty and k is valid
  if (dists.empty() || k <= 0) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  if (pq.empty()) {
    return;
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void dlio::OdomNode::buildSubmap(const State& vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    const Eigen::Vector3f delta = vehicle_state.p - this->keyframes[i].first.first;
    const float d = delta.norm();
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  std::vector<int> convex_frames;
  for (const auto& c : this->keyframe_convex) {
    if (c >= 0 && c < ds.size()) {
      convex_ds.push_back(ds[c]);
      convex_frames.push_back(c);
    }
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, convex_frames);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  std::vector<int> concave_frames;
  for (const auto& c : this->keyframe_concave) {
    if (c >= 0 && c < ds.size()) {
      concave_ds.push_back(ds[c]);
      concave_frames.push_back(c);
    }
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, concave_frames);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    {
      std::unique_lock<decltype(this->keyframes_mutex)> submap_lock(this->keyframes_mutex);
      for (auto k : this->submap_kf_idx_curr) {
        if (k < 0 || k >= static_cast<int>(this->keyframes.size()) ||
            k >= static_cast<int>(this->keyframe_normals.size())) {
          continue;
        }

        // create current submap cloud
        *submap_cloud_ += *this->keyframes[k].second;

        // grab corresponding submap cloud's normals
        submap_normals_->insert( std::end(*submap_normals_),
            std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
      }
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(const State& vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](const Eigen::Matrix4d& cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publishKeyframe(this->keyframes[i], this->keyframe_timestamps[i]);
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running || this->shouldStop(); });
}

void dlio::OdomNode::debug() {

  // Only one debug() call may run at a time: concurrent calls race on cpu_percents
  // and the CPU usage sampling state (lastCPU/lastSysCPU/lastUserCPU).
  std::unique_lock<std::mutex> debug_lock(this->mtx_debug_, std::try_to_lock);
  if (!debug_lock.owns_lock()) {
    return;
  }

  // length_traversed is already maintained incrementally in processPointCloud().
  const double length_traversed = this->length_traversed;

  // Snapshot comp_times under lock to avoid racing with processPointCloud().
  std::vector<double> comp_snapshot;
  {
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    comp_snapshot.reserve(this->comp_times.size());
    for (const auto& [ts, ct] : this->comp_times) {
      comp_snapshot.push_back(ct);
    }
  }

  const double avg_comp_time = comp_snapshot.empty() ? 0.0 :
    std::accumulate(comp_snapshot.begin(), comp_snapshot.end(), 0.0) /
        static_cast<double>(comp_snapshot.size());
  const double max_comp_time = comp_snapshot.empty() ? 0.0 :
    *std::max_element(comp_snapshot.begin(), comp_snapshot.end());
  const double last_comp_time = comp_snapshot.empty() ? 0.0 : comp_snapshot.back();

  // RAM Usage
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  const long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  resident_set = static_cast<double>(rss) * static_cast<double>(page_size_kb);

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = static_cast<double>(timeSample.tms_stime - this->lastSysCPU) +
                    static_cast<double>(timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= static_cast<double>(now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  this->cpu_percents.push_back(cpu_percent);
  if (this->cpu_percents.size() > 400) {
    this->cpu_percents.pop_front();
  }
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) /
        static_cast<double>(this->cpu_percents.size());

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << '\n'
            << "+-------------------------------------------------------------------+\n";
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << '\n';
  std::cout << "+-------------------------------------------------------------------+\n";

  const std::time_t curr_time = static_cast<std::time_t>(this->scan_stamp);
  std::tm tm_info{};
  localtime_r(&curr_time, &tm_info);
  std::array<char, 32> time_buf{};
  std::strftime(time_buf.data(), time_buf.size(), "%a %b %d %H:%M:%S %Y", &tm_info);
  const std::string asc_time(time_buf.data());
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << last_comp_time*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << max_comp_time*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::callbackExternalOdom(nav_msgs::msg::Odometry::SharedPtr odom)  // NOLINT(performance-unnecessary-value-param)
{
  std::unique_lock<std::mutex> lock(this->mtx_external_odom);

  this->prevExternalOdomPose = this->externalOdomPose;

  this->externalOdomPose.p = Eigen::Vector3f(
      static_cast<float>(odom->pose.pose.position.x),
      static_cast<float>(odom->pose.pose.position.y),
      static_cast<float>(odom->pose.pose.position.z));
  this->externalOdomPose.q = Eigen::Quaternionf(
      static_cast<float>(odom->pose.pose.orientation.w),
      static_cast<float>(odom->pose.pose.orientation.x),
      static_cast<float>(odom->pose.pose.orientation.y),
      static_cast<float>(odom->pose.pose.orientation.z));

  if (!this->first_external_odom_received) {
    // Seed DLIO's world pose from the absolute Gazebo ground-truth pose on first message.
    this->T = Eigen::Matrix4f::Identity();
    this->T.block<3,3>(0,0) = this->externalOdomPose.q.toRotationMatrix();
    this->T.block<3,1>(0,3) = this->externalOdomPose.p;

    this->prevExternalOdomPose = this->externalOdomPose;
    this->first_external_odom_received = true;
  }

}
