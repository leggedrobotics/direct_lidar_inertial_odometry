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

#include "dlio/dlio.h"
#include "dlio/imu_timestamp_tracker.h"
#include "dlio/ring_range_filter.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

#include <array>
#include <condition_variable>
#include <deque>

class dlio::OdomNode: public rclcpp::Node {

public:

  OdomNode();
  ~OdomNode() override;

  void start();
  void requestStop();

private:

  struct State;
  struct ImuMeas;
  struct ImuBias;
  struct InitialImuBaseline;

  void getParams();

  void callbackPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr pc);  // NOLINT(performance-unnecessary-value-param)
  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void callbackImu(sensor_msgs::msg::Imu::SharedPtr imu);  // NOLINT(performance-unnecessary-value-param)
  void resetService(std::shared_ptr<std_srvs::srv::Trigger::Request> req,  // NOLINT(performance-unnecessary-value-param)
                    std::shared_ptr<std_srvs::srv::Trigger::Response> res);  // NOLINT(performance-unnecessary-value-param)
  void captureInitialImuBaseline(const Eigen::Vector3f& gravity_vec,
                                 const Eigen::Quaternionf& gravity_align_q,
                                 const Eigen::Vector3f& accel_bias,
                                 const Eigen::Vector3f& gyro_bias);
  bool beginPendingReset();
  void finishPendingReset(bool success, const std::string& message);
  void performReset();
  bool triggerInternalReset(const std::string& reason);
  void requestMapReset(const std::string& origin);
  bool scanPassesGeometryGate(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  std::size_t filterPointCloudByRingRange(sensor_msgs::msg::PointCloud2& pc);
  bool shouldStop();

void publishToROS(const pcl::PointCloud<PointType>::ConstPtr& published_cloud,
                  const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                  const Eigen::Ref<const Eigen::Matrix4f>& T_all,
                  double scanStamp,
                  const Eigen::Vector3f& state_p_scan,
                  const Eigen::Quaternionf& state_q_scan,
                  const Eigen::Vector3f& state_vlin_b_scan,
                  const Eigen::Vector3f& state_vang_b_scan);

void publishCloud(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                  const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                  const Eigen::Ref<const Eigen::Matrix4f>& T_all,
                  const Eigen::Ref<const Eigen::Matrix4f>& T_map_odom,
                  const rclcpp::Time& cloud_stamp);
                  
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void callbackExternalOdom(nav_msgs::msg::Odometry::SharedPtr odom);  // NOLINT(performance-unnecessary-value-param)

  bool getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  bool predictScanState(double target_time);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps, const ImuBias& bias,
                 State* state_at_first_timestamp = nullptr);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(const Eigen::Quaternionf& q_init, const Eigen::Vector3f& p_init, const Eigen::Vector3f& v_init,
                         const std::vector<double>& sorted_timestamps, const ImuBias& bias,
                         State* state_at_first_timestamp,
                         const boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                         const boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  void propagateGICP();

  bool propagateState(const ImuMeas& imu);
  void updateState();

  void setAdaptiveParams();
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();
  void computeMotionDeviation();

  sensor_msgs::msg::Imu::SharedPtr transformImu(
      const sensor_msgs::msg::Imu::SharedPtr& imu,
      const std::optional<double>& dt);

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(const State& vehicle_state);
  void buildKeyframesAndSubmap(const State& vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void publishPoseSnapshot();
  void onKeyframesTrim(std::size_t removed);
  // Velocity markers
  void publishVelocityMarkers(const rclcpp::Time& stamp,
                              const Eigen::Vector3f& vlin_b,  // NOLINT(bugprone-easily-swappable-parameters)
                              const Eigen::Vector3f& vang_b);
  void publishCorrectionMarker(const rclcpp::Time& stamp,
                               const Eigen::Ref<const Eigen::Matrix4f>& T_corr,
                               const Eigen::Ref<const Eigen::Matrix4f>& T_all);
  void analyzeDegeneracyFromCurrentScan(const Eigen::Ref<const Eigen::Matrix4f>& T_map_base);

  void publishDegeneracyMarkers(const rclcpp::Time& stamp);
  void createLinVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                               const Eigen::Vector3f& v_b,
                               visualization_msgs::msg::Marker& out);
  void createAngularVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                                   const Eigen::Vector3f& w_b,
                                   visualization_msgs::msg::Marker& out);
  void createCorrectionMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                              const Eigen::Vector3f& start_m,
                              const Eigen::Vector3f& corr_vec_m,
                              visualization_msgs::msg::Marker& out);

  void debug();

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, reset_srv_cb_group_, external_odom_cb_group;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr external_odom_sub;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_odom_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_map_prop_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_not_transformed_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_map_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_map_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_baselink_pub;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr map_reset_client_;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_lin_vel_marker_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_ang_vel_marker_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_corr_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_degen_marker_;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  nav_msgs::msg::Path path_odom_ros;
  nav_msgs::msg::Path path_map_prop_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Deque-backed pose histories for O(1) front-removal.
  // Copied into the corresponding Path message only when subscribers exist.
  std::deque<geometry_msgs::msg::PoseStamped> path_poses_;
  std::deque<geometry_msgs::msg::PoseStamped> path_odom_poses_;
  std::deque<geometry_msgs::msg::PoseStamped> path_map_prop_poses_;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::future<void> debug_future_;

  // Pointcloud rate estimation (used when debug is disabled)
  std::deque<std::chrono::steady_clock::time_point> pc_rate_window_;
  std::chrono::steady_clock::time_point pc_rate_last_print_;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;

  std::size_t kMaxKeyframes = 30;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  dlio::SensorType sensor = dlio::SensorType::UNKNOWN;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready = false;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running = false;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  // Each entry: {scan_stamp (s), computation_time (s)}.
  // Trimmed to a rolling 2-second window by processPointCloud().
  std::deque<std::pair<double, double>> comp_times;
  std::mutex mtx_comp_times_;
  std::mutex mtx_debug_; // prevents concurrent debug() calls from racing on cpu state
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_self_; // dedicated to restart geometry gate

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;
  // Latest scan-time map->odom used to map IMU-propagated lidar poses into dlio_map.
  Eigen::Matrix4f T_map_odom_latest = Eigen::Matrix4f::Identity();
  bool has_T_map_odom_latest = false;
  std::mutex mtx_T_map_odom_latest;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  rclcpp::Time imu_stamp;
  double first_imu_stamp;
  double imu_dp, imu_dq_deg;

  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    // Transformed/calibrated body-frame measurements with bias still present.
    // Bias is applied explicitly at integration time so callback scheduling
    // cannot change a scan's IMU trajectory.
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  }; ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  // Serializes callback-side IMU timing/transform state with internal reset.
  std::mutex mtx_imu_callback_;
  std::condition_variable cv_imu_stamp;

  ImuTimestampTracker imu_input_timestamps_;
  ImuTimestampTracker imu_integration_timestamps_;

  // Resettable IMU calibration accumulation state.
  int imu_calib_samples_ = 0;
  Eigen::Vector3f imu_calib_gyro_sum_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f imu_calib_accel_sum_ = Eigen::Vector3f::Zero();
  bool imu_calib_printed_ = false;

  // Resettable IMU frame-transform history.
  Eigen::Vector3f imu_transform_ang_vel_prev_ = Eigen::Vector3f::Zero();

  static bool comparatorImu(const ImuMeas& m1, const ImuMeas& m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  // The observer correction is defined at scan_stamp. Keep that corrected
  // state separate from the live IMU-rate state, which may be newer.
  State scan_state;
  State scan_state_prior;
  double scan_state_stamp_ = 0.0;
  double scan_state_prior_stamp_ = 0.0;
  double live_state_stamp_ = 0.0;
  bool scan_state_valid_ = false;
  bool scan_state_prior_valid_ = false;

  struct InitialImuBaseline {
    bool valid = false;
    Eigen::Vector3f gravity_vec = Eigen::Vector3f::Zero();
    float gravity_norm = 0.0f;
    Eigen::Quaternionf gravity_align_q = Eigen::Quaternionf::Identity();
    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
  }; InitialImuBaseline initial_imu_baseline_;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // External odometry (replaces IMU for initialization and T_prior in sim mode)
  Pose externalOdomPose;
  Pose prevExternalOdomPose;
  bool first_external_odom_received;
  std::mutex mtx_external_odom;

  // Metrics
  struct Metrics {
    std::deque<float> spaciousness;
    std::deque<float> density;
    std::vector<float> motion_deviation;
  }; Metrics metrics;

  bool spaciousness_lpf_initialized_ = false;
  float spaciousness_lpf_prev_ = 0.0f;
  bool density_lpf_initialized_ = false;
  float density_lpf_prev_ = 0.0f;

  struct DegeneracyInfo {
    bool valid = false;
    // Eigenpairs of the raw scan normal-spread matrix. Small eigenvalues
    // indicate weak translation observability along the corresponding directions.
    Eigen::Vector3d eigvals_trans_dec = Eigen::Vector3d::Zero();
    Eigen::Matrix3d eigvecs_trans_map = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_map_base = Eigen::Vector3d::Zero();
    std::array<bool, 3> weak_trans{{false, false, false}};
    double trans_condition = 0.0;
  }; DegeneracyInfo degen_info_;

  std::string cpu_type;
  std::deque<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;

  bool debug_enabled_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;
  float adaptive_sp_min_;
  float adaptive_sp_max_;
  float adaptive_den_factor_min_;
  float adaptive_den_factor_max_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool ring_range_filter_enabled_ = false;
  bool ring_range_filter_reported_ = false;
  std::array<float, dlio::kRingRangeCount> ring_range_squared_{};

  bool vf_use_;
  double vf_res_;
  int pointcloud_queue_size_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;
  double gicp_freeze_trial_trigger_translation_ = 0.0;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

  bool   viz_vel_markers_ = true;
  double viz_lin_gain_ = 1.0;              // [m per (m/s)] arrow length gain
  double viz_ang_radius_gain_ = 1.0;       // [m per (rad/s)]
  double viz_ang_radius_min_ = 0.1;
  double viz_ang_radius_max_ = 1.0;
  double viz_disc_thickness_ = 0.01;        // [m]
  double viz_marker_lifetime_ = 1.0;       // [s]
  bool   viz_corr_marker_ = true;
  double viz_corr_gain_ = 1.0;             // [m per m] correction line gain
  double viz_corr_line_width_ = 0.03;      // [m]
  int    viz_corr_max_segments_ = 2000;    // number of stored correction segments
  double viz_corr_lifetime_ = 0.0;         // [s], 0 keeps full correction history visible
  std::vector<geometry_msgs::msg::Point> corr_marker_points_;

  // Translation-only degeneracy analysis and visualization from the
  // raw scan normal-spread matrix.
  bool   use_degeneracy_ = false;
  double degen_trans_eig_abs_thresh_ = 200.0;
  int    degen_reset_consecutive_count_ = 5;
  int    degen_consecutive_hits_ = 0;
  bool   gicp_freeze_trials_latched_ = false;
  bool   gicp_rematch_trials_latched_ = false;

  // Restart geometry gate threshold on the weakest axis of the scan's
  // local-normal scatter matrix.
  bool   restart_gate_enabled_ = true;
  double restart_gate_min_eigenvalue_ = 50.0;

  bool   viz_degen_marker_ = true;
  double viz_degen_trans_scale_ = 0.75;
  double viz_degen_shaft_diam_ = 0.03;
  double viz_degen_head_diam_ = 0.06;
  double viz_degen_head_len_ = 0.10;
  double viz_degen_lifetime_ = 0.0;

  bool degen_prev_dirs_initialized_ = false;
  std::array<Eigen::Vector3d, 3> degen_prev_trans_dirs_map_;


struct PubJob {
  Eigen::Matrix4f T_cloud;
  Eigen::Matrix4f T_all;
  Eigen::Quaternionf state_q_scan;
  double scanStamp;
  pcl::PointCloud<PointType>::ConstPtr cloud;
  builtin_interfaces::msg::Time scan_header_stamp;

  // Odom-state snapshot at the same scan reference time as T_all/T_cloud
  Eigen::Vector3f state_p_scan;
  Eigen::Vector3f state_vlin_b_scan;
  Eigen::Vector3f state_vang_b_scan;
};

  struct PointCloudJob {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
    bool ring_range_filtered = false;
  };

  std::thread pointcloud_worker_;
  std::mutex pc_q_mtx_;
  std::condition_variable pc_q_cv_;
  std::deque<PointCloudJob> pc_q_;
  // Metrics are used independently of the timing feature flag.
  std::mutex g_metrics_mutex;

  std::thread pub_worker_;
  std::mutex q_mtx_;
  std::condition_variable q_cv_;
  std::deque<PubJob> q_;
  std::atomic_bool stop_{false};

  std::mutex reset_mutex_;
  std::condition_variable reset_done_cv_;
  bool reset_requested_ = false;
  std::atomic<bool> reset_in_progress_{false};
  bool reset_succeeded_ = false;
  std::string reset_status_message_;

  void pointCloudWorkerLoop();
  void enqueuePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);

  void workerLoop();
  void enqueuePublish(pcl::PointCloud<PointType>::ConstPtr cloud,
                      const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                      const Eigen::Ref<const Eigen::Matrix4f>& T_all,
                      double scanStamp,
                      const Eigen::Vector3f& state_p_scan,
                      const Eigen::Quaternionf& state_q_scan,
                      const Eigen::Vector3f& state_vlin_b_scan,
                      const Eigen::Vector3f& state_vang_b_scan);


};
