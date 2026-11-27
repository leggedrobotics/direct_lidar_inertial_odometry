from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare("direct_lidar_inertial_odometry")

    # Args
    robot_namespace = LaunchConfiguration('robot_namespace')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/livox/lidar')
    imu_topic = LaunchConfiguration('imu_topic', default='/livox/imu')
    odom_frame = LaunchConfiguration('odom_frame', default='robot/odom')
    baselink_frame = LaunchConfiguration('baselink_frame', default='robot/base_link')
    lidar_frame = LaunchConfiguration('lidar_frame', default='robot/lidar')
    imu_frame = LaunchConfiguration('imu_frame', default='robot/imu')

    declare_robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='robot',
        description='Namespace for the DLIO nodes and their topics'
    )
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/livox/lidar',
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/livox/imu',
        description='IMU topic name'
    )
    declare_odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='robot/odom',
        description='Frame id used for odometry outputs'
    )
    declare_baselink_frame_arg = DeclareLaunchArgument(
        'baselink_frame',
        default_value='robot/base_link',
        description='Base link frame id used by DLIO'
    )
    declare_lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='robot/lidar',
        description='Internal LiDAR frame id published by DLIO'
    )
    declare_imu_frame_arg = DeclareLaunchArgument(
        'imu_frame',
        default_value='robot/imu',
        description='Internal IMU frame id published by DLIO'
    )
    # Two INDEPENDENT things happen in DLIO's start-up window, and they have very
    # different failure modes -- do not tie them to one switch.
    #
    # imu_bias_calibration: averages the first imu/calibration/time seconds of
    #   gyro and accel and freezes that as the bias. Motion in the window is
    #   absorbed as bias; a settling robot produced a 2.05 rad/s "gyro bias"
    #   (117 deg/s, then clamped to 0.5). A simulated IMU has no bias to find, so
    #   this is pure harm there.
    #
    # gravity_align: sets the initial attitude so measured gravity maps to
    #   (0,0,g). This is what absorbs the sensor's mounting tilt -- mid360_joint
    #   carries rpy="0 0.0401 0" (2.30 deg) while DLIO's baselink2imu extrinsic is
    #   a pure 180 deg roll with no pitch. Turn it off and that 2.30 deg is never
    #   accounted for, leaving 9.81*sin(2.30) = 0.39 m/s^2 of horizontal
    #   acceleration in the gravity compensation, which biases the velocity the
    #   policy consumes as base_lin_vel. Keep it on; it works from ANY stationary
    #   orientation, so it only needs the robot to be still, not level.
    imu_bias_calibration = LaunchConfiguration('imu_bias_calibration', default='true')
    gravity_align = LaunchConfiguration('gravity_align', default='true')
    # Simulation with /clock (e.g. a slowed-down MuJoCo): the nodes take ROS time from /clock.
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    declare_imu_bias_calibration_arg = DeclareLaunchArgument(
        'imu_bias_calibration',
        default_value='true',
        description='Estimate gyro/accel biases from the first seconds of data. '
                    'Requires a stationary robot; harmful on a bias-free '
                    'simulated IMU.'
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Take ROS time from /clock (simulation).'
    )
    declare_gravity_align_arg = DeclareLaunchArgument(
        'gravity_align',
        default_value='true',
        description='Set the initial attitude from measured gravity. Absorbs the '
                    'sensor mounting tilt; needs the robot still, not level.'
    )

    # Load DLIO parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "params.yaml"])

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_odom_node",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                "frames/odom": odom_frame,
                "frames/baselink": baselink_frame,
                "frames/lidar": lidar_frame,
                "frames/imu": imu_frame,
                "frames/publish_sensor_tf": True,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                # last entry wins over the yaml above
                "odom/imu/calibration/gyro": ParameterValue(
                    imu_bias_calibration, value_type=bool),
                "odom/imu/calibration/accel": ParameterValue(
                    imu_bias_calibration, value_type=bool),
                "odom/imu/approximateGravity": ParameterValue(
                    gravity_align, value_type=bool),
            },
        ],
        remappings=[
            ("pointcloud", pointcloud_topic),
            ("imu", imu_topic),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ("odom", "dlio/odom_node/odom"),
            ("pose", "dlio/odom_node/pose"),
            ("path", "dlio/odom_node/path"),
            ("kf_pose", "dlio/odom_node/keyframes"),
            ("kf_cloud", "dlio/odom_node/pointcloud/keyframe"),
            ("deskewed", "dlio/odom_node/pointcloud/deskewed"),
            ('deskewed_not_transformed', 'dlio/odom_node/pointcloud/deskewed_not_transformed'),
        ],
        respawn=True,
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_map_node",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                "odom/odom_frame": odom_frame,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
        ],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    return LaunchDescription([
        # 1) Arguments
        declare_robot_namespace_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_odom_frame_arg,
        declare_baselink_frame_arg,
        declare_lidar_frame_arg,
        declare_imu_frame_arg,
        declare_imu_bias_calibration_arg,
        declare_gravity_align_arg,
        declare_use_sim_time_arg,
        # 2) Nodes
        dlio_odom_node,
        dlio_map_node,
    ])
