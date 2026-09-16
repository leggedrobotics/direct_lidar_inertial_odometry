from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
    self_filter = LaunchConfiguration('self_filter')
    body_filter_urdf = LaunchConfiguration('body_filter_urdf')

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
    # CZ and Claude: G1 self-filter (see cfg/body_filter.yaml)
    declare_self_filter_arg = DeclareLaunchArgument(
        'self_filter',
        default_value='true',
        description='Run the G1 body filter on the deskewed cloud -> .../pointcloud/deskewed_self_filtered'
    )
    declare_body_filter_urdf_arg = DeclareLaunchArgument(
        'body_filter_urdf',
        default_value=PathJoinSubstitution([FindPackageShare("g1_description"), "g1_29dof_body_filter.urdf"]),
        description='Dedicated URDF whose collisions are AABB boxes / spheres (fast contains-tests)'
    )
    # end of CZ and Claude

    # Load DLIO parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "params.yaml"])
    body_filter_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "body_filter.yaml"])

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
            },
        ],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    # CZ and Claude: G1 self-filter: removes robot-body points (+dilation) from the deskewed cloud.
    # Requires base_odom.launch.py running (full-body TF). Uses a dedicated URDF whose collisions are
    # AABB boxes (fast contains-tests); queue size 1 so a lagging filter never builds a backlog.
    body_filter_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="g1_body_filter_node",
        name="body_filter",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            body_filter_yaml_path,
            {"urdf_path": body_filter_urdf},
        ],
        remappings=[
            ("input", "dlio/odom_node/pointcloud/deskewed"),
            ("output", "dlio/odom_node/pointcloud/deskewed_self_filtered"),
        ],
        condition=IfCondition(self_filter),
        respawn=True,
    )
    # end of CZ and Claude

    return LaunchDescription([
        # 1) Arguments
        declare_robot_namespace_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_odom_frame_arg,
        declare_baselink_frame_arg,
        declare_lidar_frame_arg,
        declare_imu_frame_arg,
        declare_self_filter_arg,
        declare_body_filter_urdf_arg,
        # 2) Nodes
        dlio_odom_node,
        dlio_map_node,
        body_filter_node,
    ])
