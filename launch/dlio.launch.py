from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Args
    rviz = LaunchConfiguration('rviz', default='false')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/lidar/point_cloud')
    imu_topic = LaunchConfiguration('imu_topic', default='/imu_sensor_broadcaster/imu')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    declare_rviz_arg = DeclareLaunchArgument('rviz', default_value=rviz, description='Launch RViz')
    declare_pointcloud_topic_arg = DeclareLaunchArgument('pointcloud_topic', default_value=pointcloud_topic, description='Pointcloud topic name')
    declare_imu_topic_arg = DeclareLaunchArgument('imu_topic', default_value=imu_topic, description='IMU topic name')
    declare_use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value=use_sim_time, description='Use /clock (sim time)')

    # Params
    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])

    # Nodes
    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',The fix. A third AbsoluteUnaryAlignmentRecoveryPolicy — ReinitializeFromCurrentGuess — that throws the stale belief away and re-initializes from the fresh geometric guess instead:
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': use_sim_time}],
        remappings=[
            ('pointcloud', pointcloud_topic),
            ('imu', imu_topic),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ('map_pose_inverted', 'dlio/odom_node/map_pose_inverted'),
            ('odom', 'dlio/odom_node/odom'),
            ('pose', 'dlio/odom_node/pose'),
            ('path_map', 'dlio/odom_node/path_map'),
            ('path_odom', 'dlio/odom_node/path_odom'),
            ('path_map_prop', 'dlio/odom_node/path_map_prop'),
            ('kf_pose', 'dlio/odom_node/keyframes'),
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
            ('deskewed_not_transformed', 'dlio/odom_node/pointcloud/deskewed_not_transformed'),
            ('deskewed_and_transformed_to_map', 'dlio/odom_node/pointcloud/deskewed_and_transformed_to_map'),
            ('markers/velocity_linear', 'dlio/odom_node/markers/velocity_linear'),
            ('markers/velocity_angular', 'dlio/odom_node/markers/velocity_angular'),
            ('markers/correction', 'dlio/odom_node/markers/correction'),
            ('markers/degeneracy_directions', 'dlio/odom_node/markers/degeneracy_directions'),
            ('degenerate', 'dlio/odom_node/degenerate'),
        ],
        respawn=True,
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': use_sim_time}],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'dlio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_use_sim_time_arg,
        dlio_odom_node,
        dlio_map_node,
        rviz_node
    ])