#!/usr/bin/env python3
"""
Visual Odometry Node for RoboSub
=================================
This node connects to a ZED camera on the Jetson, fuses IMU data with visual
odometry, and publishes pose and velocity estimates for underwater navigation.

The ZED SDK handles the heavy lifting of:
- Stereo matching (computing depth from left/right camera images)
- Feature tracking across frames
- IMU integration for robust odometry
- Pose estimation in 3D space
"""

import rospy  # ROS Python library for creating nodes, publishers, subscribers
import sys    # For system-level operations and exit codes
import numpy as np  # Numerical operations for transformations

# Import ZED SDK - this is the official SDK from Stereolabs
# You'll need to install it on your Jetson: https://www.stereolabs.com/developers/release/
import pyzed.sl as sl

# Import ROS message types for publishing our data
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

# Import transformations library for converting between rotation representations
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class VisualOdometryNode:
    """
    This class encapsulates all the visual odometry functionality.

    It handles:
    1. ZED camera initialization and configuration
    2. Positional tracking (the ZED's term for visual-inertial odometry)
    3. Publishing pose and velocity data to ROS topics
    4. Graceful shutdown
    """

    def __init__(self):
        """
        Initialize the ROS node and set up the ZED camera.
        This constructor is called once when the node starts.
        """
        # Initialize the ROS node
        # 'visual_odometry' is the name that will appear in rosnode list
        # anonymous=False means we won't add random numbers to the name
        rospy.init_node('visual_odometry', anonymous=False)
        rospy.loginfo("Initializing Visual Odometry Node for RoboSub...")

        # Create ZED camera object
        # This is the main interface to the camera hardware
        self.zed = sl.Camera()

        # Get ROS parameters (these can be set in a launch file)
        # The get_param() function retrieves parameters from the ROS parameter server
        # Format: get_param('parameter_name', default_value)
        self.camera_resolution = rospy.get_param('~camera_resolution', 'HD720')  # Camera resolution
        self.camera_fps = rospy.get_param('~camera_fps', 30)  # Frames per second
        self.pub_rate = rospy.get_param('~publish_rate', 30)  # How often to publish data

        # ROS Publishers - these send data to other ROS nodes
        # The second parameter is the message type, third is queue size
        self.pose_pub = rospy.Publisher('/zed/pose', PoseStamped, queue_size=10)
        self.velocity_pub = rospy.Publisher('/zed/velocity', TwistStamped, queue_size=10)
        self.odom_pub = rospy.Publisher('/zed/odom', Odometry, queue_size=10)
        self.imu_pub = rospy.Publisher('/zed/imu', Imu, queue_size=10)

        # Initialize the camera
        if not self.initialize_camera():
            rospy.logerr("Failed to initialize ZED camera. Exiting...")
            sys.exit(1)

        # Set up positional tracking (visual-inertial odometry)
        if not self.setup_positional_tracking():
            rospy.logerr("Failed to enable positional tracking. Exiting...")
            self.zed.close()
            sys.exit(1)

        rospy.loginfo("Visual Odometry Node initialized successfully!")

    def initialize_camera(self):
        """
        Configure and open the ZED camera.

        Returns:
            bool: True if successful, False otherwise
        """
        rospy.loginfo("Initializing ZED camera...")

        # Create initialization parameters object
        # This holds all the settings for how the camera should operate
        init_params = sl.InitParameters()

        # Set camera resolution
        # HD720 (1280x720) is a good balance between quality and performance
        # For underwater, you might want to experiment with different resolutions
        resolution_dict = {
            'HD2K': sl.RESOLUTION.HD2K,    # 2208x1242 - highest quality, most compute
            'HD1080': sl.RESOLUTION.HD1080, # 1920x1080 - high quality
            'HD720': sl.RESOLUTION.HD720,   # 1280x720 - balanced (RECOMMENDED for RoboSub)
            'VGA': sl.RESOLUTION.VGA        # 672x376 - lowest quality, fastest
        }
        init_params.camera_resolution = resolution_dict.get(self.camera_resolution, sl.RESOLUTION.HD720)

        # Set frame rate - how many images per second the camera captures
        init_params.camera_fps = self.camera_fps

        # Coordinate system: Right-Handed, Y-Up
        # This means: X = right, Y = up, Z = forward (robot-centric)
        # This is important for underwater navigation where "up" matters
        init_params.coordinate_units = sl.UNIT.METER  # Use meters for measurements
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Depth mode: ULTRA for best quality
        # The camera computes depth by comparing left and right images
        # ULTRA mode uses more processing but gives better depth accuracy (important underwater!)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        # Depth stabilization: helps reduce noise in depth measurements
        # Underwater environments can have particles and lighting issues
        init_params.depth_stabilization = True

        # Try to open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            rospy.logerr(f"Failed to open ZED camera: {err}")
            return False

        rospy.loginfo(f"ZED camera opened successfully at {self.camera_resolution} @ {self.camera_fps} FPS")
        return True

    def setup_positional_tracking(self):
        """
        Enable positional tracking (visual-inertial odometry).

        This is where the magic happens! The ZED SDK will:
        1. Track visual features between frames
        2. Fuse IMU data for better motion estimation
        3. Estimate the camera's 6-DOF pose (position + orientation)

        Returns:
            bool: True if successful, False otherwise
        """
        rospy.loginfo("Enabling positional tracking...")

        # Create tracking parameters object
        tracking_params = sl.PositionalTrackingParameters()

        # Set the initial position (where the robot starts)
        # We assume it starts at the origin (0, 0, 0) facing forward
        initial_position = sl.Transform()
        initial_position.set_identity()  # Identity = no translation or rotation
        tracking_params.set_initial_world_transform(initial_position)

        # Enable area memory: the camera remembers places it has seen
        # This helps with loop closure (recognizing you've returned to a previous location)
        # For a competition pool, this could help maintain accuracy over multiple runs
        tracking_params.enable_area_memory = True

        # Enable IMU fusion: combines visual odometry with IMU data
        # This is CRITICAL for underwater where visibility can be poor
        # The IMU helps maintain tracking even when visual features are scarce
        tracking_params.enable_imu_fusion = True

        # Set floor as reference plane (or in your case, pool bottom)
        # This can help stabilize the vertical position estimate
        tracking_params.set_floor_as_origin = False  # We'll handle our own reference frame

        # Enable pose smoothing: reduces jitter in pose estimates
        # Good for control systems that don't like noisy inputs
        tracking_params.enable_pose_smoothing = True

        # Try to enable tracking
        err = self.zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            rospy.logerr(f"Failed to enable positional tracking: {err}")
            return False

        rospy.loginfo("Positional tracking enabled successfully!")
        return True

    def run(self):
        """
        Main processing loop.

        This runs continuously, grabbing frames from the camera,
        getting pose/velocity estimates, and publishing them to ROS topics.
        """
        rospy.loginfo("Starting visual odometry processing loop...")

        # Create a Rate object to control loop frequency
        # This ensures we publish at a consistent rate
        rate = rospy.Rate(self.pub_rate)

        # Create objects to store data from the ZED
        # These are reused each iteration for efficiency
        runtime_params = sl.RuntimeParameters()  # Parameters for grab()
        camera_pose = sl.Pose()  # Will store the camera's pose
        camera_imu_data = sl.SensorsData()  # Will store IMU data

        # Main loop - runs until ROS shuts down (Ctrl+C)
        while not rospy.is_shutdown():
            # Grab a new frame from the camera
            # This captures images from both left and right cameras,
            # computes depth, and updates the pose estimate
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                # Get the current pose estimate
                # REFERENCE_FRAME.WORLD means relative to where we started
                tracking_state = self.zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)

                # Check if tracking is working
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    # Publish the pose and velocity data
                    self.publish_pose(camera_pose)
                    self.publish_velocity(camera_pose)
                    self.publish_odometry(camera_pose)
                else:
                    # Tracking lost - this can happen if the camera can't see enough features
                    rospy.logwarn(f"Tracking state: {tracking_state}. Visual odometry may be unreliable.")

                # Get and publish IMU data
                if self.zed.get_sensors_data(camera_imu_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    self.publish_imu(camera_imu_data)

            # Sleep to maintain the desired loop rate
            rate.sleep()

    def publish_pose(self, camera_pose):
        """
        Publish the camera's pose (position + orientation) as a PoseStamped message.

        Args:
            camera_pose: sl.Pose object from the ZED SDK
        """
        # Create a PoseStamped message
        # "Stamped" means it includes a timestamp and coordinate frame
        pose_msg = PoseStamped()

        # Set the header (metadata about the message)
        pose_msg.header.stamp = rospy.Time.now()  # Current time
        pose_msg.header.frame_id = "world"  # Coordinate frame name

        # Extract position (translation) from the pose
        # The pose contains a 4x4 transformation matrix
        translation = camera_pose.get_translation().get()  # Returns [x, y, z]
        pose_msg.pose.position.x = translation[0]  # Forward/backward (meters)
        pose_msg.pose.position.y = translation[1]  # Left/right (meters)
        pose_msg.pose.position.z = translation[2]  # Up/down (meters)

        # Extract orientation (rotation) from the pose
        # The orientation is stored as a quaternion (4 values: x, y, z, w)
        # Quaternions are a mathematical way to represent 3D rotations without gimbal lock
        orientation = camera_pose.get_orientation().get()  # Returns [ox, oy, oz, ow]
        pose_msg.pose.orientation.x = orientation[0]
        pose_msg.pose.orientation.y = orientation[1]
        pose_msg.pose.orientation.z = orientation[2]
        pose_msg.pose.orientation.w = orientation[3]

        # Publish the message
        self.pose_pub.publish(pose_msg)

    def publish_velocity(self, camera_pose):
        """
        Publish the camera's velocity (linear + angular) as a TwistStamped message.

        Args:
            camera_pose: sl.Pose object from the ZED SDK
        """
        # Create a TwistStamped message
        # "Twist" in ROS means velocities (linear and angular)
        vel_msg = TwistStamped()

        # Set the header
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = "world"

        # Extract linear velocity (how fast we're moving in x, y, z)
        # Units are meters per second
        linear_vel = camera_pose.get_velocity().get()  # Returns [vx, vy, vz]
        vel_msg.twist.linear.x = linear_vel[0]  # Forward/backward velocity
        vel_msg.twist.linear.y = linear_vel[1]  # Left/right velocity
        vel_msg.twist.linear.z = linear_vel[2]  # Up/down velocity

        # Extract angular velocity (how fast we're rotating around x, y, z axes)
        # Units are radians per second
        angular_vel = camera_pose.get_angular_velocity().get()  # Returns [wx, wy, wz]
        vel_msg.twist.angular.x = angular_vel[0]  # Roll rate
        vel_msg.twist.angular.y = angular_vel[1]  # Pitch rate
        vel_msg.twist.angular.z = angular_vel[2]  # Yaw rate

        # Publish the message
        self.velocity_pub.publish(vel_msg)

    def publish_odometry(self, camera_pose):
        """
        Publish an Odometry message containing both pose and velocity.

        This is a standard ROS message type used by navigation stacks.
        It combines position, orientation, and velocities in one message.

        Args:
            camera_pose: sl.Pose object from the ZED SDK
        """
        # Create an Odometry message
        odom_msg = Odometry()

        # Set the header
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"  # Fixed world frame
        odom_msg.child_frame_id = "base_link"  # Robot's base frame

        # Set pose (same as publish_pose)
        translation = camera_pose.get_translation().get()
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]

        orientation = camera_pose.get_orientation().get()
        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]

        # Set velocity (same as publish_velocity)
        linear_vel = camera_pose.get_velocity().get()
        odom_msg.twist.twist.linear.x = linear_vel[0]
        odom_msg.twist.twist.linear.y = linear_vel[1]
        odom_msg.twist.twist.linear.z = linear_vel[2]

        angular_vel = camera_pose.get_angular_velocity().get()
        odom_msg.twist.twist.angular.x = angular_vel[0]
        odom_msg.twist.twist.angular.y = angular_vel[1]
        odom_msg.twist.twist.angular.z = angular_vel[2]

        # Set covariance matrices (uncertainty estimates)
        # These tell other systems how confident we are in our measurements
        # The ZED SDK provides these, but we'll use conservative estimates here
        # Format: [x, y, z, rotation_x, rotation_y, rotation_z] variances
        # Smaller values = more confident

        # Pose covariance (6x6 matrix flattened to 36 values)
        # For underwater, we might be less confident in Z (depth) than X/Y
        pose_covariance = [
            0.01, 0,    0,    0, 0, 0,  # x variance
            0,    0.01, 0,    0, 0, 0,  # y variance
            0,    0,    0.02, 0, 0, 0,  # z variance (less confident)
            0,    0,    0,    0.01, 0, 0,  # roll variance
            0,    0,    0,    0, 0.01, 0,  # pitch variance
            0,    0,    0,    0, 0, 0.01   # yaw variance
        ]
        odom_msg.pose.covariance = pose_covariance

        # Twist covariance (6x6 matrix flattened to 36 values)
        twist_covariance = [
            0.02, 0,    0,    0, 0, 0,
            0,    0.02, 0,    0, 0, 0,
            0,    0,    0.04, 0, 0, 0,
            0,    0,    0,    0.02, 0, 0,
            0,    0,    0,    0, 0.02, 0,
            0,    0,    0,    0, 0, 0.02
        ]
        odom_msg.twist.covariance = twist_covariance

        # Publish the message
        self.odom_pub.publish(odom_msg)

    def publish_imu(self, sensors_data):
        """
        Publish IMU data (acceleration, angular velocity, orientation).

        This publishes the raw IMU data from the ZED's built-in IMU.
        Useful for sensor fusion or debugging.

        Args:
            sensors_data: sl.SensorsData object from the ZED SDK
        """
        # Create an IMU message
        imu_msg = Imu()

        # Set the header
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "imu_link"

        # Get IMU data from the sensors
        imu_data = sensors_data.get_imu_data()

        # Set orientation (from IMU's internal fusion)
        orientation = imu_data.get_pose().get_orientation().get()
        imu_msg.orientation.x = orientation[0]
        imu_msg.orientation.y = orientation[1]
        imu_msg.orientation.z = orientation[2]
        imu_msg.orientation.w = orientation[3]

        # Set angular velocity (gyroscope data)
        # This is how fast the camera is rotating
        angular_vel = imu_data.get_angular_velocity()
        imu_msg.angular_velocity.x = angular_vel[0]
        imu_msg.angular_velocity.y = angular_vel[1]
        imu_msg.angular_velocity.z = angular_vel[2]

        # Set linear acceleration (accelerometer data)
        # This measures acceleration including gravity
        linear_accel = imu_data.get_linear_acceleration()
        imu_msg.linear_acceleration.x = linear_accel[0]
        imu_msg.linear_acceleration.y = linear_accel[1]
        imu_msg.linear_acceleration.z = linear_accel[2]

        # Set covariance matrices (sensor noise characteristics)
        # These values are typical for the ZED's IMU
        # You can tune these based on your specific sensor calibration
        orientation_covariance = [0.001] * 9  # 3x3 matrix
        angular_velocity_covariance = [0.002] * 9
        linear_acceleration_covariance = [0.004] * 9

        imu_msg.orientation_covariance = orientation_covariance
        imu_msg.angular_velocity_covariance = angular_velocity_covariance
        imu_msg.linear_acceleration_covariance = linear_acceleration_covariance

        # Publish the message
        self.imu_pub.publish(imu_msg)

    def shutdown(self):
        """
        Clean shutdown procedure.

        This is called when the node is terminated (e.g., Ctrl+C).
        It ensures the camera is properly closed to avoid resource leaks.
        """
        rospy.loginfo("Shutting down Visual Odometry Node...")

        # Disable positional tracking
        self.zed.disable_positional_tracking()

        # Close the camera
        self.zed.close()

        rospy.loginfo("Visual Odometry Node shut down successfully.")


def main():
    """
    Main entry point for the script.

    This function is called when you run: python visual_odometry.py
    """
    try:
        # Create the node
        node = VisualOdometryNode()

        # Register shutdown callback
        # This ensures cleanup happens when the node is terminated
        rospy.on_shutdown(node.shutdown)

        # Run the main processing loop
        node.run()

    except rospy.ROSInterruptException:
        # This exception is raised when the node is terminated
        rospy.loginfo("Visual Odometry Node interrupted.")
    except Exception as e:
        # Catch any other errors
        rospy.logerr(f"Unexpected error in Visual Odometry Node: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # This block only runs if the script is executed directly
    # (not if it's imported as a module)
    main()
