import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DockBehavior(Node):
    def __init__(self, video_path=None, aruco_dict=cv2.aruco.DICT_4X4_50):
        super().__init__('dock_behavior')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        if video_path == None:
            self.get_logger().info(f"Creating subscription for '/oak/rgb/image_raw'")
            self.subscription = self.create_subscription(
                Image,
                '/oak/rgb/image_raw', #TODO: Change to init param
                self.image_callback,
                10)
            self.bridge = CvBridge()

        else:
            self.cap = cv2.VideoCapture(video_path)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not self.cap.isOpened():
                raise Exception("Could not open video file")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)


        # TODO: Put this in a config file (json or yaml)
        self.second_marker = [(161, 449), (310, 452), (311, 602), (160, 601)]
        self.first_marker = [(905, 433), (1065, 432), (1069, 591), (907, 589)]
        
        #in the format: marker_id: 1 {'top_left': (x,y), 'top_right': (x,y), ...}
        self.marker_gt_dict = {
            1: {'top_left': (self.first_marker[0]),
                'top_right': (self.first_marker[1]),
                'bottom_right': (self.first_marker[2]),
                'bottom_left': (self.first_marker[3])},
            2: {'top_left': (self.second_marker[0]),
                'top_right': (self.second_marker[1]),
                'bottom_right': (self.second_marker[2]),
                'bottom_left': (self.second_marker[3])}
        }

        # Calculate area of the markers
        self.first_marker_area = self.calculate_marker_area(self.marker_gt_dict[1])
        self.second_marker_area = self.calculate_marker_area(self.marker_gt_dict[2])
        self.get_logger().info(f"Marker 1 area: {self.first_marker_area}")
        self.get_logger().info(f"Marker 2 area: {self.second_marker_area}")


    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def draw_ground_truth(self, frame):
            # Draw the ground truth markers
            for marker in [self.first_marker, self.second_marker]:
                for i in range(len(marker)):
                    cv2.line(frame, marker[i], marker[(i + 1) % len(marker)], (0, 255, 0), 2)
                    cv2.circle(frame, marker[i], 5, (0, 255, 0), -1)
            return frame
    
    # def draw_detected_markers(self, frame, corners, ids):

    #     cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    #     return 
    def extract_corners(self, ids, corners):
        
        corners_dict = {} #TODO: rename this variable if necessary
        for i, corner in enumerate(corners):

            marker_id = int(ids[i][0])
            top_left = tuple(map(int, corner[0][0]))
            top_right = tuple(map(int, corner[0][1]))
            bottom_right = tuple(map(int, corner[0][2]))
            bottom_left = tuple(map(int, corner[0][3]))

            corners_dict[marker_id] = {'top_left': top_left, 
                                        'top_right': top_right, 
                                        'bottom_right': bottom_right, 
                                        'bottom_left': bottom_left}

        #print(f'Dict: {corners_dict}')
        return corners_dict

    def compare_markers_visual(self, detected_markers, frame):
        """
        Compare detected markers with ground truth and visualize differences
        
        Args:
            detected_markers: Dict with detected points in the format:
                {
                    1: {'top_left': (x,y), 'top_right': (x,y), ...},
                    2: {'top_left': (x,y), 'top_right': (x,y), ...}
                }
            frame: Input image frame to draw on
        
        Returns:
            debug_frame: Image with drawn differences
            differences: dict w/ dx and dy between detected and ground truth points
            norms: dict w/ norm of vector between detected and ground truth points
        """

        # Cores para visualização
        COLOR_GT = (0, 255, 0)    # Verde para ground truth
        COLOR_DET = (0, 0, 255)   # Vermelho para detectado
        COLOR_LINE = (255, 255, 0) # Amarelo para linhas
        COLOR_TEXT = (255, 255, 255) # Branco para texto
        
        # Mapeia corners para a ordem dos pontos nos arrays ground truth
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        
        differences = {}
        norms = {}
        
        # Para cada marcador (1 e 2)
        for marker_id in [1, 2]: #TODO: use the ids from the detected markers
            if marker_id not in detected_markers:
                print(f"Marker {marker_id} not detected")
                continue
                
            # Get ground truth points
            gt_points = self.first_marker if marker_id == 1 else self.second_marker
            
            # Prepare differences dict
            differences[marker_id] = {}
            norms[marker_id] = {}
            
            # Para cada ponto do marcador
            for i, corner in enumerate(corner_order):
                # Get detected point
                det_point = detected_markers[marker_id][corner]
                gt_point = gt_points[i]
                
                # Calculate differences
                dx = gt_point[0] - det_point[0]
                dy = gt_point[1] - det_point[1]
                differences[marker_id][corner] = (dx, dy)

                # Compute norm
                norm = (dx**2 + dy**2) ** 0.5
                norms[marker_id][corner] = norm
                # Draw visualization
                # 1. Line connecting GT and detected
                cv2.line(frame, gt_point, det_point, COLOR_LINE, 2)
                
                # 2. Draw GT point (green)
                cv2.circle(frame, gt_point, 6, COLOR_GT, -1)
                
                # 3. Draw detected point (red)
                cv2.circle(frame, det_point, 4, COLOR_DET, -1)
                
                # 4. Draw text with differences
                text = f"D({dx}, {dy})"
                text_pos = ( (gt_point[0] + det_point[0])//2, 
                            (gt_point[1] + det_point[1])//2 )
                cv2.putText(frame, text, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        return frame, differences, norms
    

    def calculate_marker_area(self, corners_dict):
        """
        Calculate the area of a marker given its corners.
        
        Args:
            corners_dict: Dictionary with marker corners in the format:
                {'top_left': (x,y), 'top_right': (x,y), 
                 'bottom_right': (x,y), 'bottom_left': (x,y)}
        
        Returns:
            float: Area of the marker
        """
        top_left = np.array(corners_dict['top_left'])
        top_right = np.array(corners_dict['top_right'])
        bottom_right = np.array(corners_dict['bottom_right'])
        bottom_left = np.array(corners_dict['bottom_left'])

        # Calculate the area using the shoelace formula
        area = 0.5 * abs(top_left[0] * top_right[1] + top_right[0] * bottom_right[1] +
                         bottom_right[0] * bottom_left[1] + bottom_left[0] * top_left[1] -
                         (top_left[1] * top_right[0] + top_right[1] * bottom_right[0] +
                          bottom_right[1] * bottom_left[0] + bottom_left[1] * top_left[0]))
        return area


    def image_callback(self, msg):        
        self.curr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info("New image")
        self.detect_and_control()
        return 
    
    def movement_controller(self, linear_error, angular_error, kp_linear=0.00001, kp_angular=0.001):
        """
        Simple P controller for linear and angular velocities."
        """
        twist = Twist()
        twist.linear.x = kp_linear * linear_error
        twist.angular.z = kp_angular * angular_error
        
        self.publisher.publish(twist)
        return
    
    def check_if_adjusted(self, norms, threshold):
        """ Goes through the norms and checks if all are below threshold"""
        for marker_id in norms:
            for corner in norms[marker_id]:
                if norms[marker_id][corner] > threshold:
                    return False
        return True
    
    def calculate_error(self, dt_first_mk_area, dt_second_mk_area, diff_dict):
        """"
        Calculate the error between the detected markers and the ground truth using"
        the area discrepancy and the x delta"
        Args:
            dt_first_mk_area: Area of the first detected marker
            dt_second_mk_area: Area of the second detected marker
            diff_dict: Dictionary with differences between detected and ground truth points
        
        Returns:
            distance_error: Linear error based on area discrepancy
            hx_error: Angular error based on x delta (horizontal component)
        """
        # area_error1 = self.first_marker_area - dt_first_mk_area
        # area_error2 = self.second_marker_area - dt_second_mk_area 

        # #print(f'Area error 1: {area_error1}, Area error 2: {area_error2}\n')
        area_total = self.first_marker_area + self.second_marker_area
        distance_error = area_total - (dt_first_mk_area + dt_second_mk_area )
         
        # distance_error = (area_error1 + area_error2) / 2
        #print(f'Distance error: {distance_error}')
        # hx_error is the mean of all the x deltas
        dx_list = []
        for marker_id in diff_dict:
            for corner in diff_dict[marker_id]:
                dx = diff_dict[marker_id][corner][0]
                #print(f'Detected marker {marker_id} corner {corner} dx: {dx}, dy: {dy}')
                dx_list.append(dx)
        
        hx_error = np.mean(dx_list)
        # se ta negativo, o robô deve girar para a direita
        print(f'HX error: {hx_error}')
        print(f'Distance error: {distance_error}')  

        return distance_error, hx_error

    # Detect aruco info and feed the controller
    def detect_and_control(self):

        # Get image or video
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if not ret:
                return
        else:
            frame = self.curr_frame
        corners, ids = self.detect_markers(frame)

        # Make sure id 1 and id 2 are detected
        if ids is not None and 1 in ids and 2 in ids:
            
            # Extract corners
            corners_dict = self.extract_corners(ids, corners)
            # Calculate detected marker areas
            dt_first_mk_area = self.calculate_marker_area(corners_dict[1])
            dt_second_mk_area = self.calculate_marker_area(corners_dict[2])
            self.get_logger().info(f'Ground Truth Marker 1 area: {self.first_marker_area}')
            self.get_logger().info(f"Detected Marker 1 area: {dt_first_mk_area}")
            self.get_logger().info(f"Detected Marker 2 area: {dt_second_mk_area}")
            self.get_logger().info(f"Ground Truth Marker 2 area: {self.second_marker_area}")

            # Draw functions
            frame = self.draw_ground_truth(frame)
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            frame, differences, norms = self.compare_markers_visual(corners_dict, frame)
            
            adjusted = self.check_if_adjusted(norms, threshold=100)
            if adjusted:
                    cv2.putText(frame, "ADJUSTED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Calculate errors
                distance_error, hx_error = self.calculate_error(dt_first_mk_area, dt_second_mk_area, differences)
                # Feed controller
                self.movement_controller(distance_error, hx_error)

        cv2.imshow('Docking Calibration', frame)
        cv2.waitKey(1)
        return

    
def main(args=None):
    rclpy.init(args=args)
    dock_behavior = DockBehavior()
    rclpy.spin(dock_behavior)
    dock_behavior.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
