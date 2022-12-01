"""grocery controller"""

# Nov 2, 2022

from controller import Robot
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

# Helper Functions #
######### A(star) algorithm def and class 
from typing import Tuple, List, TypedDict, Union
from queue import PriorityQueue
import math

class Node:
    def __init__(self, row, col, heading, dist_gscore, prev, fscore):
        self.row = row 
        self.col = col
        self.heading = heading
        self.dist_gscore = dist_gscore #from start
        self.prev: Node = prev
        self.fscore = fscore #huristic
        
    #override for < comp    
    def __lt__(self, other):
        return self.fscore < other.fscore
    
    def __str__(self):
        return f"Row: {self.row} Col: {self.col} Head: {self.heading} dist_gscore: {self.dist_gscore} fscore: {self.fscore}"

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col)

def hscore(p1, goal):
    return np.linalg.norm([p1.row - goal[0], p1.col - goal[1]])

def get_neighbor_nodes(map, target:Node)-> List[Tuple[int,int,float]]:
    heading_key = {
        (-1,-1):    math.pi/4,
        (-1,0):     math.pi/2,
        (-1,1):     (3*math.pi)/4,
        (0,-1):     0,
        (0,0):      math.inf,
        (0,1):      math.pi,
        (1,-1):     -math.pi/4,
        (1,0):      -math.pi/2,
        (1,1):      -(3*math.pi)/4,
    }

    neighbors: List[Tuple[int,int,float]] = []
    for row_off in [-1,0,1]:
        for col_off in [-1,0,1]:
            if not (col_off==0 and row_off==0):
                try:
                    if map[target.row+row_off][target.col+col_off] == 0:
                        neighbors.append((target.row+row_off, target.col+col_off, heading_key[(row_off,col_off)]))
                except:
                    pass
    return neighbors

def reverse_traverse_nodes(final_node)->List[Node]:
    path: List[Node] = []
    crawl = final_node
    while crawl is not None:
        path.append(crawl)
        crawl = crawl.prev
    path.reverse()
    return path

def nodes_to_path(nodes: List[Node])->List[Tuple[int,int,float]]:
    return [(node.row,node.col,node.heading) for node in nodes]

def a_star_path(map: np.matrix, start: Tuple[int,int], goal: Tuple[int,int])->List[Tuple[int,int,float]]:
    discovery_queue: PriorityQueue[Node] = PriorityQueue()
    discovery_queue.put(Node(start[0], start[1], 0., 0, None, math.inf))
    
    # 2d array of empty space or node's we've visited
    node_map: List[List[Union[Node,None]]] = [[Node(row,col,0,math.inf,None,math.inf) for col in range(len(map))] for row in range(len(map[0]))]
    
    while not discovery_queue.empty():
        curr_node = discovery_queue.get()

        if curr_node == node_map[goal[0]][goal[1]]:
            break
        for row,col,heading in get_neighbor_nodes(map, curr_node):
            next_gscore = curr_node.dist_gscore + np.linalg.norm([curr_node.row - row, curr_node.col - col])
            if next_gscore < node_map[row][col].dist_gscore:
                #if this node is new or if it's being visited from a shorter path
                node_map[row][col].dist_gscore = next_gscore
                node_map[row][col].prev = curr_node
                curr_node.heading = heading # set the heading at the previos waypoint to point towards the next waypoint
                node_map[row][col].fscore = next_gscore + hscore(node_map[row][col], goal)
                discovery_queue.put(node_map[row][col])
                
            # reverse traverse and convert to waypoints from the goal
    return nodes_to_path(reverse_traverse_nodes(node_map[goal[0]][goal[1]]))


#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)
pi = 3.14159
headingError = 0
bearingError = 0
distError = 10000
# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

#

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

#Enable keyboard for remore control
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Initialize Odometry
pose_y = 0
pose_x = -5
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
#lidat_angle_range was negative
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

#states
#mode =  'manual' # mapping
control = 'searching'
#mode = 'get_item'
mode = 'follow_path'

#initialize map
map = np.empty((360,360))
waypoints = []


gripper_status="closed"

robot_stops = []
#converts robot coords into world coords for each stopping point
object_world_coords = [(-5,0),(2.62,2.689),(3.504,2.689),(3.499,6.269),(5.802,6.26),(.777,1.27),(-2.665,1.1),(-2.665,1.1),(-.8647,1.2606),(2.341,-2.633), (-1.329,-4.953)]
i = 0
for x in object_world_coords: #get robot to world coords for astar
    robot_stops.append((int(180+ x[0]*12), int((180 + x[1]*12))))
    i = i+1

object_theta = [0,pi/2,pi/2,pi/2,pi/2,-pi/2,-pi/2,-pi/2,-pi/2,-pi/2,pi/2] 
counter = 0  
waypoints = []
print(robot_stops)

    


#get first start and end values in world coords
if(mode != 'manual'):#get first path before while loop

    start = robot_stops[counter]
    end = robot_stops[counter+1]

    map = np.load("map.npy")
    map = map > 0.5
    map = 1* map  
    
    #Get convolved map
    kernel_size = 10
    Kernel = np.ones((kernel_size, kernel_size)) # Play with this number to find something suitable, the number corresponds to the # of pixels you want to cover
    Convolved_map = convolve2d(map, Kernel, mode='same') # You still have to threshold this convolved map
    Convolved_map = Convolved_map > 0.5
    Convolved_map = 1* Convolved_map 

    #call astar
    path = a_star_path(Convolved_map, start, end)
    #print(map)
    #make waypoints in path robot coords
    for p in path:
       world_x = -(180- p[0])/12 
       world_y = -(180 - p[1])/12 
       waypoints.append((world_x,world_y))
    end_point = waypoints[len(waypoints)-1]
    waypoints.append(robot_stops[counter+1])
    waypoints = waypoints[::20]
    #waypoints.append(end_point)
    waypoints.append(object_world_coords[counter+1])
    print(waypoints)

    
        


while robot.step(timestep) != -1:
    pose_y = gps.getValues()[1] #get initial pose for the robot odo at bottom
    pose_x = gps.getValues()[0]
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
    n = compass.getValues() #
    rad = ((math.atan2(n[0], n[1])))#-1.5708)
    pose_theta = rad
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

       # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = -math.cos(alpha)*rho + 0.202
        ry = math.sin(alpha)*rho -0.004

       
       # Convert detection from robot coordinates into world coordinates
        if((pose_theta > pi/2 and pose_theta < pi) or (pose_theta < -(pi/2) and pose_theta > -pi) ):
            #backwards
            wy =  -(math.cos(pose_theta)*ry + math.sin(pose_theta)*rx) + pose_y
        else:
        #foewards
            wy =  (math.cos(pose_theta)*ry + math.sin(pose_theta)*rx) + pose_y
              
        wx =  (math.cos(pose_theta)*rx - math.sin(pose_theta)*ry) + pose_x
 
           ####mapping####
           
        if rho < 2:

            try:
                #increments pixel value in world coordinares
                map[180+int(wx*12)][180+int(wy*12)] = map[180+int(wx*12)][180+int(wy*12)] + .005

                #update color for that pizel and set to white if value is >= 1
                color = (map[180+int(wx*12)][180+int(wy*12)]*256**2+map[180+int(wx*12)][180+int(wy*12)]*256+map[180+int(wx*12)][180+int(wy*12)]*255)
                if map[180+int(wx*12)][180+int(wy*12)] + .005 >= 1: 
                    map[180+int(wx*12)][180+int(wy*12)] = 1
                    display.setColor(0xFFFFFF)
                    display.drawPixel(180+int(wx*12),180+int(wy*12))
                else: 

                    color = (map[180+int(wx*12)][180+int(wy*12)]*256**2+map[180+int(wx*12)][180+int(wy*12)]*256+map[180+int(wx*12)][180+int(wy*12)]*255)
                    display.setColor(int(color))
                    display.drawPixel(180+int(wx*12),180+int(wy*12))
            except: pass

   # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(180+int(pose_x*12),180+int(pose_y*12))

   #Main Controller
   ###################
    if mode == 'manual': #for mapping
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED/2
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED/2
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
           #saves map to filesystem
            map = map > 0.5
            map = 1* map           
            np.save("map.npy",map)
            print("Map file saved")
        elif key == ord('L'):
           # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
            plt.imshow(map)
            plt.show()

        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode after we get mapping done
        
        if headingError < -3.14: headingError += 6.28 #keep angles within bounds

        if bearingError < -3.14: bearingError += 6.28

        if(distError < .1 and abs(bearingError) < 3.1415/4): #change waypoint on arrival
            #get new object and make new path
            counter = counter + 1
            waypoints = []
            start = robot_stops[counter] ##gets new waypoints between next object and curr object
            end = robot_stops[counter+1]
            path = astar(convolved_map,start,end)
            for p in path:
                world_x = -(180- p[0])/12 
                world_y = -(180 - p[1])/12 
                waypoints.append((world_x,world_y))
            last = len(waypoints-1)
            end_point = waypoints[len(waypoints)-1]
            waypoints = waypoints[::20]
            waypoints.append(object_world_coords[counter])
            control = 'turning' ##sends control to turning mode
            vL = 0
            vR = 0
        if(control == 'turning'): #orientates robot so it faces object
            vl = MAX_SPEED/2
            vR = -MAX_SPEED/2
            # if(pose_theta > 0 and pose_theta < pi/2):
            #     vR = MAX_SPEED/2
            #     vL = -MAX_SPEED/2
            # if(pose_theta < 0 and pose_theta > -pi/2):
            #     vR = -MAX_SPEED/2
            #     vL = MAX_SPEED/2
            # if(pose_theta > pi/2 and pose_theta < pi):
            #     vR = MAX_SPEED/2
            #     vL = -MAX_SPEED/2
            # if(pose_theta < -pi/2 and pose_theta > -pi):
            #     vR = MAX_SPEED/2
            #     vL = -MAX_SPEED/2
            if(abs(pose_theta - object_theta[counter] ) < .05): ##robot is facing object
                vL = 0
                vR = 0
                control = 'grab_object'
            
        if(control == 'grab_object' ): ##manipulate arm to get object

            if(object_grabbed == True):
                control = 'searching'
                continue#
        #if()
        if(control == 'searching'): ##so we can update the arms
            distError= math.sqrt((waypoints[counter][0]-pose_x)**2+(waypoints[counter][1]-pose_y)**2)
            bearingError = math.atan2(waypoints[counter][1]-pose_y,waypoints[counter][0]-pose_x)-pose_theta 
            headingError = math.atan2(waypoints[counter+1][1] - waypoints[counter][1], waypoints[counter+1][0] - waypoints[counter][0]) - pose_theta
            dx = 0
            dTheta = 0
            dX = distError
            dTheta = bearingError

            if(distError <= .1):
                dX = 0
                dTheta = 100 * headingError
            elif(distError < .2):
                dTheta = (bearingError * (distError*5)) + (headingError * (1- 5*distError))
            elif distError > .75 and abs(bearingError) > 3.1415/6 :
                dX = 0
            
            dTheta *= 10 
            
            #Proportional velocities
            vL = (dX - (dTheta * AXLE_LENGTH/2.0)) 
            vR = (dX + (dTheta * AXLE_LENGTH/2.0))

            #normalization
            vLhold = vL
            MAX = max(abs(vL),abs(vR))
            vL = vL + MAX
            vR = vR + MAX
            pL = ((vL / max(vL,vR)) - 0.5) * 2.0
            pR = ((vR / max(vL,vR)) - 0.5) * 2.0
            
            
            
            vL = pL * MAX_SPEED
            vR = pR * MAX_SPEED
            

            # Clamp wheel speeds
            if(vL > MAX_SPEED):
                vL = MAX_SPEED
            if(vL < -MAX_SPEED):
                vL = -MAX_SPEED
            if(vR > MAX_SPEED):
                vR = MAX_SPEED
            if(vR < -MAX_SPEED):
                vR = -MAX_SPEED
                
            vL = vL * .5
            vR = vR * .5
        
        
     ##################
    #####ODOMETRY#####
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

   # Actuator commands
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)