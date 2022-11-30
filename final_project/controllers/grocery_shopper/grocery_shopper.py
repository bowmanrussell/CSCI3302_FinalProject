"""grocery controller"""

# Nov 2, 2022

from controller import Robot
import math
import numpy as np
from matplotlib import pyplot as plt

######### A(star) algorithm def and class 

class Node():
   """A node class for A* Pathfinding"""

   def __init__(self, parent=None, position=None):
       self.parent = parent
       self.position = position

       self.g = 0
       self.h = 0
       self.f = 0

   def __eq__(self, other):
       return self.position == other.position

    

def astar(maze, start, end):
   """Returns a list of tuples as a path from the given start to the given end in the given maze"""

   # Create start and end node
   start_node = Node(None, start)
   start_node.g = start_node.h = start_node.f = 0
   end_node = Node(None, end)
   end_node.g = end_node.h = end_node.f = 0

   # Initialize both open and closed list
   open_list = []
   closed_list = []

   # Add the start node
   open_list.append(start_node)

   # Loop until you find the end
   while len(open_list) > 0:


       # Get the current node
       current_node = open_list[0]

       current_index = 0
       for index, item in enumerate(open_list):
           if item.f < current_node.f:
               current_node = item
               current_index = index

       # Pop current off open list, add to closed list
       open_list.pop(current_index)
       closed_list.append(current_node)

       # Found the goal
       node_position = (current_node.position[0], current_node.position[1] )
       if current_node == end_node:
           path = []
           current = current_node
           while current is not None:
               path.append(current.position)
               current = current.parent
           return path[::-1] # Return reversed path

       # Generate children
       children = []
       for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

           # Get node position
           node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

           # Make sure within range
           if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
               continue

           # Make sure walkable terrain
           if maze[node_position[0]][node_position[1]] != 0:
               continue

           # Create new node
           new_node = Node(current_node, node_position)

           # Append
           if(new_node not in closed_list):
               children.append(new_node)

       # Loop through children
       for child in children:

           # Child is on the closed list
           for closed_child in closed_list:
               if child == closed_child:
                   continue

           # Create the f, g, and h values
           child.g = current_node.g + 1
           child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
           child.f = child.g + child.h

           # Child is already in the open list
           for open_node in open_list:
               if child == open_node and child.g > open_node.g:
                   continue

           # Add the child to the open list
           open_list.append(child)




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
pose_x = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
#lidat_angle_range was negative
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

#states
mode =  'manual' # mapping
#mode = get_item
#mode = follow_path

#initialize map
map = np.empty((360,360))

# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"


#
#
#






#converts robot coords into world coords for each stopping point
robot_stops = [(-5,0),(2.62,2.689),(3.504,2.689),(3.499,6.269),(5.802,6.26),(.777,1.27),(-2.665,1.1),(-2.665,1.1),(-.8647,1.2606),(2.341,-2.633), (-1.329,-4.953)]
i = 0
for x in robot_stops: 
    robot_stops[i] = (int(180+ x[0]*12), int((180 + x[1]*12)))
    i = i+1
  

object_theta = [0,pi/2,pi/2,pi/2,pi/2,-pi/2,-pi/2,-pi/2,-pi/2,-pi/2,pi/2] 
counter = 0  

#load map
#convolve map if not already

#get first start and end values in world coords
start = robot_stops[counter]
end = robot_stops[counter+1]
#if(mode != manual))#get first path before while loop
    #load map
    #convolve map if not already
    #start = robot_stops[counter]
    #end = robot_stops[counter+1]
    #path = astar(convolved_map,start,end) #path in world coords
    ##make waypoints in path robot coords
    #for p in path:
        #world_x = -(180- p[0])/12 
        #world_y = -(180 - p[1])/12 
        #waypoints.append((world_x,world_y))
    #last = len(waypoints-1)
    #end_point = waypoints[len(waypoints)-1]
    #waypoints = waypoints[::20]
    #waypoints.append(end_point)
    
        


while robot.step(timestep) != -1:

#path = astar(convolved_map,robot_stops[counter])

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



    n = compass.getValues()
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


    #####ODOMETRY#####
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0


   ###################
   #
   # Controller
   #
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
    else: # not manual mode
        

        distError= math.sqrt((waypoints[counter][0]-pose_x)**2+(waypoints[counter][1]-pose_y)**2)
        bearingError = math.atan2(waypoints[counter][1]-pose_y,waypoints[counter][0]-pose_x)-pose_theta 
        headingError = math.atan2(waypoints[counter+1][1] - waypoints[counter][1], waypoints[counter+1][0] - waypoints[counter][0]) - pose_theta
        if headingError < -3.14: headingError += 6.28 #keep angles within bounds

        if bearingError < -3.14: bearingError += 6.28

        if(distError < .1 and abs(bearingError) < 3.1415/4): #change waypoint on arrival
            #get new object and make new path
            counter = counter + 1

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
            waypoints.append(end_point)
            #shelf = useColorToLocateObject() 
            #the robot must be programmed to pick from the top and bottom shelves
            ##enter into rotation and manipulation modes to pick up object before going to next object
            continue

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
        
        












    

    pose_y = gps.getValues()[1]
    pose_x = gps.getValues()[0]
   # Actuator commands
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)