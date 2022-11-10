"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space




#Code from: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2




MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
             "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
             "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
   robot_parts.append(robot.getDevice(part_names[i]))
   robot_parts[i].setPosition(float(target_pos[i]))
   robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
#mode = 'manual' # Part 1.1: manual mode
#mode = 'planner'
mode = 'autonomous'




###################
#
# Planner
#
###################
print('b4 planner ')
if mode == 'planner':
   print("IN the planner")


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










   # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
   start_w = (8.43571, 3.4553) # (Pose_X, Pose_Z) in meters
   end_w = (7, 9.8) # (Pose_X, Pose_Z) in meters

   # Convert the start_w and end_w from the webots coordinate frame into the map frame
   start = (int(start_w[0] * 30), 360 - int(start_w[1]* 30)) # (x, y) in 360x360 map
   end =  (int(end_w[0] * 30), 360 - int(end_w[1]* 30))# (x, y) in 360x360 map
   print(start,end)


   # Part 2.1: Load map (map.npy) from disk and visualize it
   map = np.load("map.npy")
   map = map > 0.5
   map = 1* map  

   # plt.imshow(np.fliplr(map))

   # plt.imshow(map) 
   # plt.show()  


    #Part 2.2: Compute an approximation of the “configuration space”
   kernel_size = 14
   Kernel = np.ones((kernel_size, kernel_size)) # Play with this number to find something suitable, the number corresponds to the # of pixels you want to cover
   Convolved_map = convolve2d(map, Kernel, mode='same') # You still have to threshold this convolved map

   Convolved_map = Convolved_map > 0.5
   Convolved_map = 1* Convolved_map 


   plt.imshow(np.fliplr(Convolved_map))

   plt.imshow(Convolved_map) 
   plt.show()  


   # Part 2.3 continuation: Call path_planner
   path = astar(Convolved_map, start, end)
   # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
   waypoints = []
   ##(int(end_w[0] * 30), 360 - int(end_w[1]* 30))##
   np.save("path.npy", path)
    
   print('path saved')

#TA reference:
# Part 2.3 continuation: Call path_planner
   #path = path_planner(cspace, start, end)

   # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
   #waypoints = []
   #for p in path:
   #    g = ((p[0]) / 30, p[1] / 30)
       #waypoints.append(g)
   #np.save("path", waypoints)
   #print("Path saved")

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.empty((360,360)) 




if mode == 'autonomous':
   # Part 3.1: Load path from disk and visualize it
    # Replace with code to load your path
   ################# display path on map
   path = np.load("path.npy").tolist() #path in world coordinates
   map = np.load('map.npy')
   map = map > 0.5
   map = 1* map 

   kernel_size = 12
   Kernel = np.ones((kernel_size, kernel_size)) # Play with this number to find something suitable, the number corresponds to the # of pixels you want to cover
   Convolved_map = convolve2d(map, Kernel, mode='same') # You still have to threshold this convolved map
   Convolved_map = Convolved_map > 0.5
   Convolved_map = 1* Convolved_map 

   cspace = np.empty((360,360))
   for p in path: cspace[p[0]][p[1]] = 2
   plt.imshow(cspace)
   plt.show()  
   waypoints = []
   for x in path: #!!! this works do not change this!!!
       world_x = (x[0] / 30)
       #world_y = (360 + x[1]) / 30
       world_y = 12 - ((x[1]/360) * 12)#(360 - int(end_w[1]* 30)) + x[1]/ 30
       waypoints.append((world_x,world_y))

   waypoints_hold = waypoints
   waypoints = waypoints_hold[::38] # gets every 15th pixel as waypoint
   waypoints.append([7.0,10]) #gets end_point#
   print(waypoints)
   counter = 1 # skips first(too close) waypoint
    


   pose_y = 3.4553
   pose_x =  8.43571
   while robot.step(timestep) != -1: #runs lab 3 code here because it works better than below while loop
       if(waypoints[counter][0]==0 and waypoints[counter][1]==0):
           robot_parts[MOTOR_LEFT].setVelocity(0)
           robot_parts[MOTOR_RIGHT].setVelocity(0)
           break

       if(counter >= len(waypoints)-1): #when we reach the last endpoint, stop moving
           robot_parts[MOTOR_LEFT].setVelocity(0)
           robot_parts[MOTOR_RIGHT].setVelocity(0)
           break


   # STEP 2.1: Calculate error with respect to current and goal position
       distError= math.sqrt((waypoints[counter][0]-pose_x)**2+(waypoints[counter][1]-pose_y)**2)
       bearingError = math.atan2(waypoints[counter][1]-pose_y,waypoints[counter][0]-pose_x)-pose_theta 
       headingError = math.atan2(waypoints[counter+1][1] - waypoints[counter][1], waypoints[counter+1][0] - waypoints[counter][0]) - pose_theta
       print('Next Waypoint position: ',waypoints[counter+1][0],waypoints[counter+1][1])
       print('Waypoint number: ', counter)
       if(counter >=3 and counter < 4): bearingError = bearingError + (3.1415/16)
       if(counter >= 4 and counter < 5): bearingError = bearingError + (3.1415/32)
       if headingError < -3.14: headingError += 6.28 #keep angles within bounds

       if bearingError < -3.14: bearingError += 6.28

       if(distError < .15 and abs(bearingError) < 3.1415/4): #change waypoint on arrival
           counter = counter + 1
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
   #pass   

   #pass

   # STEP 1: Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
   # Note that vL and vR in code is phi_l and phi_r on the slides/lecture

   #pass

   # STEP 2.3: Proportional velocities
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

   #pass

   # STEP 2.4: Clamp wheel speeds
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

   # TODO
   # Use Your Lab 2 Odometry code after these 2 comments. We will supply you with our code next week 
   # after the Lab 2 deadline but you free to use your own code if you are sure about its correctness

       distL = vL/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0

       distR = vR/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0

       pose_x += (distL+distR) / 2.0 * math.cos(pose_theta)

       pose_y += (distL+distR) / 2.0 * math.sin(pose_theta)

       pose_theta += (distR-distL)/AXLE_LENGTH


   ########## End Odometry Code ##################


       robot_parts[MOTOR_LEFT].setVelocity(vL)
       robot_parts[MOTOR_RIGHT].setVelocity(vR)


############################## matplotlib path.npy onto map.npy
#map = np.load("map.npy")


while robot.step(timestep) != -1 and mode != 'planner':
   if mode == 'autonomous':
       break

   ###################
   #
   # Mapping
   #
   ###################

   ################ v [Begin] Do not modify v ##################
   # Ground truth pose
   pose_y = -gps.getValues()[1]
   pose_x = -gps.getValues()[0]

   n = compass.getValues()
   rad = ((math.atan2(n[0], -n[2])))#-1.5708)
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
       wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
       wy =  +(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y



       ################ ^ [End] Do not modify ^ ##################

       #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

       if rho < LIDAR_SENSOR_MAX_RANGE:
           # Part 1.3: visualize map gray values.

           # You will eventually REPLACE the following 3 lines with a more robust version of the map
           # with a grayscale drawing containing more levels than just 0 and 1.
           try:
               map[int(wx*30)][360-int(wy*30)] = map[int(wx*30)][360-int(wy*30)] + .01


               color = (map[int(wx*30)][360-int(wy*30)]*256**2+map[int(wx*30)][360-int(wy*30)]*256+map[int(wx*30)][360-int(wy*30)])*255
               if map[int(wx*30)][360-int(wy*30)] + .01 >= 1: 
                   map[int(wx*30)][360-int(wy*30)] = 1
                   display.setColor(0xFFFFFF)
                   display.drawPixel(int(wx*30),360-int(wy*30))
               else: 
                   display.setColor(int(color))
                   display.drawPixel(int(wx*30),360-int(wy*30))
           except: pass
   # Draw the robot's current pose on the 360x360 display
   display.setColor(int(0xFF0000))

   #print(pose_x,pose_y,pose_theta)
   display.drawPixel(int(pose_x*30),360-int(pose_y*30))



   ###################
   #
   # Controller
   #
   ###################
   if mode == 'manual':
       key = keyboard.getKey()
       while(keyboard.getKey() != -1): pass
       if key == keyboard.LEFT :
           vL = -MAX_SPEED
           vR = MAX_SPEED
       elif key == keyboard.RIGHT:
           vL = MAX_SPEED
           vR = -MAX_SPEED
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
           # Part 1.4: Filter map and save to filesystem
           map = map > 0.5
           map = 1* map            
           np.save("map.npy",map)
           print("Map file saved")
       elif key == ord('L'):
           # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
           map = np.load("map.npy")
           print("Map loaded")
       else: # slow down
           vL *= 0.75
           vR *= 0.75
   else: # not manual mode
       x = 1


   # Odometry code. Don't change vL or vR speeds after this line.
   # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
   pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
   pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
   pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

   #print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

   # Actuator commands
   robot_parts[MOTOR_LEFT].setVelocity(vL)
   robot_parts[MOTOR_RIGHT].setVelocity(vR)