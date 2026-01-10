# KEYCONTROL + MAPPING + STREAMING + IMAGE CAPTURE FOR NAVIGATION 
from djitellopy import tello
import time
import KEYMODULE as dr
import cv2 
import numpy as np 
import math

############# PARAMETERS #############
fspeed = 117 / 10 # FORWARD SPEED (cm/s)
aspeed = 360 / 10 # ANGULAR SPEED (degrees/s)
interval = 0.25 

dinterval = fspeed * interval
ainterval = aspeed * interval 
######################################

x, y = 500, 500     
yaw = 0
points = []
global img 

# DRONE CONNECTION
drone = tello.Tello()
drone.connect()
print(f"BATTERY: {drone.get_battery()}%\n")
print(f"TEMPERATURE: {drone.get_temperature()}*C")
drone.streamon()

# DRONE CONTROL WINDOW
dr.init() 


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50 
    global yaw, x, y
    d = 0
    moving_angle = 0 
    state = False # IF THE DRONE IS MOVING => TRUE 

    # LEFT / RIGHT
    if dr.getKey("LEFT"): 
        lr = -speed
        d = dinterval 
        moving_angle = 180 
        state = True
    elif dr.getKey("RIGHT"): 
        lr = speed
        d = dinterval 
        moving_angle = 0 
        state = True
        
    # FORWARD / BACKWARD 
    if dr.getKey("UP"): 
        fb = speed
        d = dinterval 
        moving_angle = 90
        state = True
    elif dr.getKey("DOWN"): 
        fb = -speed
        d = -dinterval 
        moving_angle = 270 
        state = True

    # ROTATE 
    if dr.getKey("d"): 
        yv = speed
        yaw += ainterval
        state = True
    elif dr.getKey("a"): 
        yv = -speed
        yaw -= ainterval
        state = True 

    # UP / DOWN 
    if dr.getKey("w"): ud = speed
    elif dr.getKey("s"): ud = -speed

    # LAND / TAKEOFF
    if dr.getKey("q"): drone.land(); time.sleep(2)
    elif dr.getKey("e"): drone.takeoff()
    
    # CAPTURE IMAGE FOR NAVIGATION
    if dr.getKey("c"): 
        cv2.imwrite(f'Resources/Image/{time.time()}.jpg', img_cam)
            
    # OXY COORDINATES 
    if state:
        time.sleep(interval) # REAL TIME DELAY 
        angle_rad = math.radians(yaw + moving_angle)        
        x += int(d * math.cos(angle_rad))
        y += int(d * math.sin(angle_rad))
    return [lr, fb, ud, yv, x, y]

# MAPPING 
def drawPoints(img, points, height):

    # DRONE TRAJECTORY 
    for point in points:
        cv2.circle(img, point, 2, (0, 0, 255), cv2.FILLED) 
    
    # DRONE POSITION AT TIME t 
    #cv2.drawMarker(img, pos, color, markertype, size, thickness)
    cv2.drawMarker(img, points[-1], (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 15, 2)
    cv2.putText(img, f'({(points[-1][0]-500)/100:.2f}, {-(points[-1][1]-500)/100:.2f}, {height/100:.2f})m',
                (points[-1][0]+10, points[-1][1]+30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)

while True:
    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    height = drone.get_height()

    # VIDEO 
    try:
        frame_read = drone.get_frame_read()
        myFrame = frame_read.frame
        if myFrame is not None:
            img_cam = cv2.resize(myFrame, (360, 240))
            cv2.imshow("TELLO'S CAMERA", img_cam)
    except Exception as e:
        print("CAMERA ERROR:", e)

    # MAP
    img_map = np.zeros((1000, 1000, 3), np.uint8)
    if len(points) == 0 or points[-1] != (vals[4], vals[5]):
        points.append((vals[4], vals[5]))
    drawPoints(img_map, points, height)

    cv2.imshow("TRAJECTORY MAP", img_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
