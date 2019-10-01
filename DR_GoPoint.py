from LabDeltaRobot import *
import time

#### Call DeltaRobot class ####
deltarobot = DeltaRobot()

deltarobot.RobotTorqueOn()
deltarobot.GripperTorqueOn()

################## Go to stand by position before starting  ###########################

deltarobot.GoHome()
deltarobot.GripperCheck()
time.sleep(0.5)
#deltarobot.GotoPoint(0.0,0.0,-355.0)
'''
deltarobot.GotoPoint(150,150,-700)
deltarobot.GripperClose()

deltarobot.GotoPoint(-150,150,-700)
deltarobot.GripperOpen()

deltarobot.GotoPoint(-150,-150,-700)
deltarobot.GripperClose()

deltarobot.GotoPoint(150,-150,-700)
deltarobot.GripperOpen()

deltarobot.GoHome()
'''