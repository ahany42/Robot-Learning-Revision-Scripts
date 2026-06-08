ik = inverseKinematics('RigidBodyTree',robot)
weights = [0.25 0.25 0.25 1 1 1]
initialguess = robot.homeConfiguration;
[configSoln,solnInfo] = ik('endeffector',T,weights,initialguess)
theta1 = rad2deg(configSoln(1).JointPosition)
theta2 = rad2deg(configSoln(2).JointPosition)