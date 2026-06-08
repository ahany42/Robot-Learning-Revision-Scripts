gw = generateGameGrid(10,10)
allStates = gw.States
allStates = allStates(allStates == "[10,10]" | allStates == "[1,1]") = []

shuffledStates = allStates(randperm(numel(allStates)))
gw.ObstaclesStates = shuffledStates(1:10)

gw.terminal = "[10,10]"
setReward(gw,"[10,10]",100)
updateStateTransitionForObstacles(gw)

env = rlMDPEnv(gw)
env.ResetFcn = @()2

qTable = rlTable(getObservationInfo(env),getActionInfo(env))
qFunAppx = rlQValueFunction(qTable,getObservationinfo(env),getActionInfo(env))

qAgent = rlQAgent(qFunAppx);
qAgent.AgentOptions.EpsilonGreedyExploration = 0.1
qAgent.CriticOptimizerOptions.LearnRate = 0.1