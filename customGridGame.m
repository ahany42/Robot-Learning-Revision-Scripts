gw = createGameWorld(10,10);
allStates = gw.States;
allStates(allStates == "[1,1]"| allStates == "[10,10]") = [];

shuffledStates = allStates(randperm(numel(allStates)));
gw.ObstacleStates = shuffledStates(1:10);

gw.TerminalStates = "[10,10]";
setReward(gw,"[10,10]",100);
updateStateTransitionForObstacles(gw);

env = rlMDPEnv(gw);
env.ResetFcn = @() 2;

qTable = rlTable(getObservationInfo(env),getActionInfo(env));
qFncAppx = rlQValueFunction(qTable,getObservationInfo(env),getActionInfo(env));

qAgent = rlQAgent(qFncAppx);
qAgent.AgentOptions.EpsilonGreedyExploration.Epsilon = 0.1;
qAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 0.1;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes = 500;
trainOpts.MaxStepsPerEpisode = 100;
trainOpts.ScoreAverageWindowLength = 30;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 80;
trainOpts.plot = "training-progress";

trainingStats = train(qAgent,env,trainOpts);
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;