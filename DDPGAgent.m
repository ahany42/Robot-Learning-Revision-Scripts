mdl ="rlQuadrupedRobot"
numObs = 44
numAct = 8
obsInfo = rlNumericSpecs([numObs 1])
actInfo = rlNumericSpec([numAct 1],"UpperLimit",1,"LowerLimit",-1)
env = rlSimulinkEnv(mdl,mdl+"/RL_Agents",obsInfo,actInfo)

actnet = [
    featureInputLayer(numObs,"Name","obs")
    fullyConnectedLayer(100,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(100,"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(100,"Name","fc3")
    reluLayer("Name","relu3")
    fullyConnectedLayer(numAct,"Name","actOut")
    tanhLayer("Name","scat")

]
actnet = dlnetwork(actnet)
actor = rlDeterministicActorRepresentation(actnet,obsInfo,actInfo,"Observation","obs","Action","act")

obsPath = [
    featureInputLayer(numObs,"Name","obs")
    fullyConnectedLayer(100,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(100,"Name","fc2")
    addinonalLayer(2,"Name","add")
    reluLayer("Name","relu2")
    fullyConnectedLayer(100,"Name","fc3")
    reluLayer("Name","relu3")
    fullyConnectedLayer(1,"Name","value")

]
actPath = [
    featureInputLayer(numAct,"Name","act")
    fullyConnectedLayer(100,"Name","fact")
]
qvalnet = layerGraph(obsPath)
qvalnet = addLayers(qvalnet,actPath)
qvalnet = connectLayers(qvalnet,"fcact","add/in2")
qvalnet = dlnetwork(qvalnet)

critic = rlQValueRepresentation(qvalent,obsInfo,actInfo,"Observation","obs","Action","act")
agent = rlDDPGAgent(actor,critic)

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 5000;
trainOpts.MaxStepsPerEpisode = 1000;
trainOpts.ScoreAveragingWindowLength= 50;
trainOpts.StopTrainingCriteria= 'Average Reward';
trainOpts.StopTrainingValue= 500;
trainOpts.Plots= "training-progress";
trainingStats = train(agent,env,trainOpts);