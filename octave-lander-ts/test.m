X=csvread ("../lander-samples.csv" );
Y=processSamples(X);
csvwrite("../lander-training.csv", Y);
