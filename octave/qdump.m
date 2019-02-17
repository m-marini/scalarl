Q = csvread("maze-qdump.csv");
P = [1 : size(Q, 1)];
PP = [floor((P'-1) / 10) mod(P'-1, 10)];
Q=[PP Q];
