clear all;
FILES = [
  "lander-dump-111.csv";
  "lander-dump-112.csv";
  "lander-dump-113.csv";
];

n = size(FILES, 1);
L = 100;
R = [];
E = [];
for i = 1 : n
  [Y X] = slideAvg(trend(csvread(FILES(i, :))), L, L);
  R(:, i) = Y(:, 1);
  E(:, i) = Y(:, 2);
endfor

subplot(1,2,1);
plot(X, R);
title("Average Reward")
grid on
grid minor on;
legend(FILES, "location", "southeast");
xlabel("Step");
ylabel("Reward");

subplot(1,2,2);
plot(X, E);
title("Average Score")
grid on
grid minor on;
legend(FILES, "location", "southeast");
xlabel("Step");
ylabel("Score");
