load('models/net_rl.mat');

conv1f = net.params(1).value;
conv1b = net.params(2).value;
conv2f = net.params(3).value;
conv2b = net.params(4).value;
conv3f = net.params(5).value;
conv3b = net.params(6).value;
fc4f = net.params(7).value;
fc4b = net.params(8).value;
fc5f = net.params(9).value;
fc5b = net.params(10).value;
fc6_1f = net.params(11).value;
fc6_2f = net.params(13).value;

save('net_rl_weights', '-v7.3', 'conv1f', 'conv1b', 'conv2f', 'conv2b', 'conv3f', 'conv3b', 'fc4f', 'fc4b', 'fc5f', 'fc5b', 'fc6_1f', 'fc6_2f')

fprintf('done\n');