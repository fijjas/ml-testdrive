const ml = require('machine_learning');


let x = [
	[0.4, 0.5, 0.5, 0.,  0.,  0., 0.4, 0.5, 0.5, 0.,  0.,  0.],
	[0.5, 0.3, 0.5, 0.,  0.,  0., 0.5, 0.3, 0.5, 0.,  0.,  0.],
	[0.4, 0.5, 0.5, 0.,  0.,  0., 0.4, 0.5, 0.5, 0.,  0.,  0.],
	[0.,  0.,  0.5, 0.3, 0.5, 0., 0.,  0.,  0.5, 0.3, 0.5, 0.],
	[0.,  0.,  0.5, 0.4, 0.5, 0., 0.,  0.,  0.5, 0.4, 0.5, 0.],
	[0.,  0.,  0.5, 0.5, 0.5, 0., 0.,  0.,  0.5, 0.5, 0.5, 0.],
];

console.log('Demo Data: ', x);

let y = [
	[1, 0, 1, 1],
	[1, 0, 1, 1],
	[1, 0, 1, 1],
	[0, 1, 0, 1],
	[0, 1, 0, 1],
	[0, 1, 0, 1],
];

console.log('Demo Results: ', y);

const mlp = new ml.MLP({
	input: x,
	label: y,
	n_ins: 12,
	n_outs: 4,
	hidden_layer_sizes : [12, 8, 4],
});

mlp.set('log level', 1); // 0 : nothing, 1 : info, 2 : warning.

mlp.train({
	'lr' : 0.6,
	'epochs' : 100000
});

let a = [
	[0.2, 0.2, 0.,  0.,  0.1, 0., 0.1, 0.,  0.1, 0., 0.1, 0.],
	[0.,  0.,  0.,  0.5, 0.5, 0., 0.1, 0.2, 0.1, 0., 0.1, 0.],
	[0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.4, 0.,  0.1, 0., 0.1, 0.],
];

console.log('Real Input Data: ', a);

console.log('Predict...');

let results = mlp.predict(a);

console.log('Results are: ', results);
