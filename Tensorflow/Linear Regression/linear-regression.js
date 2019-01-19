let xvals = [];
let yvals = [];
let m, b;
const learningRate = 0.5;
const optimazer = tf.train.sgd(learningRate);

function setup() {
	createCanvas(400, 400);
	m = tf.variable(tf.scalar(1));
	b = tf.variable(tf.scalar(0));
}

function mousePressed() {
	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, 0, height, 1,0);
	xvals.push(x);
	yvals.push(y);
	console.log("X: "+x+" - Y:"+y);
}

function draw() {
	background(0);
	stroke(255);
	strokeWeight(4);
	tf.tidy(() => {
		if(xvals.length > 0) {
			const ys = tf.tensor1d(yvals);
			optimazer.minimize(() => loss(predict(xvals), ys));
		}
	});
	for(let i = 0; i < xvals.length; i++) {
		let px = map(xvals[i], 0, 1, 0, width);
		let py = map(yvals[i], 0, 1, height, 0);
		point(px, py);
	}
	tf.tidy(() => {
		const lineX = [0, 1];
		const ys = predict([0,1]);
		ys.print();
		let x1 = map(lineX[0], 0, 1, 0, width);
		let x2 = map(lineX[1], 0, 1, 0, width);
		let lineY = ys.dataSync();
		let y1 = map(lineY[0], 0, 1, height, 0);
		let y2 = map(lineY[1], 0, 1, height, 0);
		line(x1, y1, x2, y2);
	});
}

function predict(x) {
	const xs = tf.tensor1d(x);
	const ys = xs.mul(m).add(b);
	return ys;
}

function loss(pred, labels) {
	return pred.sub(labels).square().mean();
}