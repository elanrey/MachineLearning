const model = tf.sequential();

const hidden = tf.layers.dense({
    units: 3,
    inputShape: 2,
    activation: "sigmoid"
});
model.add(hidden);

const output = tf.layers.dense({
    units: 1,
    activation: "sigmoid"
});
model.add(output);

model.compile({
    optimizer: tf.train.sgd(0.5),
    loss: tf.losses.meanSquaredError
});

const inputs = tf.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]);

const outputs = tf.tensor([
    [0],
    [0],
    [1],
    [1]
]);

async function train() {
    const config = {
        shuffle: true,
        epochs: 1
    }
    for(let i = 0; i < 10000; i++) {
        const history = await model.fit(inputs, outputs, config).then((resp) => console.log(resp.history.loss));
    }
}

train().then(() => {
    const inputData = tf.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25]
    ]);
    let prediction = model.predict(inputData);
    prediction.print();
});