const R = require('ramda');

const SeedRandom = require("seed-random")(34);

let data = [
  { input: [0, 0], output: 0 },
  { input: [1, 0], output: 1 },
  { input: [0, 1], output: 1 },
  { input: [1, 1], output: 0 },
];

const weights = {
  i1_h1: SeedRandom(),
  i1_h2: SeedRandom(),
  i2_h1: SeedRandom(),
  i2_h2: SeedRandom(),
  h1_o1: SeedRandom(),
  h2_o1: SeedRandom(),
  bias_h1: SeedRandom(),
  bias_h2: SeedRandom(),
  bias_o1: SeedRandom(),
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const derivativeSigmoid = (x) => {
  const fx = sigmoid(x);
  return fx * (1 - fx);
};

const NN = (i1, i2) => {
  const h1_input = weights.i1_h1 * i1 + weights.i2_h1 * i2 + weights.bias_h1;
  const h1 = sigmoid(h1_input);

  const h2_input = weights.i1_h2 * i1 + weights.i2_h2 * i2 + weights.bias_h2;
  const h2 = sigmoid(h2_input);

  const o1_input = weights.h1_o1 * h1 + weights.h2_o1 * h2 + weights.bias_o1;
  const o1 = sigmoid(o1_input);

  return o1;
};

const showResult = () => {
  data.forEach(({ input: [i1, i2], output: y }) =>
    console.log(`${i1} XOR ${i2} => ${NN(i1, i2)} expected ${y}`)
  );
};

showResult();

const train = () => {
  const weighstDeltas = {
    i1_h1: 0,
    i1_h2: 0,
    i2_h1: 0,
    i2_h2: 0,
    h1_o1: 0,
    h2_o1: 0,
    bias_h1: 0,
    bias_h2: 0,
    bias_o1: 0,
  };

  for (const {
    input: [i1, i2],
    output,
  } of data) {
    const h1_input = weights.i1_h1 * i1 + weights.i2_h1 * i2 + weights.bias_h1;
    const h1 = sigmoid(h1_input);

    const h2_input = weights.i1_h2 * i1 + weights.i2_h2 * i2 + weights.bias_h2;
    const h2 = sigmoid(h2_input);

    const o1_input = weights.h1_o1 * h1 + weights.h2_o1 * h2 + weights.bias_o1;
    const o1 = sigmoid(o1_input);

    const delta = output - o1;
    const o1Delta = delta * derivativeSigmoid(o1_input);

    weighstDeltas.h1_o1 += h1 * o1Delta;
    weighstDeltas.h2_o1 += h2 * o1Delta;
    weighstDeltas.bias_o1 += o1Delta;

    const h1Delta = delta * derivativeSigmoid(h1_input);
    const h2Delta = delta * derivativeSigmoid(h2_input);

    weighstDeltas.i1_h1 += i1 * h1Delta;
    weighstDeltas.i2_h1 += i2 * h1Delta;
    weighstDeltas.bias_h1 += h1Delta;

    weighstDeltas.i1_h2 += i1 * h2Delta;
    weighstDeltas.i2_h2 += i2 * h2Delta;
    weighstDeltas.bias_h2 += h2Delta;
  }

  return weighstDeltas;
};

const aplplyTrainUpdate = (deltas = train()) => 
    Object.keys(weights).forEach((key) => {
        weights[key] += deltas[key]
    });

console.log('--------------------------------');

aplplyTrainUpdate();
showResult();

console.log('--------------------------------');
R.times(() => aplplyTrainUpdate(), 100);
showResult();

console.log('--------------------------------');
R.times(() => aplplyTrainUpdate(), 1000000);
showResult();