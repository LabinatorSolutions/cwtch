//
// Tune the network.
//

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const myInputs  = [];
const myTargets = [];

const dataFile        = 'data/quiet-labeled.csv';
const weightsFile     = 'data/weights.js';
const batchSize     = 100;
const shuffleBuffer = 1000;
const inputSize     = 768;
const hiddenSize    = 16;
const numEpochs     = 100;

//{{{  loadData

async function loadData() {

  const readline   = require('readline');
  const fileStream = fs.createReadStream(dataFile);

  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  var n = 0;

  for await (const line of rl) {

    if ((n % 10000) == 0)
      process.stdout.write(n+'\r');

    const parts = line.split(',');

    if (parts.length != inputSize+1) {
      console.log('data',parts.length);
      process.exit();
    }

    myTargets[n]    = Array(1).fill(0);
    myTargets[n][0] = parseFloat(parts[0]);

    myInputs[n] = Array(inputSize).fill(0);
    for (var i=0; i < inputSize; i++) {
      myInputs[n][i] = parseInt(parts[i+1]);
    }

    n++;
  }

  console.log('samples',n);
  console.log('batches',n / batchSize | 0);
  console.log('input length',myInputs.length);
  console.log('target length',myTargets.length);
}

//}}}
//{{{  createModel

function createModel() {

  const model = tf.sequential();

  model.add(tf.layers.dense({units: hiddenSize, inputShape: [inputSize], name: 'hidden'+hiddenSize, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1,                                   name: 'output',            activation: 'sigmoid'}));

  return model;
}

//}}}
//{{{  trainModel

async function trainModel() {

  console.log('loading data...');

  await loadData();

  const xs = tf.data.array(myInputs);
  const ys = tf.data.array(myTargets);

  console.log('creating dataset...');

  const xy = tf.data.zip({xs: xs, ys: ys})
    .batch(batchSize)
    .shuffle(shuffleBuffer);

  const model = createModel();

  model.compile({
    optimizer: 'adam',
    loss:      'meanSquaredError',
    metrics:   ['mse'],
  });

  console.log('training...');

  await saveWeights(model, 0);
  //process.exit();

  await model.fitDataset(xy, {
    epochs: numEpochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log('epoch', epoch+1, 'loss', logs.loss);
        saveWeights(model, epoch+1);
        xy.shuffle(shuffleBuffer);
      }
    }
  });

  await saveWeights(model);

}

//}}}
//{{{  saveWeights

async function saveWeights(model, epochs) {

  const d = new Date();

  const weights = {};
  const layers = model.layers;

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    weights[layer.name] = layer.getWeights();
  }

  const w = {};
  const layerNames = Object.keys(weights);

  for (let i = 0; i < layerNames.length; i++) {
    const layerName = layerNames[i];
    w[layerName] = [];
    const tensors = weights[layerName];
    for (let j = 0; j < tensors.length; j++) {
      w[layerName].push(tensors[j].arraySync());
    }
  }

  var o = '{{{  weights\r\n\r\n// epochs ' + epochs + ', ' + d + '\r\n\r\n';

  var iweights = w['hidden16'][0];
  var ibiases  = w['hidden16'][1];

  for (var i=0; i < inputSize; i++) {
    const w1 = iweights[i];
    o += 'NET_H1_W[' + i + '] = [' + w1.toString() + '];\r\n';
  }
  o += 'const NET_H1_B = [' + ibiases.toString() + '];\r\n';

  var iweights = w['output'][0];
  var ibiases  = w['output'][1];

  var iweights2 = Array(hiddenSize);
  for (var i=0; i < hiddenSize; i++) {
    iweights2[i] = iweights[i][0];
  }

  o += 'const NET_O_W = [' + iweights2.toString() + '];\r\n';
  o += 'const NET_O_B = ' + ibiases[0].toString() + ';\r\n';

  o += '}}}\r\n\r\n';

  fs.writeFileSync(weightsFile, o);
}

//}}}

trainModel();

// will get to here cos not async so don't assume trainModel has finished.


