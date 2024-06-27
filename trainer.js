
const fs       = require('fs');
const readline = require('readline');
const tf       = require('@tensorflow/tfjs-node');

const dataFile     = 'data/tidy.epd';
const weightsFile  = 'data/weights.js';
const modelFile    = 'file://./data/model';
const restoreModel = false;
const batchSize    = 100;
const inputSize    = 768;
const hiddenSize   = 16;
const numEpochs    = 100;
const reportRate   = 100;

console.log('batch size', batchSize);
console.log('epochs', numEpochs);

/*{{{  error*/

function error (x,y) {
  console.log(x,y);
  process.exit();
}

/*}}}*/
/*{{{  decodeEPD*/

/*{{{  constants*/

const WHITE = 0;
const BLACK = 1;

const PAWN   = 0;
const KNIGHT = 1;
const BISHOP = 2;
const ROOK   = 3;
const QUEEN  = 4;
const KING   = 5;

var chPce = [];
var chCol = [];
var chNum = [];

chPce['k'] = KING;
chCol['k'] = BLACK;
chPce['q'] = QUEEN;
chCol['q'] = BLACK;
chPce['r'] = ROOK;
chCol['r'] = BLACK;
chPce['b'] = BISHOP;
chCol['b'] = BLACK;
chPce['n'] = KNIGHT;
chCol['n'] = BLACK;
chPce['p'] = PAWN;
chCol['p'] = BLACK;
chPce['K'] = KING;
chCol['K'] = WHITE;
chPce['Q'] = QUEEN;
chCol['Q'] = WHITE;
chPce['R'] = ROOK;
chCol['R'] = WHITE;
chPce['B'] = BISHOP;
chCol['B'] = WHITE;
chPce['N'] = KNIGHT;
chCol['N'] = WHITE;
chPce['P'] = PAWN;
chCol['P'] = WHITE;

chNum['8'] = 8;
chNum['7'] = 7;
chNum['6'] = 6;
chNum['5'] = 5;
chNum['4'] = 4;
chNum['3'] = 3;
chNum['2'] = 2;
chNum['1'] = 1;

/*}}}*/

const iLayer = Array(768);

/*{{{  decodeFEN*/

function decodeFEN(board) {

  var x  = 0;
  var sq = 0;

  iLayer.fill(0);

  for (var j=0; j < board.length; j++) {

    var ch = board.charAt(j);

    if (ch == '/')
      continue;

    var num = chNum[ch];
    var col = 0;
    var pce = 0;

    if (typeof(num) == 'undefined') {
      if (chCol[ch] == WHITE)
        x = 0   + chPce[ch] * 64 + sq;
      else if (chCol[ch] == BLACK)
        x = 384 + chPce[ch] * 64 + sq;
      else
        console.log('colour');
      iLayer[x] = 1;
      sq++;
    }
    else {
      sq += num;
    }
  }
}

/*}}}*/

/*}}}*/
/*{{{  trainEpoch*/

async function trainEpoch(model, epoch) {

  let n     = 0;
  let batch = 0;
  let err   = 0;
  let x     = 0;
  let y     = 0;
  let r     = 0;

  const inputs  = Array(batchSize);
  const targets = Array(batchSize);

  for (var i = 0; i < batchSize; i++) {
    inputs[i]  = Array(inputSize);
    targets[i] = Array(1);
  }

  const fileStream = fs.createReadStream(dataFile);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  for await (const line of rl) {

    /*{{{  build batch*/
    
    const parts = line.split(' ');
    
    targets[n][0] = parseFloat(parts[1]);
    
    decodeFEN(parts[0]);
    
    for (var i=0; i < inputSize; i++) {
      inputs[n][i] = iLayer[i];
      if (iLayer[i] != 0 && iLayer[i] != 1)
        error('decode',line);
    }
    
    if (parts.length != 2)
      error('line format', line);
    
    if (targets[n][0] != 0 && targets[n][0] != 1 && targets[n][0] != 0.5)
      error('prob',targets[n][0]);
    
    /*}}}*/

    n++;

    if (n == batchSize) {

      if (inputs.length != batchSize)
        error('inputs',inputs.length);

      if (targets.length != batchSize)
        error('targets',targets.length);

      shuffle(inputs, targets);

      x = tf.tensor(inputs);
      y = tf.tensor(targets);

      r = await model.trainOnBatch(x,y);

      x.dispose();
      y.dispose();

      err += r[0];
      batch++;
      if ((batch % reportRate) == 0)
        process.stdout.write(epoch + ', ' + batch + ', ' + err/batch + '                \r');
      n = 0;
    }
  }

  console.log(epoch,err/batch,'                     ');

  rl.close();
  fileStream.close();
}

/*}}}*/
/*{{{  train*/

async function train () {

  var model = 0;

  if (restoreModel) {
    model = await tf.loadLayersModel(modelFile + '/model.json');
  }
  else {
    model = tf.sequential();
    model.add(tf.layers.dense({units: hiddenSize, inputShape: [inputSize], name: 'hidden'+hiddenSize, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1,                                   name: 'output',            activation: 'sigmoid'}));
    model.compile({
      optimizer: 'adam',
      loss:      'meanSquaredError',
      metrics:   ['mse'],
    });
  }

  await saveWeights(model, 0);
  await model.save(modelFile);

  for (let epoch=0; epoch < numEpochs; epoch++) {

    await trainEpoch(model, epoch+1);

    await saveWeights(model, epoch+1);
    await model.save(modelFile);
  }
}

/*}}}*/
/*{{{  saveWeights*/

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

  var o = '{{{  weights\r\n\r\n// epochs ' + epochs + ' batchsize ' + batchSize + ' ' + d + '\r\n\r\n';

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

/*}}}*/
/*{{{  shuffle*/

function shuffle(a1, a2) {

  for (var i = a1.length - 1; i > 0; i--) {

    var j = Math.floor(Math.random() * (i + 1));

    var t1 = a1[i];
    a1[i] = a1[j];
    a1[j] = t1;

    var t2 = a2[i];
    a2[i] = a2[j];
    a2[j] = t2;
  }
}

/*}}}*/

train();

