
const fs       = require('fs');
const readline = require('readline');
const tf       = require('@tensorflow/tfjs-node');

let restoreModel = false;

const hiddenSize = 128;
const batchSize  = 10000;
const scale      = 200;
const numEpochs  = 100;

/*{{{  more config*/

const dataFile       = 'data/no4b/shuffled_data.epd';
const fileLines      = 539648731;
const wdlPart        = 6;
const boardPart      = 0;
const numParts       = 8;

const weightsFile    = 'data/weights.js';
const binWeightsFile = 'data/weights.bin';
const modelFile      = 'file://./data/model';

const numBatches     = fileLines / batchSize | 0;

const inputSize      = 768;
const reportRate     = 5;

/*}}}*/

console.log('data', dataFile);
console.log('hidden size', hiddenSize);
console.log('batch size', batchSize);
console.log('num batchs', numBatches);
console.log('epochs', numEpochs);

if (process.argv[2] == 'r') {
  restoreModel = true;
  console.log('loading',modelFile);
}

/*{{{  getProb*/

function getProb (r) {

  if (r == '1/2-1/2')
    return 0.5;
  else if (r == '1-0')
    return 1.0;
  else if (r == '0-1')
    return 0.0;

  else if (r == '"1/2-1/2"')
    return 0.5;
  else if (r == '"1-0"')
    return 1.0;
  else if (r == '"0-1"')
    return 0.0;

  else if (r == '[0.5]')
    return 0.5;
  else if (r == '[1.0]')
    return 1.0;
  else if (r == '[0.0]')
    return 0.0;

  else if (r == '0.5')
    return 0.5;
  else if (r == '1.0')
    return 1.0;
  else if (r == '0.0')
    return 0.0;

  else {
    console.log('unknown result',r);
    process.exit();
  }
}

/*}}}*/
/*{{{  printModel*/

function printModel(model) {

  model.summary();

  console.log(model.optimizer.constructor.name);
  console.log(JSON.stringify(model.optimizer.getConfig(), null, 2));
  console.log(model.loss);
}

/*}}}*/
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
  let mse   = 0;

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
    
    if (parts.length == numParts) {
    
      targets[n][0] = getProb(parts[wdlPart]);
    
      decodeFEN(parts[boardPart]);
    
      for (var i=0; i < inputSize; i++) {
        inputs[n][i] = iLayer[i];
        if (iLayer[i] != 0 && iLayer[i] != 1)
          error('decode',line);
      }
    
      if (targets[n][0] != 0 && targets[n][0] != 1 && targets[n][0] != 0.5)
        error('prob',targets[n][0]);
    
      n++;
    }
    
    /*}}}*/

    if (n == batchSize) {
      /*{{{  train batch*/
      
      if (inputs.length != batchSize)
        error('inputs',inputs.length);
      
      if (targets.length != batchSize)
        error('targets',targets.length);
      
      shuffle(inputs, targets);
      
      x = tf.tensor(inputs);
      y = tf.tensor(targets);
      
      await tf.ready();
      
      r = await model.trainOnBatch(x,y);
      
      x.dispose();
      y.dispose();
      
      batch++;
      
      err += r[0];
      mse = err / batch;
      
      if ((batch % reportRate) == 0)
        process.stdout.write(epoch + ', ' + batch + ', ' + mse + '                \r');
      
      n = 0;
      
      /*}}}*/
    }
  }

  rl.close();
  fileStream.close();

  return mse;
}

/*}}}*/
/*{{{  train*/

async function train () {

  var model = 0;
  var mse   = 0;

  await tf.ready();

  if (restoreModel) {
    model = await tf.loadLayersModel(modelFile + '/model.json');
    if (model.layers[0].units != hiddenSize) {
      console.log("loaded model and hiddenSize don't match",model.layers[0].units,hiddenSize);
      process.exit();
    }
  }
  else {
    model = tf.sequential();
    model.add(tf.layers.dense({units: hiddenSize, inputShape: [inputSize],  name: 'hidden1', activation: 'relu'}));
    model.add(tf.layers.dense({units: 1,                                    name: 'output',  activation: 'sigmoid'}));
  }

  await tf.ready();

  model.compile({
    optimizer: tf.train.adam(),
    loss:      'meanSquaredError',
    metrics:   ['mse'],
  });

  await tf.ready();

  printModel(model);

  if (!restoreModel) {
    await saveWeights(model, 0, mse);
    await saveBinWeights(model);
    await model.save(modelFile);
  }

  for (let epoch=0; epoch < numEpochs; epoch++) {

    mse = await trainEpoch(model, epoch+1);

    console.log(epoch+1, mse, '          ');

    await saveWeights(model, epoch+1, mse);
    await saveBinWeights(model);
    await model.save(modelFile);
  }
}

/*}}}*/
/*{{{  saveWeights*/

async function saveWeights(model, epochs, mse) {

  const d = new Date();

  /*{{{  get weights*/
  
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
  
  /*}}}*/

  var o = '//{{{  weights\r\n\r\n';

  /*{{{  write scale etc*/
  
  o += 'const net_date       = "' + d         + '";\r\n';
  o += 'const net_dataFile   = ' + dataFile   + ';\r\n';
  o += 'const net_positions  = ' + fileLines  + ';\r\n';
  o += 'const net_h1_size    = ' + hiddenSize + ';\r\n';
  o += 'let   net_scale      = ' + scale      + ';\r\n';
  o += 'const net_epochs     = ' + epochs     + ';\r\n';
  o += 'const net_batchSize  = ' + batchSize  + ';\r\n';
  o += 'const net_numBatches = ' + numBatches + ';\r\n';
  o += 'const net_loss       = ' + mse        + ';\r\n';
  
  /*}}}*/
  /*{{{  write h1 weights*/
  
  o += 'const net_h1_w = Array(768);\r\n';
  
  var a = w['hidden1'][0];
  
  for (var i=0; i < inputSize; i++) {
    const a2 = a[i];
    o += 'net_h1_w[' + i + '] = [' + a2.toString() + '];\r\n';
  }
  
  /*}}}*/
  /*{{{  write h1 biases*/
  
  var a  = w['hidden1'][1];
  
  o += 'const net_h1_b = [' + a.toString() + '];\r\n';
  
  /*}}}*/
  /*{{{  write o weights*/
  
  var a  = w['output'][0];
  var a2 = Array(hiddenSize);
  
  for (var i=0; i < hiddenSize; i++) {
    a2[i] = a[i][0];
  }
  
  o += 'const net_o_w = [' + a2.toString() + '];\r\n';
  
  /*}}}*/
  /*{{{  write o bias*/
  
  var a = w['output'][1];
  
  o += 'const net_o_b = ' + a[0].toString() + ';\r\n';
  
  /*}}}*/

  o += '\r\n//}}}\r\n\r\n';

  //o += 'module.exports = {\r\n';
  //o += '  net_h1_size,\r\n'
  //o += '  net_h1_w,\r\n'
  //o += '  net_h1_b,\r\n'
  //o += '  net_o_w,\r\n'
  //o += '  net_o_b\r\n'
  //o += '};\r\n\r\n'

  fs.writeFileSync(weightsFile, o);
}

/*}}}*/
/*{{{  saveBinWeights*/

async function saveBinWeights(model) {

  /*{{{  get the weights from the model*/
  
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
  
  /*}}}*/

  const totalBytes = 4 + 768 * hiddenSize * 4 + hiddenSize * 4 + hiddenSize * 4 + 4;
  const buffer     = new ArrayBuffer(totalBytes);
  //const view       = new DataView(buffer);

  var start = 0;

  const hSize    = new Uint32Array(buffer, start, 1);                 start += 4;
  const hWeights = new Float32Array(buffer, start, 768 * hiddenSize); start += 768 * hiddenSize * 4;
  const hBiases  = new Float32Array(buffer, start, hiddenSize);       start += hiddenSize * 4;
  const oWeights = new Float32Array(buffer, start, hiddenSize);       start += hiddenSize * 4;
  const oBias    = new Float32Array(buffer, start, 1);

  hSize[0] = hiddenSize;

  var a = w['hidden1'][0];
  for (var i=0; i < inputSize; i++) {
    for (var j=0; j < hiddenSize; j++) {
      hWeights[i*hiddenSize+j] = a[i][j];
    }
  }

  var a = w['hidden1'][1];
  for (var j=0; j < hiddenSize; j++) {
    hBiases[j] = a[j];
  }

  var a = w['output'][0];
  for (var j=0; j < hiddenSize; j++) {
    oWeights[j] = a[j];
  }

  var a = w['output'][1];
  oBias[0] = a[0];

  const nodeBuffer = Buffer.from(buffer);

  fs.writeFileSync(binWeightsFile, nodeBuffer);
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

