
//{{{  lang fold

//}}}

const fs = require('fs');
const readline = require('readline');

const dataFile = 'data/tidy.epd';
const weightsFile = 'data/weights.js';
const inputSize = 768;
const hiddenSize = 100;
const outputSize = 1;
const batchSize = 1000;
const epochs = 100;
const learningRate = 0.001;
const K = 1;
const maxActiveInputs = 32;
const beta1 = 0.9;
const beta2 = 0.999;
const epsilon = 1e-7;
const useCReLU = true;

//{{{  activations

function relu(x) {
  return Math.max(0, x);
}

function crelu(x) {
  return Math.min(Math.max(x, 0), 1);
}

function activationFunction(x) {
  return useCReLU ? crelu(x) : relu(x);
}

function activationDerivative(x) {
  if (useCReLU) {
    return (x > 0 && x < 1) ? 1 : 0;
  } else {
    return x > 0 ? 1 : 0;
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x * K));
}

//}}}
//{{{  initializeParameters

function initializeParameters() {
    const scale = Math.sqrt(2 / inputSize);

    const params = {
      W1: new Float32Array(inputSize * hiddenSize).map(() => (Math.random() * 2 - 1) * scale),
      b1: new Float32Array(hiddenSize).fill(0),
      W2: new Float32Array(hiddenSize).map(() => (Math.random() * 2 - 1) * scale),
      b2: 0,
      vW1: new Float32Array(inputSize * hiddenSize).fill(0),
      vb1: new Float32Array(hiddenSize).fill(0),
      vW2: new Float32Array(hiddenSize).fill(0),
      vb2: 0,
      sW1: new Float32Array(inputSize * hiddenSize).fill(0),
      sb1: new Float32Array(hiddenSize).fill(0),
      sW2: new Float32Array(hiddenSize).fill(0),
      sb2: 0
    };

    return params;
}

//}}}
//{{{  saveModel

function saveModel(params, epochs) {
  var o = '//{{{  weights\r\n';

  o += 'let   net_scale   = ' + 1          + ';\r\n';
  o += 'const net_h1_size = ' + hiddenSize + ';\r\n';
  o += 'const net_epochs  = ' + epochs     + ';\r\n';

  o += '//{{{  weights\r\n';

  //{{{  write h1 weights
  
  o += 'const net_h1_w = Array(768);\r\n';
  
  var a = params.W1;
  var a2 = [];
  
  for (var i=0; i < 768; i++) {
    a2 = [];
    const j = i * hiddenSize;
    for (var k=0; k < hiddenSize; k++) {
      a2.push(a[j+k]);
    }
    o += 'net_h1_w[' + i + '] = [' + a2.toString() + '];\r\n';
  }
  
  //}}}
  //{{{  write h1 biases
  
  var a  = params.b1;
  
  o += 'const net_h1_b = new Array([' + a.toString() + ']);\r\n';
  
  //}}}
  //{{{  write o weights
  
  var a = params.W2;
  
  o += 'const net_o_w = new Array([' + a.toString() + ']);\r\n';
  
  //}}}
  //{{{  write o bias
  
  var a = params.b2;
  
  o += 'const net_o_b = ' + a.toString() + ';\r\n';
  
  //}}}

  o += '\r\n//}}}\r\n';
  o += '\r\n//}}}\r\n\r\n';

  fs.writeFileSync(weightsFile, o);
}

//}}}
//{{{  forwardPropagation

function forwardPropagation(activeIndices, params) {
  const Z1 = new Float32Array(activeIndices.length * hiddenSize);
  const A1 = new Float32Array(activeIndices.length * hiddenSize);
  const Z2 = new Float32Array(activeIndices.length);
  const A2 = new Float32Array(activeIndices.length);

  for (let i = 0; i < activeIndices.length; i++) {
    for (let j = 0; j < hiddenSize; j++) {
      Z1[i * hiddenSize + j] = params.b1[j];
      for (const idx of activeIndices[i]) {
        Z1[i * hiddenSize + j] += params.W1[idx * hiddenSize + j];
      }
      A1[i * hiddenSize + j] = activationFunction(Z1[i * hiddenSize + j]);
    }
    Z2[i] = params.b2;
    for (let j = 0; j < hiddenSize; j++) {
      Z2[i] += A1[i * hiddenSize + j] * params.W2[j];
    }
    A2[i] = sigmoid(Z2[i]);
  }
  return { Z1, A1, Z2, A2 };
}

//}}}
//{{{  backwardPropagation

function backwardPropagation(activeIndices, targets, params, forward) {
  const m = activeIndices.length;
  const dZ2 = new Float32Array(m);
  const dW2 = new Float32Array(hiddenSize);

  let db2 = 0;

  const dA1 = new Float32Array(m * hiddenSize);
  const dZ1 = new Float32Array(m * hiddenSize);
  const dW1 = new Float32Array(inputSize * hiddenSize);
  const db1 = new Float32Array(hiddenSize);

  for (let i = 0; i < m; i++) {
    dZ2[i] = forward.A2[i] - targets[i];
    db2 += dZ2[i];
    for (let j = 0; j < hiddenSize; j++) {
      dW2[j] += dZ2[i] * forward.A1[i * hiddenSize + j];
      dA1[i * hiddenSize + j] = dZ2[i] * params.W2[j];
      dZ1[i * hiddenSize + j] = dA1[i * hiddenSize + j] * activationDerivative(forward.Z1[i * hiddenSize + j]);
      db1[j] += dZ1[i * hiddenSize + j];
      for (const idx of activeIndices[i]) {
        dW1[idx * hiddenSize + j] += dZ1[i * hiddenSize + j];
      }
    }
  }

  for (let j = 0; j < hiddenSize; j++) {
    dW2[j] /= m;
    db1[j] /= m;
  }

  db2 /= m;

  for (let i = 0; i < inputSize * hiddenSize; i++) {
    dW1[i] /= m;
  }

  return { dW1, db1, dW2, db2 };
}

//}}}
//{{{  updateParameters

function updateParameters(params, grads, t) {
  const updateParam = (param, grad, v, s, i) => {
    v[i] = beta1 * v[i] + (1 - beta1) * grad[i];
    s[i] = beta2 * s[i] + (1 - beta2) * grad[i] * grad[i];
    const vCorrected = v[i] / (1 - Math.pow(beta1, t));
    const sCorrected = s[i] / (1 - Math.pow(beta2, t));
    return param[i] - learningRate * vCorrected / (Math.sqrt(sCorrected) + epsilon);
  };

  for (let i = 0; i < inputSize * hiddenSize; i++) {
    params.W1[i] = updateParam(params.W1, grads.dW1, params.vW1, params.sW1, i);
  }

  for (let i = 0; i < hiddenSize; i++) {
    params.b1[i] = updateParam(params.b1, grads.db1, params.vb1, params.sb1, i);
    params.W2[i] = updateParam(params.W2, grads.dW2, params.vW2, params.sW2, i);
  }

  params.b2 = updateParam([params.b2], [grads.db2], [params.vb2], [params.sb2], 0);

  return params;
}

//}}}
//{{{  decodeLine

const WHITE = 0;
const BLACK = 1;

const PAWN = 0;
const KNIGHT = 1;
const BISHOP = 2;
const ROOK = 3;
const QUEEN = 4;
const KING = 5;

const chPce = {
  'k': KING, 'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT, 'p': PAWN,
  'K': KING, 'Q': QUEEN, 'R': ROOK, 'B': BISHOP, 'N': KNIGHT, 'P': PAWN
};

const chCol = {
  'k': BLACK, 'q': BLACK, 'r': BLACK, 'b': BLACK, 'n': BLACK, 'p': BLACK,
  'K': WHITE, 'Q': WHITE, 'R': WHITE, 'B': WHITE, 'N': WHITE, 'P': WHITE
};

const chNum = {'8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1};

function decodeLine(line) {
  const parts = line.split(' ');

  if (parts.length != 2) {
    console.log('line format', line);
    process.exit();
  }

  const board = parts[0].trim();
  const target = parseFloat(parts[1].trim());

  var x = 0;
  var sq = 0;

  const activeIndices = [];

  if (target != 0 && target != 1 && target != 0.5) {
    console.log('target format', line);
    process.exit();
  }

  for (var j = 0; j < board.length; j++) {
    var ch = board.charAt(j);
    if (ch == '/')
      continue;
    var num = chNum[ch];
    if (typeof (num) == 'undefined') {
      if (chCol[ch] == WHITE)
        x = 0 + chPce[ch] * 64 + sq;
      else if (chCol[ch] == BLACK)
        x = 384 + chPce[ch] * 64 + sq;
      else
        console.log('colour');
      activeIndices.push(x);
      sq++;
    }
    else {
      sq += num;
    }
  }

  if (activeIndices.length > 32 || activeIndices.length < 2) {
    console.log('activeIndices', activeIndices.length, activeIndices.toString());
    process.exit();
  }

  return { activeIndices, target: [target] };
}

//}}}
//{{{  train

async function train(filename) {
  let params = initializeParameters();

  saveModel(params,0);

  let t = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
      const fileStream = fs.createReadStream(filename);
      const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
      });

      let batchActiveIndices = [];
      let batchTargets = [];
      let batchCount = 0;
      let totalLoss = 0;

      for await (const line of rl) {
        const {activeIndices, target} = decodeLine(line);

        batchActiveIndices.push(activeIndices);
        batchTargets.push(target[0]);

        if (batchActiveIndices.length === batchSize) {
          t++;
          const forward = forwardPropagation(batchActiveIndices, params);
          const grads = backwardPropagation(batchActiveIndices, batchTargets, params, forward);
          params = updateParameters(params, grads, t);

          const batchLoss = forward.A2.reduce((sum, pred, i) =>
            sum + Math.pow(pred - batchTargets[i], 2), 0) / batchSize;
          totalLoss += batchLoss;
          batchCount++;

          batchActiveIndices = [];
          batchTargets = [];

          if (batchCount % 10 === 0) {
            process.stdout.write(`Epoch ${epoch + 1}, Batch ${batchCount}, Average Loss: ${totalLoss / batchCount}\r`);
          }
        }
      }

      console.log(`Epoch ${epoch + 1} completed. Average Loss: ${totalLoss / batchCount}`);
      rl.close();
      saveModel(params, epoch+1);
    }

    return params;
}

//}}}

train(dataFile).then(params => {
    console.log('Training completed.');
}).catch(error => {
    console.error('Error during training:', error);
});


