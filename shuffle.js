
// call with --expose-gc

const fs = require('fs');
const readline = require('readline');

const epdFile     = 'data/no4b/data.epd';
const shuffleFile = 'data/no4b/shuffled_data.epd';
const chunkSize   = 5000000;

function shuffleChunk(a1) {
  for (let i = a1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a1[i], a1[j]] = [a1[j], a1[i]];
  }
}

function writeChunk() {
  if (chunk.length) {
    numChunks++;
    process.stdout.write(numChunks + '\r');
    const o = chunk.join('\r\n') + '\r\n';
    fs.appendFileSync(shuffleFile, o);
  }
}

let chunk     = [];
let numChunks = 0;

fs.writeFileSync(shuffleFile, '');

const rl = readline.createInterface({
  input: fs.createReadStream(epdFile),
  output: process.stdout,
  crlfDelay: Infinity,
  terminal: false
});

rl.on('line', function (line) {
  if (line.length) {
    chunk.push(line);
    if (chunk.length == chunkSize) {
      shuffleChunk(chunk);
      writeChunk();
      chunk = [];
      global.gc();
    }
  }
});

rl.on('close', function () {
  shuffleChunk(chunk);
  writeChunk();
  process.exit();
});

