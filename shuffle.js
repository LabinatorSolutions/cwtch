//
// shuffle lines in a large file with chunkSize granulatiry.
//
// usage: node --expose-gc shuffle
//

const fs = require('fs');
const readline = require('readline');

const epdFile     = 'data/tidy.epd';
const shuffleFile = 'data/shuffled_tidy.epd';
const chunkSize   = 4000000;

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

