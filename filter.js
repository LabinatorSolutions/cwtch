//
// needs files in same format as trainer.js.
//
// filter positions if:-
//
//   bm is a capture
//   bm is a promotion
//   stm is in check
//
// usage: node filter
//

const fs = require('fs');
const readline = require('readline');

const epdFile = 'data/datagen/x.epd';
const outFile = 'data/datagen/xf.epd';

let o = '';
let n = 0;

fs.writeFileSync(outFile, '');

const rl = readline.createInterface({
  input: fs.createReadStream(epdFile),
  output: process.stdout,
  crlfDelay: Infinity,
  terminal: false
});

rl.on('line', function (line) {

  const parts = line.split(' ');

  cwtch.position(parts[0],parts[1],parts[2],parts[3]);

  cwtch.quiet = 1;
  cwtch.uciExec('go depth 5');

  if (moveIsNoisy(cwtch.bestMove)) {
    //console.log('filter noisy');
    return;
  }

  if (cwtch.bestMove & MOVE_PROMOTE_MASK) {
    //console.log('filter promote',formatMove(cwtch.bestMove));
    return;
  }

  const cx   = colourIndex(cwtch.turn);
  const list = cwtch.cxList[cx];
  if (cwtch.isKingAttacked(list[LKING], colourToggle(cwtch.turn))) {
    //console.log('filter in check');
    return;
  }

  o = o + line + '\n';
  n++;

  if ((n % 100) == 0)
    process.stdout.write(n+'\r');

  if (n > 10000) {
    fs.appendFileSync(outFile,o);
    o = '';
    n = 0;
  }
});

rl.on('close', function () {
  if (n)
    fs.appendFileSync(outFile,o);
  process.exit();
});

