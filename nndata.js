//
// Generate network input data from EPDs.
//

var epdFile  = 'data/quiet-labeled.epd';
var csvFile  = 'data/quiet-labeled.csv';
var wdlIndex = 5;

//{{{  constants

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

//}}}
//{{{  functions

//{{{  decodeFEN

function decodeFEN(board) {

  var x  = 0;
  var sq = 0;
  var b  = Array(768).fill(0);

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
      b[x] = 1;
      sq++;
    }
    else {
      sq += num;
    }
  }

  return b;
}

//}}}
//{{{  getprob

function getprob (r) {
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
  else {
    console.log('unknown result',r);
    process.exit();
  }
}

//}}}

//}}}

var thisPosition = 0;
var o            = '';

const fs       = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
    input: fs.createReadStream(epdFile),
    output: process.stdout,
    crlfDelay: Infinity,
    terminal: false
});

fs.writeFileSync(csvFile, '');

//{{{  test

var fen  = 'P7/8/8/8/8/8/8/8';
var test = decodeFEN(fen)
if (test[0] != 1) {
  console.log(fen);
  process.exit();
}

var fen  = 'p7/8/8/8/8/8/8/8';
var test = decodeFEN(fen)
if (test[384] != 1) {
  console.log(fen);
  process.exit();
}

var fen  = '1N6/8/8/8/8/8/8/8';
var test = decodeFEN(fen)
if (test[65] != 1) {
  console.log(fen);
  process.exit();
}


//}}}

rl.on('line', function (line) {

  thisPosition += 1;

  if (thisPosition % 100000 == 0)
    process.stdout.write(thisPosition+'\r');

  line = line.replace(/(\r\n|\n|\r|"|;)/gm,'');

  const parts = line.split(' ');

  if (!parts.length)
    return;

  var board = parts[0];
  var wdl   = getprob(parts[wdlIndex]);

  var b = decodeFEN(board).toString();
  b = b.replace(/(\r\n|\n|\r)/gm,'');

  o += wdl + ',' + b + '\r\n';

  if (thisPosition % 10000 == 0) {
    fs.appendFileSync(csvFile, o);
    o = '';
  }

});

rl.on('close', function(){
  if (o)
    fs.appendFileSync(csvFile, o);
  process.exit();
});

