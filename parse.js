//
// parse fast-chess uci format pgn into epd
//
// use: node parse pgnFile epdFile
//

const fs = require('fs');
const readline = require('readline');

const pgnFile = process.argv[2];
const epdFile = process.argv[3];

fs.writeFileSync(epdFile, '');
let out = '';

//{{{  parseGame

function parseGame (g) {

  const uciMoves  = [];
  const lines     = g.split('\n');
  let   result    = getValue(lines[0]);
  const ply       = parseInt(getValue(lines[1]));

  //{{{  get a uci move list
  
  if (result != "1/2-1/2" && result != "1-0" && result != "0-1") {
    console.log('bad result', result);
    process.exit();
  }
  
  if (ply <= 10) {
    console.log('bad ply', ply);
    process.exit();
  }
  
  for (i=2; i < lines.length; i++) {
    const str = removeMoveNumbers(lines[i]).trim();
    if (str.length) {
      const m = str.split(' ');
      for (j=0; j < m.length; j++)
        uciMoves.push(m[j].trim());
    }
  }
  
  if (uciMoves[uciMoves.length-1] != result) {
    console.log('inconsistent results', result, uciMoves[uciMoves.length-1]);
    process.exit();
  }
  
  if (uciMoves.length != ply + 1) {
    console.log('inconsistent game length', ply, uciMoves.length);
    process.exit();
  }
  
  if (result == "1/2-1/2")
    result = 0.5;
  else if (result == "1-0")
    result = 1.0;
  else if (result == "0-1")
    result = 0.0;
  else {
    console.log('result format problem',result);
    process.exit();
  }
  
  //}}}

  cwtch.uciExec("u");
  cwtch.uciExec("p s");

  for (let i=0; i < uciMoves.length-2; i++) {

    const uciMove = uciMoves[i];
    const fen = cwtch.fen();

    let eval = cwtch.netFastEval();
    if (eval != cwtch.netSlowEval()) {
      console.log('eval inconsistency',eval,cwtch.netSlowEval());
      process.exit();
    }
    if (cwtch.turn == BLACK)
      eval = -eval

    //{{{  incheck?
    
    let turn     = cwtch.turn;
    let nextTurn = colourToggle(turn);
    let cx       = colourIndex(turn);
    let list     = cwtch.cxList[cx];
    
    const inCheck = cwtch.isKingAttacked(list[LKING], nextTurn);
    
    //}}}

    const move = cwtch.playMove(uciMove);

    //{{{  givescheck?
    
    turn     = cwtch.turn;
    nextTurn = colourToggle(turn);
    cx       = colourIndex(turn);
    list     = cwtch.cxList[cx];
    
    const givesCheck = cwtch.isKingAttacked(list[LKING], nextTurn);
    
    //}}}

    const noisy = moveIsNoisy(move) | 0;
    const promote = ((move & MOVE_PROMOTE_MASK) != 0) | 0;

    //console.log(fen, i, uciMove, eval, inCheck, givesCheck, noisy, promote, result);
    out += fen + ' ' + i + ' ' + uciMove + ' ' + eval + ' ' + inCheck + ' ' + givesCheck + ' ' + noisy + ' ' + promote + ' ' + result + '\n';
  }
}

//}}}
//{{{  getValue

function getValue(str) {
  const match = str.match(/"([^"]*)"/)
  return match ? match[1] : null
}

//}}}
//{{{  removeMoveNumbers

function removeMoveNumbers(str) {
  return str.replace(/\d+\.\s*/g, '');
}

//}}}

const rl = readline.createInterface({
  input: fs.createReadStream(pgnFile),
  output: process.stdout,
  crlfDelay: Infinity,
  terminal: false
});

let game  = '';
let games = 0;

rl.on('line', function (line) {

 if (game.length && line == '[Event "Fast-Chess Tournament"]') {
   games++;
   process.stdout.write(games+'\r');
   parseGame(game);
   game = '';
   if (out.length > 1000000) {
     fs.appendFileSync(epdFile, out);
     out = '';
   }
 }
 else if (line.length) {
   if (line.startsWith("[Sit")) return;
   if (line.startsWith("[Dat")) return;
   if (line.startsWith("[Rou")) return;
   if (line.startsWith("[Whi")) return;
   if (line.startsWith("[Bla")) return;
   if (line.startsWith("[Gam")) return;
   if (line.startsWith("[Ter")) return;
   if (line.startsWith("[Tim")) return;
   if (line.startsWith("[Eve")) return;

   const cleanLine = line.replace(/\s*\{[^}]*\}/g, '');
   game += cleanLine + '\n';
 }
});

rl.on('close', function () {
  games++;
  process.stdout.write(games+'\r');
  parseGame(game);
  if (out.length)
    fs.appendFileSync(epdFile, out);
  process.exit();
});

