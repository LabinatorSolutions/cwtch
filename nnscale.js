//
// Find a reasonable scaling factor using the classic Texel tuning sigmoid.
//

cwtch.newGame();

fs    = require('fs');
board = cwtch.board;

var epds   = [];
var params = [];

var gFiles = [
  {wdl: 5, file: 'data/quiet-labeled.epd'}
];

//{{{  functions

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
  else if (r == '[0.5]')
    return 0.5;
  else if (r == '[1.0]')
    return 1.0;
  else if (r == '[0.0]')
    return 0.0;
  else {
    console.log('unknown result',r);
    process.exit();
  }
}

//}}}
//{{{  sigmoid

function sigmoid (x) {
  return 1.0 / (1.0 + Math.pow(10.0,-x/400.0));
}

//}}}
//{{{  tryScale

function tryScale (scale) {

  SCALE = scale;

  var err = 0;

  for (var i=0; i < epds.length; i++) {

    var epd = epds[i];

    cwtch.position(epd.board,epd.turn,epd.rights,epd.ep);

    var pr = epd.prob;
    var ev = cwtch.netFastEval();

    if (cwtch.turn == BLACK)
      ev = -ev;

    var sg = sigmoid(ev);

    err += (pr-sg) * (pr-sg);
  }

  err = err / epds.length;

  process.stdout.write(scale+' '+err+'\r');
  return err;
}

//}}}
//{{{  findScale

function findScale () {

    console.log('finding scale...');

    var start =  0;
    var finish = 500;
    var step = 100;
    var err = 0;
    var best = 99999;
    var scale = 0;

    for (var i = 0; i < 3; i++) {

      scale = start - step;
      while (scale < finish) {
        scale += step;
        err = tryScale(scale);
        if (err <= best) {
          best = err,
          start = scale;
        }
      }

      console.log(start, best);

      finish = start + step;
      start = start - step;
      step  = step  / 10.0;
    }

    return best;
}

//}}}

//}}}
//{{{  load the epds

epds = [];

for (var j=0; j < gFiles.length; j++) {

  var f = gFiles[j];

  console.log('loading',f.file);

  var data  = fs.readFileSync(f.file, 'utf8');
  var lines = data.split('\n');

  data = '';

  for (var i=0; i < lines.length; i++) {

    var line = lines[i];

    line = line.replace(/(\r\n|\n|\r|"|;)/gm,'');
    line = line.trim();

    if (!line.length)
      continue;

    var parts = line.split(' ');

    if (!parts.length)
      continue;

    epds.push({board:   parts[0],
               turn:    parts[1],
               rights:  parts[2],
               ep:      parts[3],
               prob:    getprob(parts[f.wdl])});
  }

  lines = [];
}

//}}}

var scale = findScale();
console.log('tuned scale', scale);

process.exit();

