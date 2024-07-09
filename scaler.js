//
// Find a reasonable scaling factor for the net using the classic Texel tuning sigmoid.
//

cwtch.newGame();

fs    = require('fs');
board = cwtch.board;

var epds   = [];
var params = [];

var gFiles = [
  {wdl: 5, file: 'data/quiet-labeled.epd'},
  {wdl: 6, file: 'data/lichess-big3-resolved.epd'}
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

  net_scale = scale;

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

  console.log(scale,err);

  return err;
}

//}}}
//{{{  findScale

function findScale (minScale, maxScale) {

  while (minScale < maxScale) {

    var mid1 = Math.floor(minScale + (maxScale - minScale) / 3);
    var mid2 = Math.floor(maxScale - (maxScale - minScale) / 3);

    var loss1 = tryScale(mid1);
    var loss2 = tryScale(mid2);

    if (loss1 < loss2)
      maxScale = mid2 - 1;
    else
      minScale = mid1 + 1;
  }

  return minScale;
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

console.log('scale =', net_scale);
console.log('h1 =', net_h1_size);
console.log('epochs =', epochs);

var best = findScale(0, 500);
console.log(best);

process.exit();

