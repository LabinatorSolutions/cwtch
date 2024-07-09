//
// Write tidy epds to stdout.
//

const fs       = require('fs');
const readline = require('readline');

/*{{{  getprob*/

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

const epdFile = 'data/lichess-big3-resolved.epd';
const wdl     = 6;
const bi      = 0;

const rl = readline.createInterface({
    input: fs.createReadStream(epdFile),
    output: process.stdout,
    crlfDelay: Infinity,
    terminal: false
});

rl.on('line', function (line) {
  /*{{{  process line*/
  
  line = line.replace(/(\r\n|\n|\r|"|;)/gm,'');
  line = line.replace(/,/gm,' ');
  line = line.trim();
  
  //console.log(line);
  
  if (!line.length)
    return;
  
  var parts = line.split(' ');
  
  if (!parts.length)
    return;
  
  console.log(parts[bi],getprob(parts[wdl]));
  
  /*}}}*/
});

rl.on('close', function(){
  process.exit();
});

