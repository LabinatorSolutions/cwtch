//
// write tidy epds to stdout for use with trainer.js.
//
// usage: node tidy > file
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

const epdFile = ''; // file to tidy
const wdl     = 5;  // index of wdl - extend getprob() as needed
const bi      = 0;  // index of board
const n       = 6;  // number space separated elements per line

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
  
  if (parts.length != n)
    return;
  
  console.log(parts[bi],parts[bi+1],parts[bi+2],parts[bi+3],getprob(parts[wdl]));
  
  /*}}}*/
});

rl.on('close', function(){
  process.exit();
});

