
fs = require('fs');

{{{  getprob

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

}}}

const epdFile = 'data/lichess-big3-resolved.epd';
const wdl     = 6;

var data  = fs.readFileSync(epdFile, 'utf8');
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

  console.log(parts[0],getprob(parts[wdl]));
}

process.exit();

