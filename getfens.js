//
// grab fens from file and send to stdout.
//
// use: node getfens fenfile boardindex
//

const fs = require('fs');
const readline = require('readline');

const fenFile = process.argv[2];
const bi      = process.argv[3];

const rl = readline.createInterface({
  input: fs.createReadStream(pgnFile),
  output: process.stdout,
  crlfDelay: Infinity,
  terminal: false
});

rl.on('line', function (line) {

  parts = line.split(' ');

  if (parts.length > 3)
    console.log(parts[bi++],parts[bi++],parts[bi++],parts[bi++]);

});

rl.on('close', function () {
  process.exit();
});

