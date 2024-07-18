# Cwtch

A Javascript UCI chess engine with NNUE eval.

In development but usable. 

See uciExec() for command rep.

Runs in [Node](https://nodejs.org/en) which is available for pretty much every platform.

## Example

```
c:> node cwtch
u                       # shortcut for ucinewgame
net                     # show net info and some metrics
bench                   # useful time and node count when testing
p s                     # shortcut for position startpos
board                   # display the board
g d 5                   # shortcut for go depth 5
e                       # show eval of current position 
et                      # eval tests
q                       # quit
```
`pt` does perft tests but takes a long time:-
```
c:> node cwtch pt q
```
## Chess User Interfaces

Cwtch can be run in chess UIs by using Node as the exectable and cwtch.js as an argument or using a batch file etc. On Linux systems (or WSL) you can use a shebang.

## The Network

Currently the network is simple unquantized relu white-relative 768 -> 70 -> 1 net trained on a relatively small number of unshuffled EPDs (7.8M) from quiet_lebaled.epd and lichess-big3-resolved.epd. The future plan is to train from 'zero' and use my own data.

## The Trainer

The trainer is Javascript running in Node with a sigmoid stretch of 100. It does not use any network libraries. It generates weights in Javascript syntax that can be copied into the engine. It streams batches from a single file and does not use much memory.





