# Cwtch

Javascript UCI chess engine with NNUE eval.

In development but usable. 

See uciExec() for command rep.

Currently a full width PVS searcher without any reductions or beta pruning or null move etc.

Runs in [Node](https://nodejs.org/en) which is available for pretty much every platform.

## Example

```
c:> node cwtch
u                       # shortcut for ucinewgame
net                     # show net info and some metrics
bench                   # useful time and node count when testing
position startpos
board                   # display the board
go depth 5
eval                    # show eval of current position 
et                      # eval tests
q                       # quit
```

## Chess User Interfaces

Can be run in chess UIs by using node as the exectable and cwtch.js as an argument or using a batch file etc.


