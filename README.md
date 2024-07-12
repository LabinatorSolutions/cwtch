# Cwtch

Javascript UCI chess engine with NNUE eval.

In development but usable. 

See uciExec() for command rep.

Currently a full width PVS searcher without any reductions or beta pruning or null move etc.

Runs in [Node](https://nodejs.org/en) which is available for pretty much every platform.

## Example

```
c:> node cwtch
ucinewgame
net
bench
position startpos
board
go depth 5
eval
et
q
```

## Chess UIs

Can be run in chess UIs by using node at the exectable and cwtch.js as an argument or using a batch file etc.


