# Cwtch
An experimental UCI Javascript chess engine with a simple NNUE eval, internal data generation and DIY Javascript trainer.
```
tc=60+1
Score of cwtch1 vs lozza2.5: 411 - 252 - 220  [0.590] 883
...      cwtch1 playing White: 203 - 134 - 104  [0.578] 441
...      cwtch1 playing Black: 208 - 118 - 116  [0.602] 442
...      White vs Black: 321 - 342 - 220  [0.488] 883
Elo difference: 63.3 +/- 20.1, LOS: 100.0 %, DrawRatio: 24.9 %
SPRT: llr 2.96 (100.6%), lbound -2.94, ubound 2.94 - H1 was accepted
```
## Acknowledgements
The [Engine Programming](https://discord.com/invite/F6W6mMsTGN) discord server for answering my newbie NNUE and data gen. questions.

The [Chess Programming Wiki](https://www.chessprogramming.org). Most of Cwtch's algorithms come from here.

Stockfish's [NNUE](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) Overview.

I initially used some of the search magic numbers from [Ethereal](https://github.com/AndyGrant/Ethereal) but I'm gradually tuning them to values that suit Cwtch. I also used Ethereal's aspiration window algorithm but again it's gradually morphing to suit Cwtch. 

Logo [raygirl](https://www.deviantart.com/raygirl)
