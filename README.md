# Cwtch
An experimental UCI Javascript chess engine with a simple NNUE evaluation, internal data generation and DIY Javascript trainer. No further work is planned. I just wanted to see if a (hand coded) Javascript NNUE could beat my traditional evaluation in Lozza. Yes!
```
tc=60+1
Score of cwtch vs lozza2.5: 411 - 252 - 220  [0.590] 883
...      cwtch playing White: 203 - 134 - 104  [0.578] 441
...      cwtch playing Black: 208 - 118 - 116  [0.602] 442
...      White vs Black: 321 - 342 - 220  [0.488] 883
Elo difference: 63.3 +/- 20.1, LOS: 100.0 %, DrawRatio: 24.9 %
SPRT: llr 2.96 (100.6%), lbound -2.94, ubound 2.94 - H1 was accepted
```
The network is very simple: 768 -> 75 (relu) -> 1. It was booted using WDL from quiet_labeled.epd and lichess-big3-resolved.epd (8M positions), but (inspired by the Stormphrax author) I spent some time experimenting with booting from a random init and that seems to work well too; I just didn't have the patience to go through with the necessary iterations. The boot net was then improved with two rounds of datagen/training using a lerp of 0.5 and around 114M positions. As some have pointed out 75 is a pretty weird layer size, but it seemed to be the best tradeoff between strength and speed for my implementation.

Again, I don't have the patience, but I think Javascript is probably fast enough for somebody to develop a NNUE hand coded engine with a CCRL Elo over 3000. 
