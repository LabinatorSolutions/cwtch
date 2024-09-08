# Cwtch
An experimental Javascript chess engine with a simple NNUE eval. Internal data generation and DIY Javascript trainer.
## Goal
Determine if a (hand-coded) Javascript NNUE is viable by testing against my HCE engine [Lozza](https://github.com/op12no2/lozza).
## Progress
```
build=1, tc=60+1
Score of cwtch vs lozza: 411 - 252 - 220  [0.590] 883
...      cwtch playing White: 203 - 134 - 104  [0.578] 441
...      cwtch playing Black: 208 - 118 - 116  [0.602] 442
...      White vs Black: 321 - 342 - 220  [0.488] 883
Elo difference: 63.3 +/- 20.1, LOS: 100.0 %, DrawRatio: 24.9 %
SPRT: llr 2.96 (100.6%), lbound -2.94, ubound 2.94 - H1 was accepted
```
## Plan
Try perspective, quantisation, adam2, srelu, screlu, halfk* (with >> self gen data), buckets, etc. At some point reboot from a random init or Lozza HCE.
## Acknowledgements
Engine Programming discord server for answering my newbie questions.
