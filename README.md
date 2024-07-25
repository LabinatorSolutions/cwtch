# Cwtch

An experimental Javascript UCI chess engine with NNUE evaluation (`cwtch.js`).

The network is being evolved using a DIY Javascript trainer (`trainer.js`) and fast-chess self-play games bootstrapped from a tiny 768 x 1 linear material-only net. `parse.js` converts fast-chess .pgn files into .epd files (with a simplified syntax). `shuffle.js` shuffles lines in a text file.

.epd file format is:-

```
simplified-fen move-ply uci-move white-relative-eval in-check(0|1) gives-check(0|1) noisy(0|1) promotion(0|1) white-relative-result(1|0|0.5)
```

For example:-
```
3rkb2/1p2pp1p/7P/p1p1pQ2/P7/1P2PBr1/1B2PK2/3q3R b - - 53 g3f3 -19 0 1 1 0 0.5
```
