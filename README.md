# Cwtch

An experimental Javascript UCI chess engine with NNUE evaluation.

Internal data generation and DIY Javascript trainer.

Currently +45 over Lozza with this net:-

```
[C:\projects\cwtch]node cwtch n q
datagen soft-nodes 6000 hard-nodes 12000 random-ply 10 first-ply 16                                                                                
hidden-size 75                                                                                                                                     
learning-rate 0.001                                                                                                                                
batch-size 500                                                                                                                                     
activation relu                                                                                                                                    
sigmoid-stretch 100                                                                                                                                
score-wdl-lambda 0.5                                                                                                                               
num-batches 227547                                                                                                                                 
optimiser Adam                                                                                                                                     
l2-reg false                                                                                                                                       
epochs 157                                                                                                                                         
loss 0.023536278157536648                                                                                                                          
min h weight 0.000005815227268612944                                                                                                               
max h weight 98.90455627441406                                                                                                                     
min o weight 0.8036063313484192                                                                                                                    
max o weight 328.6114196777344
```

Data format:-

```
//
// 0                         1    2      3  4    5   6     7    8     9       10          11
// 8/8/8/8/6p1/5nk1/p7/3RrK2 w    -      -  3    169 -1124 d1e1 n     c       -           0.0
// board                     turn rights ep game ply score move noisy incheck givescheck  wdl
//
```

See the ```skipP``` function in ```trainer.js``` for how the data is filtered during training:-

https://github.com/op12no2/cwtch/blob/main/trainer.js#L450

If you want to try cwtch, it can be fired up in Node.js like Lozza. For details see the readme in the Lozza 2.5 release.

https://github.com/op12no2/lozza/releases/tag/2.5
