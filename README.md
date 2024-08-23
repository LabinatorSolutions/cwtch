# Cwtch

An experimental Javascript UCI chess engine with NNUE evaluation.

Internal data generation and DIY Javascript trainer.

Currently +45 over Lozza with this internal data gen net:-

```
h1 size 75                                                                                                                                         
lr 0.001                                                                                                                                           
batch size 500                                                                                                                                     
activation relu                                                                                                                                    
stretch 100                                                                                                                                          
interp 0.5                                                                                                                                           
num_batches 227547                                                                                                                                   
opt Adam                                                                                                                                           
l2reg false                                                                                                                                        
epochs 157                                                                                                                                         
loss 0.023536278157536648
```

If you want to try cwtch, it can be fired in Node.js like Lozza. For details see the readme in the Lozza 2.5 release.

https://github.com/op12no2/lozza/releases/tag/2.5


