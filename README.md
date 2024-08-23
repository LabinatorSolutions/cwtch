# Cwtch

An experimental Javascript UCI chess engine with NNUE evaluation.

Internal data generation and DIY Javascript trainer.

Currently +45 over Lozza with this net:-

```
datagen soft-nodes 6000 hard-nodes 12000 random-moves 10ply first-used 16ply
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
```

See the ```skipP``` function in ```trainer.js``` for how the data is filtered during training:-

https://github.com/op12no2/cwtch/blob/main/trainer.js#L450

If you want to try cwtch, it can be fired up in Node.js like Lozza. For details see the readme in the Lozza 2.5 release.

https://github.com/op12no2/lozza/releases/tag/2.5
