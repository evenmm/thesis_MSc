# MSc thesis at NTNU, spring 2020
Robustness Analysis of Anqi Wu's Latent Manifold Tuning Model (https://github.com/waq1129/LMT)

Application to head direction data by Peyrache et al. 

Infer head direction:

`python em-algorithm.py Mouse12-120806_stuff_simple_awakedata.mat`

Infer only tuning curves:

`python tuning-curve-inference.py Mouse12-120806_stuff_simple_awakedata.mat`

Robustness evaluation with a peak firing rate of 4 (14 is the index in an array of tuning strengths): 

`python cluster-parallel-robustness-evaluation.py 14`

Example plots:

`python example-plotting.py`

Even Moa Myklebust 2020
