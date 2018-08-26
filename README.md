# EvoRBC
EvoRBC implementation for OpenAI Gym environments

## References
1. [Illuminating search spaces by mapping elites (MAP-Elites)](https://arxiv.org/pdf/1504.04909.pdf)
2. [Evolution of Repertoire-Based Control for Robots With Complex Locomotor Systems (EVO-RBC)](https://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/7964759/)
3. [Evolving Neural Networks through Augmenting Topologies (NEAT)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
4. [Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds](http://nmbl.stanford.edu/publications/pdf/Hamner2012.pdf)

Ant observations/todos after 3rd repo
1. some legs still don't move in a few gaits, maybe increase control freq range but then can't stay still.... (maybe penalise this.. difference in leg speed)
2. the topmost speed isn't fast enough could increase the range
3. evolved behaviors are good.
4. since things have saturated and mutation isn't leading anywhere great, should start including crossover

## EvoRBC high level architechture
![image](https://user-images.githubusercontent.com/27682820/43711887-db0a86ba-9991-11e8-97f2-a65152e7a6e4.png)

## Control function for joints used for hexapod
In the ant implementation, frequency is also taken as variable parameter since it helps in varying velocity
![image](https://user-images.githubusercontent.com/27682820/43755663-8ebf73c2-9a2e-11e8-9c1b-a75228f00642.png)

## MAP Elites algorithm
![image](https://user-images.githubusercontent.com/27682820/43864271-dfc186ee-9b7c-11e8-95e9-0b5e71bf1d32.png)

## Unified QD algorithm
![image](https://user-images.githubusercontent.com/27682820/43869088-6a746f8e-9b8e-11e8-957a-34453b922ce5.png)

## Regarding cross over in map elites according to the paper
![image](https://user-images.githubusercontent.com/27682820/43890083-6a89cf1e-9be3-11e8-9f75-abb636838c3f.png)

## Activation and excitation for muscles - human
![image](https://user-images.githubusercontent.com/14030793/43922915-c8b9fa3c-9c3d-11e8-852e-98abb60eaa2e.png)

## Muscles activations during one gait cycle
![image](https://user-images.githubusercontent.com/27682820/44631150-e145e980-a985-11e8-82d8-2e9399df29a5.png)

![image](https://user-images.githubusercontent.com/27682820/44631310-bfe5fd00-a987-11e8-8d54-0620d1fc7c84.png)


![image](https://user-images.githubusercontent.com/27682820/44631163-ffabe500-a985-11e8-9e23-bb13b00e99cb.png)
