# Lifelong Learning Algorithms
We provide implementations for three authentic lifelong learning algorithms from the lifelong learning community. They are representative algorithms from the memory-based, regularization-based, and dynamic-architecture-based methods.

### ER (Experience Replay)
(See [ER](https://arxiv.org/abs/1902.10486))

ER saves a small amount of data per task. When the agent learns a new task, ER trains the agent on the combined data consisting of both the prior and the new datasets. It is known as one of the strongest baselines in lifelong learning, but ER suffers from extra memory usage.

### EWC (Elastic Weight Consolidation)
(See [EWC](https://arxiv.org/pdf/1612.00796.pdf))

EWC is one of the first regularization-based methods that poses strong regularization to network parameters that contribute more to prior task learning. The importance of the parameter is estimated by the Fisher information matrix (FIM). In practice, as the number of parameters in a neural network is large, one uses a diagonal approximation of the Fisher information matrix and instead of saving K matrices for K tasks, we can online update a single matrix that is an exponential moving average of the past K diagonal FIMs (See this [paper](https://arxiv.org/abs/1801.10112)).

### PACKNET
(See [PACKNET](https://arxiv.org/abs/1711.05769))

PackNet is one of the earliest works that belong to the dynamic architecture approach. The core idea is to perform iterative pruning. When learning a new task, the agent prunes away less important neurons from the exsiting network, then re-initialize the pruned part and learn only these part of parameters for the new task. By having parameter mask that controls which part of the network is firing in a task, PackNet can achieve very minimum forgetting. But the drawback of this approach are that 1. PackNet needs to know the task ID and 2. PackNet is less effective as the number of tasks grows as the hard masking prevents efficient transfer.
