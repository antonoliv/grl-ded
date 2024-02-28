In the present, deep learning and [[Artificial Neural Network|ANNs]] have become on the most prominent approaches in Artificial Intelligence research. Approaches such as recurrent neural networks (ref) and convolutional networks (ref) have achieved remarkable results in on Euclidean data, such as images or sequence data, such as text and signals. Furthermore, techniques regarding deep learning applied to graphs have also experienced rising popularity among the research community, more specifically **Graph Neural Networks** that became the most successful learning models for graph-related tasks across an extensive number of application domains, in a more fundamental sense [[Graph Representation Learning]].



The main objective of GNNs is to iteratively update node representations by representations from their neighbourhood. Starting at the first representation $H^0 = X$, each layer encompasses two important functions:
* **Aggregate**, in each node, the information from their neighbours
* **Combine** the aggregated information with the current node representations

The general framework of GNNs can be defined mathematically as:

$\text{Initialization: } H^0 = X$
$\text{For } k = 1, 2, \dots, K$
$$ a^k_v = \text{AGGREGATE}^k\{H^{k-1}_u : u \in N(v)\}$$$$ H^k_v = \text{COMBINE}^k\{H^{k-1}_u, a^k_v\}$$



![[gf_gnn_math.png]]

Where $N(v)$ is the set of neighbours for the $v$-th node. The node representations $H^K$ in the last layer can be treated as the final node representations. The node representations, consequently, can be used for other downstream tasks.