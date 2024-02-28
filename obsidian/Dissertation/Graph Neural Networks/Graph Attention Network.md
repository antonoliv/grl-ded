**Graph Attention Networks** or GAT [1] is another type of [[Graph Neural Network|GNNs]] that focus on leveraging an attention mechanism to learn the importance of a node's neighbours. In contrast, the GCN that uses edge weight as importance, a value that may not always represent the true strength between two nodes.

## Graph Attention Layer

The Graph Attention layer defines the process of transferring the hidden node representations at layer $k - 1$ to the next node presentations at $k$. For assuring that sufficient expressive power is attained in order to allow the transformation of the lower-level node representations to higher-level ones, a linear transformation $W \in \mathbb{R}^{F \times F'}$ is applied to every node, followed by the self-attention mechanism, which measures the attention coefficients for any pair of nodes through a shared attentional mechanism $a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$. In this context, relationship strength $e_{ij}$ between two nodes $i$ and $j$ can be calculated by:
$$ e_{ij} = a(W H^{k - 1}_i, W H^{k - 1}_j) $$
$H^{k - 1}_i \in \mathbb{R}^{N \times F'}$ - Column-wise vector representation of node $i$ at layer $k - 1$ ($N$ is the number of nodes and $F$ the number of features per node)
$W \in \mathbb{R}^{F \times F'}$ - Shared linear transformation
$a: \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$ - Attentional Mechanism
$e_{ij}$ - Relationship Strength between nodes $i$ and $j$ 

Theoretically, each node can allowed to attend every other node on the graph, although it would ignore the graph topological information in the process. A more reasonable solution is to only attend nodes in the neighbourhood. In practice, only first-order node neighbours are used, including the node itself, and to make the attention coefficients comparable across the various nodes, they're are normalized with a *softmax* function:
$$ \alpha_{ij} = \text{softmax}_j(\{e_{ij}\}) = \frac{exp(e_{ij})}{\sum_{l \in N(i)} exp(e_{il})}$$
Fundamentally, $\alpha_{ij}$ defines a multinomial distribution over the neighbours of node $i$, which can also be interpreted as a transition probability from node $i$ to each node in its neighbourhood.
In the original work [1], the attention mechanism is defined as a single-layer [[Feedforward Neural Network|feedfoward neural network]] that includes a linear transformation with weigh vector $W_2 \in \mathbb{R}^{1 \times 2 F'}$ and a [[Leaky Rectified Linear Unit|LeakyReLU]] nonlinear activation function with a negative input slope $\alpha = 0.2$. More formally, the attention coefficients are calculated as follows:
$$\alpha_{ij} = \frac{ \text{exp}( \text{LeakyReLU}( W_2 [W H^{k - 1}_i || W H^{k - 1}_j]))}{ \sum_{l \in N(i)} \text{exp}( \text{LeakyReLU}( W_2 [W H^{k - 1}_i || W H^{k - 1}_l])) }$$
$||$ - Vector concatenation operation

The novel node representation is a linear composition of the neighbouring representations with weights determined by the attention coefficients, formally:
$$ H^k_i = \sigma(\sum_{j \in N(i)} \alpha_{ij} W H^{k - 1}_j) $$

## Multi-head Attention

Multi-head attention can be used in place of self-attention, determining a different similarity function over the nodes. A independent node representation can be obtained for each attention head according to the equation bellow. The final representation is a concatenation of the node representations learned by different heads, formally:
$$ H^k_i = \Big\Vert^T_{t=1} \sigma(\sum_{j \in N(i)} \alpha^t_{ij} W^t H^{k-1}_j)$$
T - Number of attention heads
$\alpha^t_{ij}$ - attention coefficient computed from the $t$-th attention head
$W^t$ - Linear transformation matrix of the $t$-th attention head

Lastly, the author also mentions that other pooling techniques can be used in the final layer for combining the node representations from different heads, for example the average node representations from different attention heads
$$ H^k_i = \sigma(\frac{1}{T} \sum^T_{t = 1} \sum_{j \in N(i)} \alpha^t_{ij} W^t H^{k-1}_j)$$
 