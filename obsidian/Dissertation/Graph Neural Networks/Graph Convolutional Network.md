A **Graph Convolutional Network** is a popular architecture of [[Graph Neural Network|GNNs]] praised by its simplicity and effectiveness in a variety of tasks. In this model, the node representations in each layer are updated according to the following convolutional operation:

![[gcn_math.png]]

$Ã = A + I$ - Adjacency Matrix with self-connections
$I \in \mathbb{R}^{N \times N}$ - Identity Matrix
$\tilde{D}$ - Diagonal Matrix,  with $\tilde{D}_{ii} = \sum_j Ã_{ij}$
$\sigma$ - Activation Function
$W^k \in \mathbb{R}^{F \times F'}$ - Laywise linear transformation matrix, that is trained during optimization ($F$ and $F'$ are the dimensions of node representations in the $k$-th and $(k + 1)$ layer, respectively)
 
Where  is the adjacency matrix of the given undirected graph $G$ with self-connections, enabling the incorporation of a node's own features when updating the representations. $I \in \mathbb{R}^{N \times N}$ represents the identity matrix, $\tilde{D}$ is a diagonal matrix with $\tilde{D}_{ii} = \sum_j Ã_{ij}$  and $\sigma$ is an activation function such as *ReLU* and *Tanh*. $W^k \in \mathbb{R}^{F \times F'}$ is a laywise linear transformation matrix, that is trained during optimization ($F$ and $F'$ are the dimensions of node representations in the $k$-th and $(k + 1)$ layer, respectively).
The previous equation can be dissected further to understand the *AGGREGATE* and *COMBINE* function definitions in a GCN. For a node $i$, the representation updating equation can be reformulated as:

![[gcn_math_plus.png]]

In the second equation the *AGGREGATE* function can be observed as the weighted average of the neighbour node representations. The weight of neighbour $j$ is defined by the weight of the edge $(i,j)$, more concretely, $A_{ij}$ normalized by the degrees of the two nodes. The *COMBINE* function consists on the summation of the aggregated information and the node representation itself, where the representation is normalized by its own degree.

## Spectral Graph Convolutions

Regarding the connection between GCNs an spectral filters defined on graphs, spectral convolutions can be defined as the multiplication of a node-wise signal $x \in \mathbb{R}^N$ with a convolutional filter $g_\theta = diag(\theta)$ in the *Fourier domain*, formally:

$$g_\theta \star \text{x} = U_{g_\theta} U^T \text{x}$$ 
$\theta \in \mathbb{R}^N$ - Filter parameter
U - Matrix of eigenvectors of the normalized graph Laplacian Matrix $L = I_N - D^{-\frac{1}{2}} AD^{-\frac{1}{2}}$

$U$ consists of the matrix of the eigenvector of the normalized graph Laplacian matrix $L = I_N - D^{-\frac{1}{2}} AD^{-\frac{1}{2}}$. $L = U \Lambda U^T$ with $\Lambda$ serving as the diagonal matrix of eigenvalues and $U^T \text{x}$ is the graph Fourier transform of the input signal $\text{x}$. In a practical context, $g_\theta$ is understood as the function of eigenvalues of the normalized graph Laplacian matrix $L$, that is $g^\theta(\Lambda)$. Computing this is a problem of quadratic complexity to the number of nodes $N$, something that can be circumvented by approximating $g_\theta (\Lambda)$ with a truncated expansion of Chebyshev polynomials $T_k(x)$ up to $K$-th order:

$$ g_{\theta'}(\Lambda) = \sum^K_{k=0} \theta'_k T_k(\tilde{\Lambda})$$
$\tilde{\Lambda} = \frac{2}{\lambda_\text{max}} \Lambda - \text{I}$  
$\lambda_\text{max}$ - Largest eigenvalue of $L$.
$\theta' \in \mathbb{R}^N$ - Vector of Chebyshev coefficients
$T_k(x)$ - Chebyshev polynomials
$T_k(x) = 2 x T_{k - 1} (x) - T_{k - 2}(x)$ with $T_0(x) = 1$ and $T_1(x) = x$ 

By combining this with the previous equation, the first can be reformulated as:
$$g_\theta \star \text{x} = \sum^K_{k=0} \theta'_k T_k(\tilde{L}) \text{x}$$
 $\tilde{L} = \frac{2}{\lambda_\text{max}} L - I$ 

From this equation, we can see that each node depends only on the information inside the $K$-th order neighbourhood and with this reformulation, the computation of the equation is reduced to $O(|\xi|)$, linear to the number of edges $\xi$ in the original graph $G$.

To build a neural network with graph convolutions, it's sufficient to stack multiple layers defined according to the previous equation, each followed by a nonlinear transformation. However, the authors of GCNs proposed, instead of limiting to the explicit parametrization by the Chebyshev polynomials, to limit the convolution number to $K = 1$ at each layer. This way each level only defines a linear function over the Laplacian Matrix $L$, maintaining the possibility of handling complex convolution filter functions on graphs by stacking multiple layers. This means the model can alleviate the overfitting of local neighbourhood structures for graphs whose node degree distribution has a high variance.

At each layer, we can further consider $\lambda_\text{max} \approx 2$, which could be accommodated by the neural network parameters during training. With this simplifications, we have 
$$g_{\theta'} \star \text{x} \approx \theta'_0 \text{x} + \theta'_1 \text{x} (L - I_N) \text{x} = \theta'_0 \text{x} - \theta'_1 D^{-\frac{1}{2}} AD^{-\frac{1}{2}}$$ $\theta'_0$ and $\theta'_1$ - Free parameters that can be shared over the entire graph

The number of parameters can, in practice, be further reduced, minimising overfitting and, furthermore, minimising the number of operations per layer as well:
$$g_\theta \star \text{x} \approx \theta (I + D^{-\frac{1}{2}} AD^{-\frac{1}{2}}) \text{x}$$
$\theta = \theta'_0 = - \theta'_1$

One potential problem is the $I_N + D^{-\frac{1}{2}} AD^{-\frac{1}{2}}$ matrix whose eigenvalues fall in the $[0,2]$ interval. In a deep GCN, the repeated utilization of the above function often leads to exploding or vanishing gradient, translating into numerical instabilities. In this context, the matrix can be further renormalized by converting $I + D^{-\frac{1}{2}} AD^{-\frac{1}{2}}$ into $\tilde{D}^{-\frac{1}{2}} \tilde{A}\tilde{D}^{-\frac{1}{2}}$. 
In this case, only the case where there is one feature channel and one filter is considered. This can be generalized to an input signal with $C$ channels $X \in \mathbb{R}^{N \times C}$ and $F$ filters (or hidden units):
$$ H = \tilde{D}^{-\frac{1}{2}} \tilde{A}\tilde{D}^{-\frac{1}{2}}XW$$
$W \in \mathbb{R}^{C \times F}$ - Matrix of filter parameters
$H$ - Convolved Signal Matrix


[1]
