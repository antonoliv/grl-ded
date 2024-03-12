## State

$$
S_{t1} = \begin{bmatrix}
				P_{G_1} & P_{G_2} & \dots & P_{G_J} \\
				q_{G_1} & q_{G_2} & \dots & q_{G_J} \\
				v_{G_1} & v_{G_2} & \dots & v_{G_J} \\
			\end{bmatrix}
$$
$P_{G_j}$ - active generation
$q_{G_j}$ - reactive generation
$v_{G_j}$ - voltage 
$F_n$ - 1 if transmission line is disconnected 

$$
S_{t2} = \begin{bmatrix}
				p_{L_1} & p_{L_2} & \dots & p_{L_K} \\
				q_{L_1} & q_{L_2} & \dots & q_{L_K} \\
				v_{L_1} & v_{L_2} & \dots & v_{L_K} \\
			\end{bmatrix}
$$

$$
S_{t3} = \begin{bmatrix}
				F_{1} & F_{2} & \dots & F_{N} \\
				rho_{1} & rho_{2} & \dots & rho_{N} \\
			\end{bmatrix}
$$