# Pytorch L2RU Architecture: LRU with l2 stability guarantees and prescribed bound

A PyTorch implementation of the L2RU architecture introduced in the paper Free Parametrization of L2-bounded State Space Models. https://arxiv.org/abs/2503.23818. Application in System Identification is included as an example.

## L2RU block
The L2RU block is a discrete-time linear time-invariant system implemented in state-space form as:
```math
\begin{align}
x_{k+1} = Ax_{x} + B u_k\\
y_k = C x_k + D u_k,
\end{align}
```
A parametrization is provided for the matrices ```(A, B, C, D)```, guaranteeing a prescribed l2 bound for the overall SSM.
Moreover, the use of [parallel scan algorithms](https://en.wikipedia.org/wiki/Prefix_sum) makes execution extremely fast on modern hardware in non-core-bound scenarios.

## Deep L2RU Architecture

L2RU units are typically organized in a deep LRU architecture like:

<div align="center">
  <img src="architecture/L2RU.png" alt="Description of image" width="800">
</div>






