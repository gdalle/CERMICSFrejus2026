# Autodiff

## How to differentiate

### Basics

- Definition of derivative
- Motivation
- Differentiable programming

### The four horsemen

- Manual
- Symbolic
- Numeric
- Algorithmic
- Computational graph

## The modes of AD

### Forward mode

- Linear maps
- Chain rule
- Link with matrices
- Examples

### Reverse mode

- Adjoint linear maps
- Reverse chain rule
- Link with transposed matrices
- Examples

### Complexity

- Time complexity
- Space complexity
- Checkpointing

### Full matrices

- Choosing a mode
- Solving linear systems

## Implementation

### Principles

- Non-standard interpretation
- Operator overloading
- Source transformation

### Limitations

- Mutation support
- Control flow

### Challenges

- Non-smooth functions
- Approximations
- Solvers as layers

### Solutions

- Writing rules
- Implicit function theorem for continuous optimization layers
- Smoothing for discrete optimization layers

## Software

### Julia libraries

General autodiff: see <https://juliadiff.org/>

- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl)
- [Zygote.jl](https://github.com/FluxML/Zygote.jl)

Optimization layers:

- [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl)
- [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl)
- [InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)

### Python libraries

General autodiff:

- [JAX](https://github.com/jax-ml/jax)
- [PyTorch](https://github.com/pytorch/pytorch)

Optimization layers:

- [Optax](https://github.com/google-deepmind/optax)
- [torchopt](https://github.com/metaopt/torchopt)
- [cvxpylayers](https://github.com/cvxpy/cvxpylayers)
- [PyEPO](https://github.com/khalil-research/PyEPO)
- [SoftJAX](https://github.com/a-paulus/softjax) / [SoftTorch](https://github.com/a-paulus/softtorch)

## References

Baydin, Atilim Gunes, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. 2018. “Automatic Differentiation in Machine Learning: A Survey.” *Journal of Machine Learning Research* 18 (153): 1–43. <http://jmlr.org/papers/v18/17-468.html>.

Blondel, Mathieu, and Vincent Roulet. 2025. “The Elements of Differentiable Programming.” Pre-published June 24. <https://doi.org/10.48550/arXiv.2403.14606>.

Bradbury, James, Roy Frostig, Peter Hawkins, et al. 2018. *JAX: Composable Transformations of Python+NumPy Programs*. V. 0.3.13. Released. <http://github.com/google/jax>.

Gebremedhin, Assefaw Hadish, Fredrik Manne, and Alex Pothen. 2005. “What Color Is Your Jacobian? Graph Coloring for Computing Derivatives.” *SIAM Review* 47 (4): 629–705. <https://doi.org/10/cmwds4>.

Hückelheim, Jan, Harshitha Menon, William Moses, Bruce Christianson, Paul Hovland, and Laurent Hascoët. 2023. “Understanding Automatic Differentiation Pitfalls.” Pre-published May 12. <https://doi.org/10.48550/arXiv.2305.07546>.

Hückelheim, Jan, Harshitha Menon, William Moses, Bruce Christianson, Paul Hovland, and Laurent Hascoët. 2024. “A Taxonomy of Automatic Differentiation Pitfalls.” *WIREs Data Mining and Knowledge Discovery* 14 (6): e1555. <https://doi.org/10.1002/widm.1555>.

Margossian, Charles C. 2019. “A Review of Automatic Differentiation and Its Efficient Implementation.” *WIREs Data Mining and Knowledge Discovery* 9 (4): e1305. <https://doi.org/10.1002/widm.1305>.

Mohamed, Shakir, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. 2020. “Monte Carlo Gradient Estimation in Machine Learning.” *Journal of Machine Learning Research* 21 (132): 1–62. <http://jmlr.org/papers/v21/19-346.html>.
