# Source Code for "Scalable Methods for Computing Sharp Extreme Event Probabilities in Infinite-Dimensional Stochastic Systems"

### Problem formulation

The code in this repository can be used to determine the tail probability
$\mathbb{P} \left[ f(X(T), Y(T)) \geq z \right]$
for the two-dimensional stochastic differential equation

$$
\begin{cases}
    \mathrm{d} X = (-X-XY)\mathrm{d} t + \sqrt{\varepsilon}\mathrm{d} B_X\\
    \mathrm{d} Y = (-4Y+X^2)\mathrm{d} t + \tfrac12\sqrt{\varepsilon}\mathrm{d} B_Y
  \end{cases}
$$

with deterministic initial condition $(X(0),Y(0)) = (0,0)$ and observable
$f(x, y) = x + 2 y$. The code contains functions to obtain sampling estimates,
as well as to compute the instanton and the leading-order prefactor using
either a Riccati equation or the dominant eigenvalues of the projected
second variation operator. All details can be found in the paper.

### Standard Parameters

These are the default parameters that were also used in the paper:
* $T = 1$
* $\varepsilon = 0.5$
* $z = 3$
* $n_t = 2000$

### Prerequisites

The python3 scripts just require numpy, scipy and matplotlib.
They were tested under Python 3.6.9, numpy 1.19.5, scipy 1.4.1 and matplotlib 3.3.4.

### How to execute

1. Run
   ```sh
   python3 compute.py
   ```
   to collect the following data for the toy problem:
   * `nPaths = 100` sample paths with $f(X(T), Y(T)) \geq z$ using `getSamplePaths()` via direct sampling
   * the instanton for $z$ using `instanton.searchInstantonViaAugmented()` to compute the rate function
   * the Riccati solution along this instanton using `solveForwardRiccati()` to compute the prefactor
   * the eigenvalues and eigenfunctions of $A_z$ using `scipy.sparse.linalg.eigs` as an alternative to compute the prefactor
   * `nPaths = int(1e5)` sample paths with $f((X_T^\varepsilon - \phi_z(T))/\sqrt{\varepsilon}) < 0.05$ via instanton based importance sampling with `getTransitionPathStatisticsImportanceSampling()`
   
   The data will be stored in a 'data' subdirectory and take up about 30 MB.
   The sampling routines may take some time for the default sample sizes.
   
2. Run
   ```sh
   python3 plot.py
   ```
   afterwards to reproduce figures 1, 2, 3 and 5 from the paper.
