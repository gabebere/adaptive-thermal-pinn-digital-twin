# Analytical 2-D Heat-Conduction Reference Study

## 1. Objective

This section recreates the analytical study presented by Hsu, Tu, and Chang,
*An Analytic Solution for 2D Heat Conduction Problems with General Dirichlet
Boundary Conditions*, Axioms 12(5), 416 (2023). The implementation provides an
independent analytical reference for checking thermal-learning models before
they are applied to sparse-sensor reconstruction or online adaptation.

The supplied program evaluates the authors' shifting-function and
eigenfunction series for their parabolic-boundary example. It is not a
Crank-Nicolson, finite-difference, or PINN solution.

## 2. Mathematical Problem

The dimensionless temperature satisfies

\[
\frac{\partial\theta}{\partial\tau}
=L_r^2\frac{\partial^2\theta}{\partial X^2}
+\frac{\partial^2\theta}{\partial Y^2},
\qquad 0<X<1,\quad 0<Y<1,
\]

where \(L_r=L_y/L_x\). The published numerical study uses a square domain,
so \(L_r=1\), and prescribes a parabolic spatial profile on every edge:

\[
\begin{aligned}
\theta(0,Y,\tau)&=(Y-Y^2)e^{-d_1\tau},\\
\theta(1,Y,\tau)&=(Y-Y^2)e^{-d_2\tau},\\
\theta(X,0,\tau)&=(X-X^2)e^{-d_3\tau},\\
\theta(X,1,\tau)&=(X-X^2)e^{-d_4\tau}.
\end{aligned}
\]

The initial condition is

\[
\theta(X,Y,0)=(X-X^2)+(Y-Y^2).
\]

The parabolic factors vanish at the four corners. This is important because
the method in the 2023 paper requires zero corner values.

## 3. Analytical Solution Method

The paper divides the temperature into two subsystems by superposition. One
subsystem carries the left and right boundary temperatures; the other carries
the bottom and top boundary temperatures. Linear shifting functions convert
the nonhomogeneous edge conditions into homogeneous conditions, after which
each subsystem is expanded in sine eigenfunctions.

The implementation evaluates Equations (108), (110), and (111) of the paper.
The infinite double series is truncated using equal limits
\(m=n=N\), with

\[
N\in\{1,3,5,10,20\}.
\]

This reproduces the convergence study performed by the authors at the center
of the plate, \((X,Y)=(0.5,0.5)\).

## 4. Boundary-Condition Cases

Three decay-rate cases are included:

| Case | \((d_1,d_2,d_3,d_4)\) | Description |
|---|---:|---|
| Table 1 | \((1,1,1,1)\) | Equal exponential decay on all four edges |
| Table 2 | \((1,1,2,2)\) | Faster stated decay on the bottom and top edges |
| Table 3 | \((1,2,3,4)\) | A different stated decay rate on every edge |

Every generated CSV repeats the four complete boundary-condition expressions
and the initial condition. The data therefore remain interpretable without
referring back to the source code.

## 5. Reproducibility

Requirements are Python 3.10+ and NumPy. From the repository root, run:

```bash
python Literature_results/mdpi_416_series.py
```

The program recreates all files in `Literature_results/mdpi_416_tables/`. For each
boundary-condition case it produces:

- a `study_times` CSV at the eight times printed in the paper,
  \(\tau=0,0.1,0.2,0.4,0.6,0.8,1.0,1.2\); and
- a `40_times` CSV at exactly 40 equally spaced points over
  \(0\leq\tau\leq1.2\), including both endpoints.

The original-time CSVs contain equation-derived results and the values printed
in the publication side by side. The extended CSVs contain equation-derived
values because the article does not publish reference values at those times.

## 6. Results and Verification

For Table 1, direct evaluation of the published series recreates the reported
center temperatures and convergence behavior to the precision shown in the
article. For example, at \(\tau=0.1\), the calculated center temperatures for
\(N=1,3,5,10,20\) are approximately

\[
(0.22873,\;0.24557,\;0.24245,\;0.24313,\;0.24336),
\]

compared with the published values

\[
(0.229,\;0.246,\;0.243,\;0.243,\;0.243).
\]

The result confirms the rapid series convergence reported by the authors;
approximately 5-10 terms are sufficient for the displayed precision in this
case.

## 7. Reproducibility Limitation

The values printed in Tables 2 and 3 are retained exactly in the output files,
but they are not reproduced by Equations (108)-(111) when the decay constants
stated in the corresponding captions are used. The calculated and published
columns are intentionally kept separate rather than silently replacing one
with the other. This indicates an internal inconsistency among the printed
equations, parameter descriptions, and tabulated values. Consequently, Table 1
is the fully verified benchmark, while Tables 2 and 3 should be treated as
reported reference data with an unresolved publication-level discrepancy.

## 8. Files

- `mdpi_416_series.py`: self-contained analytical series evaluator and table generator.
- `mdpi_416_tables/table_*_study_times.csv`: original study times, calculated values, and exact published values.
- `mdpi_416_tables/table_*_40_times.csv`: 40-time extensions calculated from the printed equations.
- `mdpi_416_tables/README.txt`: compact description of the generated data.
- `expanded_analytical_results/`: long-horizon and combined-boundary analytical
  reference artifacts used by the validated adaptive workflow.

## Reference

H.-P. Hsu, T.-W. Tu, and J.-R. Chang, "An Analytic Solution for 2D Heat
Conduction Problems with General Dirichlet Boundary Conditions," *Axioms*,
vol. 12, no. 5, article 416, 2023. DOI: 10.3390/axioms12050416.
