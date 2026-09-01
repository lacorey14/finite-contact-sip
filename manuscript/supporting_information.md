# Supporting Information for

## Finite Electronic Contacts Link Contact-Mediated Relaxation to Scale-Dependent Directionality in Spectral Induced Polarization

**Authors:** Hai Li¹,²; **[COAUTHOR INPUT NEEDED]**

**Corresponding author:** Hai Li (lihai@mail.iggcas.ac.cn)

This Supporting Information contains eight supplementary texts and six tables.
Machine-readable values underlying the main-text figures and the supplementary
tables are included in the planned public archive described in the Open
Research section of the main text.

## Text S1. Coupled weak-form system

Discretization of Equations (1)--(3) of the main text with continuous
first-order tetrahedral basis functions gave the coupled harmonic system

\[
\begin{bmatrix}
\mathbf K_e+i\omega\mathbf C_{ee} & -i\omega\mathbf C_{em}\\
-i\omega\mathbf C_{me} & \mathbf L_G+i\omega\mathbf C_{mm}
\end{bmatrix}
\begin{bmatrix}\mathbf v_e\\\mathbf v_m\end{bmatrix}
=\mathbf b,
\tag{S1}
\]

Here, \(\mathbf K_e\) is the electrolyte conduction matrix. The four
\(\mathbf C\) blocks collect the mineral--water interfacial-capacitance
contributions. The matrix \(\mathbf L_G\) is the weighted Laplacian of the
finite electronic-contact graph. We define the stacked complex unknown as
\(\mathbf z=(\mathbf v_e^\mathsf{T},\mathbf v_m^\mathsf{T})^\mathsf{T}\). Writing
\(\mathbf H\) for the complex matrix in Equation (S1), the real system passed
to the sparse solver was

\[
\begin{bmatrix}\mathbf A&-\mathbf B\\\mathbf B&\mathbf A\end{bmatrix}
\begin{bmatrix}\operatorname{Re}\mathbf z\\\operatorname{Im}\mathbf z\end{bmatrix}
=
\begin{bmatrix}\operatorname{Re}\mathbf b\\\operatorname{Im}\mathbf b\end{bmatrix},
\qquad \mathbf A=\operatorname{Re}\mathbf H,
\quad \mathbf B=\operatorname{Im}\mathbf H,
\tag{S2}
\]

Using the two electrode-flux estimates defined in Section 2.1 of the main
text, we quantified the normalized current-closure mismatch as

\[
\epsilon_I=\frac{|\sigma^*_{L}-\sigma^*_{R}|}
{\max(|\tfrac12(\sigma^*_{L}+\sigma^*_{R})|,10^{-30})}.
\tag{S3}
\]

## Text S2. Parameters, meshes, and solver scope

Table S1 consolidates the continuum parameters and dimensionless controls used
in the full-FEM calculations. The physical model, linear capacitive interface
condition, and excluded processes are defined in Section 2.1 and discussed in
Section 4.4 of the main text.

First-order tetrahedral meshes were generated with Gmsh, with the electrode
faces, insulating exterior faces, and mineral--electrolyte interfaces tagged
separately. The coupled system was assembled in scikit-fem and solved with
`scipy.sparse.linalg.spsolve`. The current project requirements specify minimum
versions of NumPy 1.24, SciPy 1.10, matplotlib 3.7, scikit-fem 10.0, meshio 5.3,
and Gmsh 4.11. Exact package versions will be recorded in a pinned environment
file distributed with the reproducibility archive.

The principal full-FEM models used cubic domains with side lengths of 47, 70,
75, 80, and 80 mm for the two-particle phase diagram, rotated chain, T network,
irregular networks, and eight-particle endpoints, respectively. The
corresponding nominal bulk and mineral-interface characteristic lengths were
6.0/1.2, 8.0/1.3, 8.5/1.4, 9.0/1.5, and 10.0/1.8 mm. Selected endpoints were
recomputed on finer meshes using the characteristic lengths recorded in the
mesh-refinement scripts. External boundary facets were first classified
geometrically as electrode or insulating faces; each mineral-interface facet
was then associated with the nearest particle center.

**Table S1. Fixed continuum parameters and dimensionless controls.**

| Quantity | Symbol | Value | Role |
|---|---:|---:|---|
| Particle radius | \(R\) | 5 mm | Reference particle size |
| Electrolyte conductivity | \(\sigma_0\) | \(0.2\,\mathrm{S\,m^{-1}}\) | Ionic background conduction |
| Specific interfacial capacitance | \(C_0\) | \(0.4\,\mathrm{F\,m^{-2}}\) | Linear mineral--electrolyte charge storage |
| Applied field amplitude | \(E_0\) | \(50\,\mathrm{V\,m^{-1}}\) | Linear harmonic forcing |
| Characteristic frequency | \(f_0\) | 31.83 Hz | \(\sigma_0/(\pi R C_0)\) |
| Characteristic angular frequency | \(\omega_0\) | \(2\pi f_0\) | Reference angular frequency |
| Particle surface area | \(A_p\) | \(4\pi R^2\) | Total interface area of an isolated sphere |
| Contact ratio | \(\gamma\) | \(G_c/(\omega_0 C_0 A_p)\) | Electronic-contact/interfacial competition |
| Frequency ratio | \(f/f_0\) | 0.08--12 (phase diagram); 0.1--10 (directional tests) | Dimensionless forcing frequency |

## Text S3. Finite-contact phase diagram and limiting states

Peak frequency and amplitude were estimated by fitting a quadratic function to
\(|\sigma''|\) as a function of
\(\log_{10}f\) at the sampled maximum and its two adjacent frequency points; the
fitted vertex defined the reported peak. The full width at half maximum was
calculated by linearly interpolating the two half-maximum crossings in
\(\log_{10}f\) and taking their separation in decades. For the finite-contact
states, contact dissipation was evaluated using the expression given in
Section 2.2 of the main text. Table S2 reports the complete descriptor series
underlying main-text Figures 1 and 2.

Recovery of the limiting contact states was evaluated at \(0.3\), 1, and
\(3f_0\). A contact conductance of \(10^{-8}\) S differed from the disconnected
response by less than \(4.7\times10^{-8}\) in relative complex conductivity,
whereas the 20-S contact differed from the ideal-connected limit by no more
than \(1.05\times10^{-4}\). The base 47-mm model, its refined-mesh
counterpart, and the enlarged 65-mm domain preserved the ordering of the
disconnected, finite-contact, and ideal-connected responses. Relative to the
base configuration, the corresponding volume-scaled imaginary conductivities
changed by no more than 4.02%.

Across the phase-diagram, limiting-state, mesh-refinement, and domain-size
calculations, the independently calculated left- and right-electrode
conductivities differed by at most 0.0194%.

**Table S2. Peak descriptors for the two-particle finite-contact series.**

| \(G_c\) (S) | \(\gamma\) | \(f_{\mathrm{peak}}\) (Hz) | \(\lvert\sigma''\rvert_{\mathrm{peak}}\) \((\mathrm{S\,m^{-1}})\) | FWHM (decades) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 39.674 | 0.004945 | 1.153 |
| 0.0001 | 0.00398 | 39.674 | 0.004937 | 1.154 |
| 0.0005 | 0.01989 | 39.629 | 0.004910 | 1.157 |
| 0.002 | 0.07958 | 38.937 | 0.004869 | 1.223 |
| 0.01 | 0.39789 | 25.759 | 0.005677 | 1.669 |
| 0.02 | 0.79577 | 16.848 | 0.007775 | 1.428 |
| 0.05 | 1.98944 | 19.958 | 0.010696 | 1.247 |
| 0.2 | 7.95775 | 23.464 | 0.013338 | 1.176 |
| 1 | 39.78874 | 24.893 | 0.014289 | 1.166 |
| 5 | 198.94368 | 25.229 | 0.014497 | 1.164 |
| \(\infty\) | \(\infty\) | 25.316 | 0.014550 | 1.164 |

*Note.* \(G_c=0\) denotes the disconnected state, whereas
\(G_c\rightarrow\infty\) denotes the ideal-connected limit.

## Text S4. Explicit conductive-neck mapping

The isolated conductive-neck mapping introduced in Section 2.2 of the main
text was implemented by prescribing a unit potential difference between the
two end faces and an insulating condition on the curved sidewall. These
conditions isolate the static end-to-end conductance defined by Equation (6)
of the main text.

Seven neck geometries spanning \(L_n=0.25\)--1.00 mm and
\(r_n=0.05\)--0.50 mm were evaluated at \(\sigma_n=1\), 100, and
\(10^4\ \mathrm{S\,m^{-1}}\), giving 21 three-dimensional calculations. Across
this ensemble, the energy-based FEM conductance agreed with Equation (6) within
2.01%, and the normalized mismatch between the independently calculated
terminal currents remained below \(2.5\times10^{-14}\). Table S3 gives the
four-level refinement sequence for the representative neck geometry.

**Table S3. Mesh convergence for the representative explicit conductive neck.**

| Neck resolution, \(r_n/h\) | Mesh nodes | Tetrahedral elements | \(G_c^{\mathrm{FEM}}\) (S) | Relative error (%) |
|---:|---:|---:|---:|---:|
| 2.5 | 318 | 1,058 | 0.019247 | 1.977 |
| 4.0 | 997 | 3,956 | 0.019491 | 0.734 |
| 6.0 | 2,831 | 13,110 | 0.019568 | 0.340 |
| 8.0 | 6,110 | 30,178 | 0.019598 | 0.190 |

*Note.* The analytical reference conductance is \(G_c=0.019634954\) S for
\(L_n=1\) mm, \(r_n=0.25\) mm, and
\(\sigma_n=100\ \mathrm{S\,m^{-1}}\). The characteristic mesh length is
denoted by \(h\).

## Text S5. Three-particle topology and modal comparison

The eight three-particle contact states defined in Section 2.3 of the main text
were also used to evaluate the transferability of a reduced graph-based
spectral predictor.

To test whether contact-graph modes alone could reproduce the full-FEM
conductivity increments, we evaluated the reduced predictor

\[
\Delta\sigma^*(\omega)=K(\omega)\,
\mathbf{b}^{\mathrm{T}}\mathbf{L}
\left(\mathbf{L}+i\omega C_n\mathbf{I}\right)^{-1}\mathbf{b},
\]

where \(\mathbf{L}\) is the weighted contact-conductance Laplacian,
\(\mathbf{b}\) is the mean-free background potential sampled at the particle
centers, and \(C_n\) is a single effective nodal capacitance. The
frequency-dependent transfer function \(K(\omega)\) was fixed from the
strong-chain response, whereas \(C_n\) was selected by minimizing the
aggregate prediction error over the other four finite-contact topologies.
Even under this globally optimized choice, the relative complex \(L_2\)
errors were 60.5% for the transition chain, 63.1% for the weak chain, 27.4%
for the bottleneck chain, and 55.7% for the single-edge case.
Thus, the contact graph and its field projections capture part of the network
structure, but their effects cannot be transferred across topologies through
a single nodal capacitance and a strong-chain calibration. The surrounding
electrolyte and the spatial arrangement of the particles remain essential to
the continuum response.

## Text S6. Full-FEM calculation inventory

The directional tests and their results are described in Sections 2.3 and 3.3
of the main text. Table S4 consolidates the number of geometries, frequencies,
orientations, and full-FEM solutions used in the contact-relaxation,
directional, and continuum-endpoint calculations.

**Table S4. Full-FEM calculation inventory.**

| Calculation | Geometries/networks | Frequencies | Directions/angles | Full FEM solutions | Role |
|---|---:|---:|---:|---:|---|
| Two-particle phase diagram | 1 | 15 | 1 | 165 | Contact-mediated relaxation |
| Explicit neck | 7 | Static | Terminal | 21 | Conductance mapping |
| Straight-chain rotation | 1 | 5 | 5 | 66 | Selection-law test; 36 endpoint solves reused |
| Branched-network rotation | 1 | 5 | 5 | 50 | Held-out directional prediction |
| Irregular contact deletion | 6 | 3 | 2 | 72 | Directional generalization |
| Eight-particle endpoint | 2 fabrics | 3 | 3 | 36 | Full-FEM upscaling endpoint |

## Text S7. Random-generation and ensemble records

The directional graph screen described in Section 2.3 of the main text used
fixed master seed 20260813. For each realization, 12 node coordinates were
drawn independently from a uniform distribution on \([-1,1]^2\). A connected
backbone was constructed by applying a randomized Euclidean minimum-spanning
procedure in which candidate edge lengths were multiplied by independent
\(\operatorname{lognormal}(0,0.35)\) factors. Occupied-edge conductances were
then drawn from \(\operatorname{lognormal}(0,1)\), and additional local loops
were admitted with probability 0.22 from the 36 shortest candidate pairs not
already in the backbone. The archived output stores the seed, per-network
summary quantities, and the directional screening values for all 33,302
edges; the node coordinates and graph realizations are regenerated
deterministically from the recorded seed and algorithm.

The representative-volume design and response tensor are defined in Section
2.4 of the main text. Network realization \(d=(L,p,a,r)\), where \(r\) is the
replicate index, used the deterministic seed

\[
s_d=73129+1{,}000{,}003L+10{,}007r
+101\,\operatorname{round}(1000p)
+\operatorname{round}(1000a).
\]

Each nearest-neighbor bond was retained by an independent Bernoulli draw using
its direction-dependent occupation probability, after which its conductance
was drawn independently from \(\ln G\sim\mathcal N(0,0.65^2)\). The
design-specific seeds make each network realization independent of execution
order. The archived output stores the seed and calculated network diagnostics
for every design and replicate.

Table S5 provides the occupancy-specific scaling exponents and the
corresponding 4,096-node directional descriptors underlying main-text Figure
7. The fabric-dependent reduced-network and full-FEM results are reported in
main-text Figure 8.

**Table S5. Occupancy-specific RVE scaling descriptors.**

| Mean bond occupancy | Fitted exponent | Median directional variability | Median weakest/strongest response |
|---:|---:|---:|---:|
| 0.18 | -0.572 | 0.0276 | 0.9352 |
| 0.25 | -0.477 | 0.0286 | 0.9372 |
| 0.32 | -0.539 | 0.0222 | 0.9517 |
| 0.45 | -0.449 | 0.0189 | 0.9556 |
| 0.65 | -0.490 | 0.0154 | 0.9654 |

*Note.* The fitted exponent was obtained across the complete 64--4,096-node
size series. Directional variability and weakest-to-strongest response ratios
are reported at 4,096 nodes.

## Text S8. Verification summary and reproducibility

Table S6 consolidates the numerical verification metrics used to assess
limiting-state recovery, discretization sensitivity, directional prediction,
geometric similarity, and current closure.

**Table S6. Numerical verification summary.**

| Check | Decisive metric | Outcome |
|---|---|---|
| Contact limits | Tiny-contact and strong-contact relative errors | \(<4.7\times10^{-8}\); \(<1.05\times10^{-4}\) |
| Explicit-neck convergence | Finest conductance error | 0.190% |
| Straight-chain prediction | Median/maximum complex error | 0.60% / 0.70% |
| Branched held-out prediction | Median/maximum complex error | 1.63% / 1.78% |
| Branched mesh refinement | Maximum contact-increment change | 2.95% |
| Irregular endpoints | Fine-mesh suppression factors | 22.27 and 500.14 |
| Geometric similarity | Maximum aligned-response change | 1.36% |
| RVE full-FEM endpoint | Mesh change in \(z/xy\) ratio | 0.26% |
| Electrode closure | Maximum in selected refined endpoints | 0.0362% |

The reproducibility archive groups scripts by computational role. Full-FEM
scripts regenerate the underlying numerical solutions and may require
substantial computational time. Analysis and plotting scripts can instead
operate on the archived raw outputs. The repository README records the
corresponding entry points, dependencies, expected input files, and the
distinction between full recomputation and regeneration from frozen results.
