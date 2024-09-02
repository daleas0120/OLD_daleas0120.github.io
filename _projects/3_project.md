---
layout: page
title: Ising Model Simulations
description: 3D heterostructures and their activation energies
img: assets/gif/proj_3_3DMCIMS.gif
importance: 1
category: physics
---

Spin crossover systems have been successfully described[1] using an Ising-like Hamiltonian of the form

<center>$H = -J \sum_{\langle N \rangle}\sigma_i \sigma_j + \mu \sum_{i} \sigma_i$</center>

where $-J$ represents the interaction strength between the $i$th and $j$th site, $N$ represents the total number of sites, and $\m$ represents the mean field.  When combined with the Metropolis Hastings algorithm [2], simulations of how the system evolves can reveal some fun behaviors.

We can represent the mean field of the bistable spin crossover molecule as

<center>$\mu = \frac{\Delta}{2} - \frac{k_B T}{2} ln(g)$</center>

where $\Delta$ is the energy gap between the high-spin and low-spin state of the molecule, $T$ is the thermodynamic temperature of the system, and $ln(g)$ is the degeneracy ratio between the high-spin and low-spin states.

Here, a 3D lattice was simulated using fixed, non-interacting boundary conditions on the top and bottom of the cell, and periodic boundary conditions along the remaining dimensions.  This represents a spin crossover thin film, where the top and bottom of thin film interface has pinned sites which are unable to update.

<center>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/gif/proj_3_3DMCIMS.gif" title="Simulation of 3D Ising Lattice" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Simulation of 3D Ising Lattice.  The top and bottom of the simulation cell have non-interacting sites, with the periodic boundary conditions along the remaining edges. Results presented at 2021 APS April meeting [3]. 
</div>
</center>

The compact minority-state domain is observed to persist and permeate the simulation cell due to the nearest-neighbors interaction[3].  This agrees with the minority compact state domains observed in XPS data for a spin crossover molecule [4]

The effect of the non-interacting sites can be probed by introducing a simulation where the non-interacting sites are replaced with sites that have a fixed value and are not free to update.

<center>
<div class="row>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj03_pinned_layers.png" title="Adjusting the interactions of pinned sites.  Presented at 2021 APS April meeting [3]." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</center>

Repeating the simulation, we can see what happens when the fixed value sites take a value of $J=1$ (only $J=-1$ sites are shown)

<center>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/gif/proj03_MCIMS_up.gif" title="Simulation of 3D Ising Lattice" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<!-- <div class="caption">
    Simulation of 3D Ising Lattice.  The top and bottom of the simulation cell have non-interacting sites, with the periodic boundary conditions along the remaining edges. Results presented at 2021 APS April meeting [3]. 
</div> -->
</center>

and when the fixed value sites take a value of $J=-1$ (only $J=1$ sites are shown)

<center>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/gif/proj03_MCIMS_down.gif" title="Simulation of 3D Ising Lattice" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</center>

Plotting this as a phase-transition diagram, we can see that the transition point from low-spin dominated to high-spin dominated state occurs at a different temperature depending on whether pinned sites are pinned to $J=1$ "FE pinned up", pinned to $J=-1$ "FE pinned down", or non interacting.  The change in temperature indicates a different value for the energy gap $\Delta$.

<center>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj03_pinned_layers.png " title="Simulation of 3D Ising Lattice" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
</center>

In a practical sense, this kind of pinning can be engineered into the system using different fabrication methods, or through heterostructure design[5].

[1]<a href="https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.054119">Two-dimensional Ising-like model with specific edge effects for spin-crossover nanoparticles: A Monte Carlo study</a>

[2]<a href="https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm">Wikipedia: Metropolis Hastings Algorithm</a>

[3]<a href="https://meetings.aps.org/Meeting/APR21/Session/Q08.8">Determination of High-Spin to Low-Spin Phase Transition of Organic Spintronic Device by Monte Carlo Simulation of 3D Ising-like Model </a>

[4]<a href="https://iopscience.iop.org/article/10.1088/1361-648X/ac6cbc/meta">Intermolecular interaction and cooperativity in an Fe(II) spin crossover molecular thin film system</a>

[5]<a href="https://doi.org/10.1021/acs.jpclett.0c02209">Quantitative Study of the Energy Changes in Voltage-Controlled Spin Crossover Molecular Thin Films
</a>