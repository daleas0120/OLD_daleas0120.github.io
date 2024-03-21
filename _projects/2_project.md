---
layout: page
title: Ferroelectric Capacitor
description: Ferroelectric Capacitor Fabrication and Characterization
img: assets/img/proj02_sample_structure.jpg
importance: 3
category: physics
---

A side project was learning how to improve the quality of the ferroelectric thin films we fabricated in our lab using poly(vinylidene fluoride-co-hexafluoropropylene) (PVDF-HFP).  Samples were fabricated with the following structure to avoid edge affects:

<div class="row justify-content-md-center">
    <div class="col col-lg-2">
    </div>
    <div class="col-md-auto">
        {% include figure.html path="assets/img/proj02_sample_structure.jpg" title="Ferroelectric capacitor" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="row justify-content-md-center">
    </div>
</div>
<div class="caption">
    Ferroelectric capacitor structure visualization.  The SiO2 wafer substrate is shown in dark grey, with a bottom electrode formed by a thin films of Cu (orange) capped by an Al thin film (light grey) partially covering the substrate.  The ferroelectric layer is shown in clear, with the PVDF-HFP polymer chains oriented on the substrate .  The top electrode is another Al thin film (light grey) wih a thicker thin film of Au (yellow).
</div>

The differences in samples between those fabricated manually and those from an automated system are visible to the eye.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj02_bad_film.png" title="Sample dipped by hand" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/proj02_lb_dep_sample.jpg" title="Sample dipped by automated Langmuir Blodgett deposition system." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (LEFT) A PVDF-HFP thin film on SiO2 formed by <strong>manual</strong> Langmuir Blodgett deposition.  (RIGHT)  A PVDF-HFP thin film formed by <strong>automated</strong> Langmuir Blodgett deposition.  A slight irridescence is visible; always a good sign.
</div>

A simple Sawyer-Tower circuit can do the trick for hysteresis measurements, but we used Radiant Ferroelectric tester to evaluate the improvement in thin film quality

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj02_60ML_hand.png" title="Sample dipped by hand" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/proj02_80ML_LB.png" title="Sample dipped by automated Langmuir Blodgett deposition system." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (LEFT) Hysteresis characterization of a PVDF-HFP thin film formed by <strong>manual</strong> Langmuir Blodgett deposition.  (RIGHT)  Hysteresis characterization of a PVDF-HFP thin film formed by <strong>automated</strong> Langmuir Blodgett deposition.
</div>

[1]<a href="http://ulib.iupui.edu/cgi-bin/proxy.pl?url=http://search.proquest.com/dissertations-theses/developing-approach-improve-beta-phase-properties/docview/2827705046/se-2?accountid=7398">Dale, A. S. (2020). Developing an Approach to Improve Beta-Phase Properties in Ferroelectric PVDF-HFP Thin Films. Purdue University.</a>

[2]<a href="https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c02209">Mosey, A., Dale, A. S., Hao, G., N’Diaye, A., Dowben, P. A., & Cheng, R. (2020). Quantitative study of the energy changes in voltage-controlled spin crossover molecular thin films. The Journal of Physical Chemistry Letters, 11(19), 8231-8237.</a>