---
layout: page
title: Langmuir Blodgett Trough Rehabilitation
description: Fulfilling the shop class requirement
img: assets/img/proj12_lbTrough_small.jpg
importance: 3
category: physics
---

By far the most extensive instrumentation project I undertook, and documented in my MS Physics thesis.

Langmuir Blodgett deposition was pioneered by the work of <a href="https://en.wikipedia.org/wiki/Agnes_Pockels">Agnes Pockels</a> and <a href="https://en.wikipedia.org/wiki/Katharine_Burr_Blodgett">Katharine Blodgett</a>.  The basic idea is to deposit a self-orienting monolayer on top of a polar subphase, then leverage the surface dynamics to transfer the monolayer to a substrate

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj12_meniscus_1a.png" title="Meniscus immersion" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/proj12_meniscus_2a.png" title="Meniscus emmersion" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Meniscus dynamics during the Langmuir Blodgett dipping process.  The angle of contact is shown as a red dashed line; this directly impacts the deposition rate.
</div>

I inherited a KSV Langmuir Blodgett chassis and some microelectronics c. 1991, kitted out for surface tension studies.  The first step to rehabilitation was designing and fabricating a new tray with a reservoir suitable for thin film deposition.  Ebay is an excellent source of raw materials.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj12_trough_fab.JPG" title="Milling the delrin part one" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/proj12_trough_fab_shellCutter.jpg" title="Milling the delrin part two" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/proj12_me_milling.JPG" title="Action shot" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fabricating the new deposition trough, including an action shot of me happily using the mill.  
</div>

After the trough was fabricated, I had to reverse engineer some control software since the original drivers were long gone.  The basic control flow is shown below

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj12_systemArchi_2.png" title="open source software architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

This included assembling and calibrating a desktop dipper from an arduino, stepper motor, and linear actuator.  When finally assembled, the system looked like this.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj12_lbTrough_small.jpg" title="rehabilitated LB trough" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

I then built a giant plastic box to keep debris from falling onto the trough surface, and jammed the whole thing inside a fume hood.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj12_trough_in_hood.jpg" title="trough in its new home" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

I eventually added a surface balance and put the trough chassis in a tray of sand to further damp vibrations which might disturb the thin film formation.  The whole thing works pretty good.  If you want the labview code, shoot me an email.  

Thanks to A. Mosey for training me on the mill, and D. Emerson for his reverse engineering software expertise.
