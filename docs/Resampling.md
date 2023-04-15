# Resampling

## Sinc interpolation

The `sample_quality` settings from 3 to 10 are interpolations based on the windowed sinc method.

https://ccrma.stanford.edu/~jos/lumped/Windowed_Sinc_Interpolation.html

In the RGC sfz software, we have a number of quality settings available from a drop-down menu.  
These sinc settings are labelled: 08 12 16 24 36 48 60 72  
One can easily guess that these designate a number of points. The numbers are multiples of 4, which makes them practical for SIMD processing.

For implementation purposes, we want to precompute the sinc function in a large table.
A window must be applied to the windowed sinc, in order to squash the edges.
A Kaiser window is most appropriate, which permits to keep aliasing ripples under a determined amplitude threshold. It has a parameter `Beta`, permitting to establish a compromise: the higher `Beta`, the lower is the alias magnitude, but the selectivity of the "brick wall" filter worsens near the cutoff point.

At first it seems an alright choice of `Beta` may be from about 6 to 10, as table size increases from 8 to 72.


[Tool](https://gist.github.com/jpcima/f446ae9862965ee77b5ffa8e80882c21) for plotting frequency responses of sinc interpolators dynamically, when size and `Beta` are varied, based on the `deip.pdf` paper.