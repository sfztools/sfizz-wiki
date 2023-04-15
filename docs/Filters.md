# Filters

The filter design of sfizz is made after the models listed in SFZ specification.
This design is compared against other established sampler software for verification.

Filter reference:
- https://sfzformat.com/opcodes/fil_type
- https://sfzformat.com/opcodes/fil2_type

## Comparison protocol

For comparison, one can implement a SFZ using a `*noise` generator input, and observe the spectrum of the output.

For an equivalence to this noise generator, one can compile a white noise generator of 1/4 amplitude.
```
import("stdfaust.lib");
process = no.noise : *(0.25);
```

SFZ Example
```
<region>
sample=*noise
lokey=0
hikey=127
cutoff=1000.0
fil_type=lpf_2p
fil_gain=0.0
resonance=10.0
```

## The biquad

The filters of 2, 4 or 6 poles kind are provided as biquad filters. A biquad is an IIR filter of the second order. 4-pole and 6-pole are obtained by cascading.

The transfer function of the generic biquad is:

`H(z) = (b0 + b1*z^-1 + b2 * z^-2) / (a0 + a1 * z^-1 + a2 * z^-2)`

From this formula, the response is computed by substituting `z` for `exp(2*pi*j*fc/fs)`.
The polar form of the result are the filter's amplitude and phase response.

When `H(z)` is substituted with `Y(z) / X(z)`, and then this identity applied to pass from z-domain to time domain:
`(A*X(z)*z^-N)` → `(A*x[n-N])`

It's the implementable Direct Form I equation:

`y[n] = (1/a0) * (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2])`

Then, `a0` is factored into the equation to save an instruction in the computation:

`y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2] - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]`

It's the normalized form which is needed to use this filter in faust function `fi.iir`.

### The RBJ biquad

The most frequently used biquad kind is [RBJ](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html), named after initials of the author.

It has a set of formulas for calculating coefficient for various kinds of responses.

We select this filter for our implementation, because it matches near perfectly in the comparisons.

It's available in faust in `maxmsp.lib`.
Only in late or developer versions, this library was relicensed permissively, so instead it's copied in our source tree with the appropriate license mention. (as `rbj.lib`)

Relevant discussions and PR:
- https://github.com/sfztools/sfizz/issues/30

## Implementation

The implementation of the filters is subject to following prerequisites:
- have controls which map to the documented opcodes
- have ability to modulate frequency and resonance and fast rate
- have stability, particularly in presence of important modulation

We have a source `sfz_filters.dsp` written in faust language, an entry point to the filters.

Faust is a specialized programming language which is able to generated optimized DSP code, in some aspects which are not sometimes possible in a non-specialized language.

For instance, Faust is aware of some notions of variability, and adapt code generation and optimization accordingly.
It can be defined like this:
- constant: a value which never varies
- a-rate: a value which updates at the audio rate, once per frame
- k-rate: a value which updates at the control rate, once per `process` call

When you have a filter which can modulate dynamically its response, there is potentially expensive computation involved.
For instance, formulas of RBJ biquads involve trigonometry.

To make modulation efficient as much as possible, this computation must be avoided at a-rate, but rather be made at k-rate.

For this reason, we define out filter controls as parameters, as in the sliders of a hypothetical UI, and not signals.
It's how to implement k-rate update. (even if no UI, faust calls it "sliders")
```
cutoff = hslider("[01] Cutoff [unit:Hz] [scale:log]", 440.0, 50.0, 10000.0, 1.0);
Q = vslider("[02] Resonance [unit:dB]", 0.0, 0.0, 40.0, 0.1) : ba.db2linear;
```

Our filter corresponding to each type has a naming convention starting with "sfz".
As an example, the 2-pole lowpass is the function `sfzLpf2p`. It is 1-in, 1-out.

## Generation

When filter functions are implemented, we generate all with a script since there exist a high quantity of them.

See `scripts/generate_filters.sh`.

The output is a file of `.cxx` extension for each. It's a special extension because it's not listed in CMake sources, rather all filters are included by a single implementation file. It permits also to inline them into their caller.

The following faust options are used which are notable:
- `-pn`: sets the name of the function to generate from `sfz_filters.dsp`, in this case the name of our filter function as noted in the convention above
- `-cn`: sets the class name of generated C++
- `-double`: indicates to perform internal computations in double precision, which is found to be necessary for numeric stability
- `-inpl`: allow to process the buffers in place; from what I can tell, it doesn't have performance impact.

If you want to use a `faust` Docker image, you have to update the scripts to replace the line that calls `faust $FAUSTARGS ...` with `docker run -v $PWD:/faust faust:latest $FAUSTARGS ...`.

On 2020-02-20, the version of `faust` used was `2.20.2` -- alot of 2 and 0 but it's not on purpose...

## Smoothing

When the filter is modulated, we need a form of smoothing to apply, such that a modulation never applies a too brutal change which produces a discontinuity, destabilizing the filter.

The smoothing function is a single-pole lowpass filter. Defined in `sfz_filters.dsp`, it's the function `smoothCoefs`.

Handling this problem, there is one special consideration to consider:
- if smoothing is applied to parameters Fc and Q, they will be varying at a-rate, and also the filter computation will also become a-rate, regardless that Fc and Q controls are k-rate, killing efficiency.
- the solution is to apply smoothing to the individual coefficients `bN` and `aN`; since biquads are normalized, hence eliminates coefficient `a0`, there is generally need for 5 smooth filters (`b0`, `b1`, `b2`, `a1`, `a2`).
In some RBJ filter types, there are some coefficients which are constant or equal to another, in this case faust will be able to eliminate and simplify as needed, so it needs less smooth filters than general case.

Note that in Faust, the smoothing starts from for all parameters and as such it can cause a strange transient effect at the beginning of the filter.
To handle this in the least hacky way, there is a boolean value in the Faust filters that deactivate the smoothing.
The idea is to deactivate the smoothing, process a single empty sample with the proper parameter values, and then reactivate the smoothing.
The parameters are now set to their initial values and processing can continue with the original smoothing function.
In the wrapper, the `prepare` function call does this.

Technically, the smoothing is a 1-pole IIR filter with a time constant `tau` set at 1ms. 

Relevant discussions and PR:
- https://github.com/sfztools/sfizz/pull/70
- https://github.com/sfztools/sfizz/pull/64

## Filter wrapper

There exists a filter wrapper class written for multi-mode. This is found in `SfzFilters.h`.

This is what is special regarding this wrapper:
- the type of filter is selected through a choice of enumerated constants;
- all filter objects are instantiated in this, but only one used based on value of type;
- when there is a change of type, the memory of the new filter is cleared to zero, to avoid any glitches; theoretically a type change may cause discontinuity, but not in SFZ, since filter type is always fixed.
- when the filter is not subject to modulations, the function call `process` is optimal.
- when is is modulated, invoke `processModulated`. In this case, the filter updates will occur at a fixed interval `N` frames, to limit the frequency of updates. It's a latency-performance compromise. `N` is defined as `kFilterControlInterval` in the file `SfzFilterDefs.h`.

## Relation of controls to SFZ opcodes

- `cutoff`: it's the opcode `filN_cutoff` (Hz)
- `q`: it's the opcode `filN_resonance` (dB)
- `pksh`: it's the opcode `filN_gain` (dB)
