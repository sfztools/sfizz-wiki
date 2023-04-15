## Curve formula

**0. linear, from 0 to 1**

`f(x)=x` with `x=(CCval/127)`

**1. bipolar, from -1 to 1 (useful for things such as tuning and panning, used by CC10 panning by default)**

`f(x)=2*x-1` with `x=(CCval/127)`

**2. linear inverted, from 1 to 0**

`f(x)=1-x` with `x=(CCval/127)`

**3. bipolar inverted, from 1 to -1**

`f(x)=1-2*x` with `x=(CCval/127)`

**4. concave (used for CC7 volume tracking and amp_veltrack)**

`f(x)=x*x` with `x=(CCval/127)`

**5. Xfin power curve**

Crossfading formula.
It's like a typical pan formula for the right channel.

ARIA's exact formula is `f(x)=pow(x, 0.5)` with `x=(CCval/127)`

Another candidate will be `sin(0.5*pi*x)`, for the pan formula already existing in Sfizz.

**6. Xfout power curve**

It's the opposite of Xfin power curve.

`f(x)=pow(1-x, 0.5)`,  with `x=(CCval/127)`

Alternative `cos(0.5*pi*x)`

## Curve numbers

When a `<curve>` element without an explicit `curve_index` is met, it takes the next available index starting from 1.

Most times, software will reserve 6 slots for predefined curves.
This will make the first user curve be 7, and it's a good idea to follow this as a convention.

Example
```
<curve> v000=0 v64=0.1 v127=1.0 // the curve at index 7
<curve> v000=0 v64=0.1 v127=0.2 // the curve at index 8
```

## Explicit curve index

ARIA opcode `curve_index` permits to specify the explicit index.

It can't seem to mix curves which are explicitly indexed and those which aren't.

Explicitly-indexed curves after implicitly-indexed ones seem to be discarded by the loader, and vice-versa.

## Interpolation

Both software are using linear interpolation to evaluate between the defined values. (verified)

## Bounds

- In ARIA, the curve values are not bounded.
- In Cakewalk, the curve values are clamped in the domain [-1:+1].