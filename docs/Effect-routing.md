## Bus chaining

When multiple effects are in a bus, they are chained according to their order of appearance in SFZ.

Example: chained effects in bus
```
<region>
sample=sinewave.wav
oscillator=on
lokey=0
hikey=127
effect1=100

<effect>
//
directtomain=0
fx1tomain=100
//
bus=fx1
type=lofi
decim=100
decim_dry=0
decim_wet=100

<effect>
bus=fx1
type=fverb
reverb_type=hall
reverb_size=20
reverb_tone=90
reverb_dry=0
reverb_wet=100
```

Example: chained effects in default chain
```
<region>
sample=sinewave.wav
oscillator=on
lokey=0
hikey=127

<effect>
type=lofi
decim=100
decim_dry=0
decim_wet=100

<effect>
type=fverb
reverb_type=hall
reverb_size=20
reverb_tone=90
reverb_dry=0
reverb_wet=100
```

## Volumes (RXP)

- `effect1`: range 0%-100%, gain linear
- `fx1tomain`: range 0%-100%, gain linear
- `apan_dry`: range 0.0-1.0, gain linear

## Analysis 1: RXP single

Synth: RXP

SFZ template
```
<region>
sample=sinewave.wav
oscillator=on
effect1=100

<effect>
bus=fx1
fx1tomain=100
type=fverb
reverb_type=hall
reverb_size=20
reverb_tone=90
reverb_input=100
reverb_dry=80
reverb_wet=50
```


| `effect1`  | `bus` | `fx1tomain` | `reverb_input` | *Result*        |
| ---------- | ----- | ----------- | -------------- | --------------- |
|            |       |             |                | Effect is heard (high volume) |
| Omit       | Omit  | Omit        |                | Effect is heard (moderate volume) |
| Omit       |       | Omit        |                | Effect is not heard |
| Omit       |       |             |                | Effect is not heard |
|            | Omit  |             |                | Effect is heard (high volume) |
|            |       | Omit        |                | Effect is not heard |
|            | Set to `fx2`  |             |                | Effect is not heard |
|            | Set to `fx2`  | Set opcode to `fx2tomain` |                | Effect is not heard |
| Set opcode to `effect2` | Set to `fx2`  | Set opcode to `fx2tomain` |                | Effect is heard (high volume) |
| Set opcode to `effect2` | Omit  | Set opcode to `fx2tomain` |                | Effect is heard (high volume) |
| Set opcode to `effect2` | Set to `fx1`  | Set opcode to `fx2tomain` |                | Effect is not heard |
|            |       |             | Any value | Does not change anything |
|            |       | Set opcode to `fx1tomix` |                | Effect is heard (high volume) |

**Note:** the all-omitted case is listed as *moderate volume*, the explicit case is *high volume*.  
It's because the latter mixes in another path of the dry signal (see next table below).
This other path is neutralized when setting `directtomain=0`.  

- With `reverb_dry` to **0**

| `effect1`  | `bus` | `fx1tomain` | `reverb_input` | *Other* | *Result*        |
| ---------- | ----- | ----------- | -------------- | ------- | --------------- |
|            |       |             |                |         | Hear a mix of reverb sound and dry sound |
| Omit       | Omit  | Omit        |                |         | Hear the reverb sound only |
|            |       |             |                | add `directtomain=0` in `<effect>` | Hear the reverb sound only |
