## Synthesis

- [x] Oscillator mode with FM & Ringmod

- [ ] Virtual analog filters with non-linearities

- [ ] 2D Wavetable

- [ ] Wavetable oscillator from formula
  - eg: `sample="sin(pi * x)"` *Serum: sin(pi * -x)*
  - see: https://s3.amazonaws.com/decembercymatics/Serum_Manual.pdf (chapter 15: Formula Parser)
  - discussion of Serum parser: https://www.kvraudio.com/forum/viewtopic.php?t=425973
  - MIT parser library: http://www.partow.net/programming/exprtk/index.html
  - This suggestion is not to copy Serum, but rather enable custom wavetables to be constructed by Sfizz

- [ ] Granular

- [ ] Physical modelling (waveguide opcodes)

- [x] Scala tuning support (load Scala file, set root note of scale & mapped A4=Hertz value)
  - see http://www.huygens-fokker.org/scala/scl_format.html
  - default: (Equal Temperament, C, A4_MIDI_note_69=440hz)

- [ ] Analogue tuning of regions, (not Analogue modelling)
  - possibly `tune=analogue`? Or additional opcode.

## Modulation

- [ ] Arpeggiator

- [ ] Random walk LFO
  - see https://www.muffwiggler.com/forum/viewtopic.php?t=10563

- [ ] LFO sync to host tempo

- [ ] LFO header for free running global LFOs

## Sample Playback

- [ ] Global memory storage, allow multiple Sfizz instances to use same file pool

- [ ] Slice sample playback, from audio file with markers, useful for beat slicing, or concatenated sample libs. eg:

```
<region>
sample=mysample.wav
slice=1
```
- Possibly allow custom map for samples without markers?`slice_map=sampleslices.csv`

- [x] Allow sfizz to load samples into RAM, instead of streaming, possibly `hint_ram` or as per Sforzando ARIA.

- [ ] Load multi-channel samples, and allow selecting stereo/mono channel per region

## Effects

- [ ] Standard effect set

- [ ] Signal flow arrangements
  (cf. `dsp_order`)

- [ ] Effects per region / voice

## Routing

- [ ] Flexible routing : allow regions to be routed to separate sends

- [ ] Multi-channel output

## Expressivity

- [ ] Metalanguage to write SFZ more easily, with less repetition.
  Consider some ideas from CamelAudio Alchemy.
  - edit several properties of an item with indenting syntax
  - load data from CSV files

## Intruments

- [ ] Import other existing sample libraries

## MIDI Standards

- [ ] MPE
- [ ] MIDI 2.0

## Community

- [ ] Sfizz Cloud integration
  - similar to Blender Cloud membership, allow download of sample sets and tutorials