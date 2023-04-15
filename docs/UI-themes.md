Unlike the `image` opcode and the (future planned) XML UI,
sfizz themes are internal to the player.
They appears as XML files inside a directory where the name of the directory
is the theme name. This because it makes it possible to include other custom
resources other than the xml definition. These directories resides inside
the `Themes` directory, located alongside with the other plugin resources.
The file name is fixed to `theme.xml`, and currently it appears like the following.\
The main `sfizz-theme` tag contains the attributes `author` and the sfizz `version`
in which the theme was introduced, so that if in the future the scheme will change,
we can bump the version, and it will not be retro-compatible.\
Currently the theme(s) supports only colors and palettes.\
Palettes are referred as `normal` for the usual color scheme and `inverted`,
used for main bar which has a different contrast.\
All colors are referred to the current layout components' colors.

```xml
<?xml version="1.0"?>
<sfizz-theme author="redtide" version="1.0.1">
    <color name="frameBackground">#121212</color>
    <palette name="normal">
        <color name="boxBackground">#1d1d1d</color>
        <color name="text">#e3e3e3</color>
        <color name="inactiveText">#a2a2a2</color>
        <color name="highlightedText">#e9e9e9</color>
        <color name="titleBoxText">#e3e3e3</color>
        <color name="titleBoxBackground">#2d2d2d</color>
        <color name="valueText">#e3e3e3</color>
        <color name="valueBackground">#121212</color>
        <color name="icon">#e3e3e3</color>
        <color name="iconHighlight">#ff6000</color>
        <color name="knobActiveTrack">#006b0b</color>
        <color name="knobInactiveTrack">#6f6f6f</color>
        <color name="knobLineIndicator">#ffffff</color>
        <color name="knobText">#ffffff</color>
        <color name="knobLabelText">#ffffff</color>
        <color name="knobLabelBackground">#006b0b</color>
    </palette>
    <palette name="inverted">
        <color name="boxBackground">#2d2d2d</color>
        <color name="text">#9e9e9e</color>
        <color name="inactiveText">#a2a2a2</color>
        <color name="highlightedText">#e9e9e9</color>
        <color name="titleBoxText">#9e9e9e</color>
        <color name="titleBoxBackground">#2d2d2d</color>
        <color name="valueText">#9e9e9e</color>
        <color name="valueBackground">#121212</color>
        <color name="icon">#9e9e9e</color>
        <color name="iconHighlight">#ff6000</color>
        <color name="knobActiveTrack">#006b0b</color>
        <color name="knobInactiveTrack">#6f6f6f</color>
        <color name="knobLineIndicator">#ffffff</color>
        <color name="knobText">#ffffff</color>
        <color name="knobLabelText">#ffffff</color>
        <color name="knobLabelBackground">#006b0b</color>
    </palette>
</sfizz-theme>
```
