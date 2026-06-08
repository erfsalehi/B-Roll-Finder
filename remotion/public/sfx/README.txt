Drop short sound-effect files here, named exactly:

  swoosh.mp3   (title cards / headings)
  ding.mp3     (stats / numbers / money)
  thud.mp3     (emphasis pops)

Optional variety: add numbered variants and one is picked per overlay
(deterministically, keyed on the overlay text — so a long video doesn't repeat
the same sound, while re-renders stay cache-stable):

  swoosh.mp3, swoosh1.mp3, swoosh2.mp3, ...
  ding.mp3,   ding1.mp3,   ding2.mp3, ...
  thud.mp3,   thud1.mp3, ...

They get baked into the rendered overlay clip (Remotion <Audio>). If a file is
missing, the Python wrapper (core/overlays_remotion.py) sets the overlay's sfx to
"none" so the render still succeeds — it just won't have sound.

Free sources: freesound.org, mixkit.co, pixabay.com/sound-effects. Keep them
under ~1s and normalized. The pipeline can also pull these automatically later
via core/sfx.py (Freesound) — for now, add them manually.
