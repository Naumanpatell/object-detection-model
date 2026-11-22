# Stickers

Put one cat meme image per expression in this folder, using exactly these filenames:

| File          | Shown when the expression is |
|---------------|------------------------------|
| `happy.png`   | happy (wide smile)           |
| `shocked.png` | shocked (mouth open / brows raised) |
| `angry.png`   | angry (brows lowered / pulled together) |
| `sad.png`     | sad (mouth corners down)     |
| `sleepy.png`  | sleepy (eyes closing)        |
| `neutral.png` | neutral (anything else)      |

- PNGs with a transparent background (alpha channel) look best; plain PNG/JPG-in-PNG also works (treated as fully opaque).
- Any size or aspect ratio is fine; the app scales each sticker to the width of the detected face.
- If a file is missing, the app draws a coloured box labelled e.g. "happy cat" instead, so it runs without any images.
