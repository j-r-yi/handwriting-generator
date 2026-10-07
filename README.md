# Handwriting Generator

Turn typed text into realistic pages of **your own handwriting**, built from
samples of characters you actually wrote.

You give the program a few handwritten versions of each character (`a`, `b`,
`c`, …). It cuts them out, cleans them up and stores them in a *handwriting
profile*. When you type text, it composes that exact text from your real
handwritten characters onto blank, ruled or graph paper, and you can download
the pages as PNG or as a multi-page PDF.

This is **not** an AI handwriting generator. No model is involved, and nothing is
synthesised: every letter on the page is one of your own samples. That makes
the app:

- **exact**: the text is never altered, corrected, rewritten or shortened.
  Characters without a sample are clearly marked instead of being dropped.
- **predictable**: the same text, profile, settings and random seed always give
  the identical image.
- **private and offline**: everything runs on your computer.
- **free**: no API keys, no paid services, no cloud.

---

## Privacy and local processing

- All image processing, extraction and rendering happen **locally** in Python
  (OpenCV, Pillow, NumPy).
- Handwriting samples, uploaded scans and the text you type are **never sent
  anywhere**. The application code makes no network requests and contains no
  telemetry.
- Streamlit's own anonymous usage statistics are **switched off** in
  `.streamlit/config.toml` (`gatherUsageStats = false`). The server only listens
  on `localhost`, so other devices on your network cannot reach it.
- No OCR, cloud APIs, image-generation services or AI models are used.
- Your data lives in plain files under `data/` (see *Where data is stored*), and
  `.gitignore` keeps it out of version control.

---

## Requirements

- Python **3.12**
- Direct dependencies, all free and open source (see `requirements.txt`):
  `numpy`, `Pillow`, `opencv-python-headless`, `streamlit` (1.64 or newer),
  `pypdfium2` (renders uploaded PDF scans locally), `pytest`
- Any modern laptop. A full page renders in well under a second.

---

## Installation

### macOS / Linux

```bash
git clone <repository-url> handwriting-generator
cd handwriting-generator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows (PowerShell)

```powershell
git clone <repository-url> handwriting-generator
cd handwriting-generator
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

(With `cmd.exe`, activate with `.venv\Scripts\activate.bat` instead. If
PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

Streamlit prints a local address such as `http://localhost:8501` and usually
opens it in your browser automatically. To stop the app, press `Ctrl+C` in
the terminal. Next time, activate the virtual environment and run
`python -m streamlit run app.py` again. Your profiles are still there.

---

## Using the app

The bar at the top switches between four pages: **Write**, **My handwriting**,
**Sample sheet** and **Settings**. The profile button in the top-right corner of
a page (it shows the current profile's name) switches profiles and creates new
ones. Your text and settings are kept while you move between pages.

### 1. Create a profile

On first start the app asks you to name a profile (e.g. *My handwriting*). A
profile is a collection of handwriting samples. You can keep several (e.g. neat
vs. quick handwriting): create more from the profile button in the top-right
corner.

### 2a. Add samples with the sample sheet (recommended)

**Sample sheet** page, which walks through three steps:

1. **Print the sheet.** Choose the paper size and **samples per character** (default 3; up to 8).
   More samples means more natural variation when a letter repeats. For messy
   notes, choose 5–8 and write the sheet quickly, the way you take notes. The
   app can wobble, tilt and distort your letters, but it can only use the
   letter *shapes* you give it.
2. Download the **PDF** (or **PNG images**; **Preview** shows the pages). The default
   sheet is three pages and covers `a–z`, `A–Z`, `0–9` and 32 punctuation marks.
3. Print it at **100% / actual size**.
4. Write each character once in each of its boxes with a dark pen, at your
   normal size. Labels sit *outside* the boxes, so they are never extracted.
   The small tick marks on the box sides show the baseline: letters such as
   `g j p q y` should hang below it. Leave a box empty to skip it.
5. Scan or photograph every page: flat, in focus, evenly lit, with all four
   black corner squares visible. Phone photos at a slight angle are fine.
   **For the sharpest letters, scan at 600 DPI.** The app reads each page at
   the resolution of the scan or photo, up to 600 DPI. A 300 DPI scan or a
   typical 12 MP phone photo gives about 300–370 DPI. A 600 DPI scan, or a
   close, frame-filling photo from a 48 MP camera, gives up to twice the
   detail per letter.
6. **Import your pages:** upload them and click **Extract handwriting**. You can
   upload several images at once (PNG, JPEG, TIFF, …) **or a multi-page PDF**
   straight from your scanner. Each PDF page is rendered on your computer at
   the resolution of the scan inside it (300–600 DPI) and imported
   separately, in any order.
7. Review the result. Each page shows an *aligned preview* (green boxes
   contained handwriting), and every extracted character is shown with a
   **Use** checkbox. Untick anything that looks wrong.
8. Click **Save … samples**. Optionally turn on *Replace existing samples of
   these characters* when re-importing a better sheet.

How it works: each sheet has corner markers and a small printed page code
(the black squares next to the top-left marker). The app finds the markers,
corrects perspective and rotation (including upside-down or sideways photos),
reads the page code to learn which page it is, snaps every box to the
printed borders, and extracts the ink. **No OCR is used**: the template
geometry already says which character belongs in each box. If the page code
cannot be read, open **Pages not recognised?**, turn on **Identify pages
manually** and choose the page number
(for a multi-page PDF, choose the number of its first page; the rest count up).

### 2b. Add samples manually

**My handwriting → Add images** is useful if you already have
cropped images of your letters:

1. Choose **Pick a character** (the standard set plus any symbols you have
   added) or **New symbol** (see below).
2. Upload one or more images of that character (PNG, JPEG, BMP, TIFF, WebP,
   GIF; dark ink on light paper, or a transparent PNG).
3. Each image is cleaned automatically (background removed, specks and ruled
   lines dropped, cropped to the ink). Check the previews and untick bad ones.
4. Click **Add N sample(s)**.

Manually added samples are sized using typical letter proportions. Sheet
imports preserve your *real* proportions (e.g. how tall your capitals are
compared with your lowercase), so the sheet gives the most faithful result.

#### New symbols

The standard set already includes every printable ASCII character on a
keyboard (`a–z`, `A–Z`, `0–9` and all 32 ASCII symbols such as `~ ^ | \ { } @`).
To write anything else, choose **New symbol**: click one of the **common
symbols** or type/paste your own, then upload pictures of it as above.
Common symbols include bullets and arrows (`• → ← ↑ ↓`), maths (`° ± × ÷ ≈ ≠ ≤
≥ √ ∞ π µ`), currency (`€ £ ¥ ¢`), typography (`© ® ™ § ¶ … – — “ ” ‘ ’ « »
¿ ¡`) and accented letters (`é è ê ë à â ä á ç ñ ö ô ó ü û ú ï î í ß æ ø`).

- Only characters you can type on a keyboard are accepted. Emoji and picture
  symbols are rejected, and accented letters typed as two code points are
  combined into one (so `é` is always the same character).
- Common symbols are sized and placed like printed text: `•` and `→` sit at
  mid height, `°` and `™` are raised, `ç` hangs below the line, and accented
  letters are as tall as the letter plus its accent. Other symbols are
  treated like a letter sitting on the line.
- New symbols appear under **Your symbols** on the **Characters** tab and can
  be browsed and deleted like any other character. They are not on the
  printable sample sheet.
- Without your own `“ ” ‘ ’ – —`, the app keeps using your `" ' -` for them.

### 3. Manage the profile

The **My handwriting** page starts with totals (samples, characters covered,
characters still missing) and has four tabs:

- **Characters**: every character with its number of samples (dashed red =
  none yet), and any symbols you have added under **Your symbols**;
- **Browse samples**: view all variants of a character, tick the bad ones,
  confirm, and delete;
- **Add images**: manual upload and new symbols (above);
- **Manage profile**: **rename** it; **clear all samples** to start over while
  keeping the profile and its name (tick the confirmation box first); or
  **delete this profile** (you must type the profile name to confirm).

Deleting and clearing are recoverable: files are moved to `data/trash/`, not
erased. A cleared profile's images and its old `profile.json` are kept in
`data/trash/<profile>/cleared-<date-time>/`. To undo a clear, copy that
`glyphs` folder and `profile.json` back into `data/profiles/<profile>/`.

### 4. Generate text

**Write** page. Controls are on the left, the page preview on the right.

1. Paste or type your text. Line breaks, blank lines and indentation are kept.
2. The app immediately lists any **characters without samples**.
3. Choose the style:
   - **Ink**: black, dark blue, blue, or any **custom** colour
   - **Paper**: narrow ruled (default; 1/4" lines, no red margin line),
     college ruled, wide ruled, graph, or blank
   - **Messiness**: *Very Consistent*, *Natural* (default), *Messy* or
     *Rushed notes*
   - **Size**: height of a lowercase *x* in millimetres (default 2.2 mm)
   - **Resolution**: output DPI (same setting as on the Settings page)
4. **More options** has everything else, in four tabs:
   - **Page**: US Letter or A4; **write on** every line / every other line
     (ruled paper) or **line spacing** in mm (blank / graph paper); margins
     (remembered per paper: narrow ruled starts 8 mm from the edges and uses
     most of each line, the others leave room for the red margin line);
     the red margin line
   - **Spacing**: letter spacing (tight by default, 0.04 x-heights) and
     word spacing
   - **Fine-tune**: a slider for every messiness effect, starting from the
     Messiness choice, grouped as:
     - **Letters**: rotation, baseline jitter, size, letter spacing,
       *touching letters* (how far a letter may run into its neighbour, as
       in quick writing: none for Very Consistent, slight for Natural, more
       for Messy and Rushed notes), and
       shape distortion (no two copies of a letter are identical)
     - **Words**: word spacing, and whole words that bounce, change size,
       tilt or lean differently, plus size that drifts as you write
     - **Lines**: lines that slope uphill or downhill, wander, float off
       the rule, and start at a ragged left edge
     - **Layout**: *uneven indents* (bullets, list items and wrapped lines
       don't line up in perfect columns) and *uneven line ends* (lines end
       at different distances from the right margin instead of filling
       every line)
     - **Pen**: pressure (some words lighter), and *gets messier towards the
       end*
     - which sample to use when a character has several
   - **Other**: whether Markdown `#`/`##` headings are underlined (off by
     default), what to
     do with characters that have no sample, and an optional **random seed**
5. Click **Generate**. The preview shows one page at a time (pick the page
   number under the toolbar). Download one combined **PDF**, the current page
   as a **PNG**, or **all PNGs** as a zip.

Each result has a *look number* (its random seed). **Keep this look** fills it
in, so the next Generate gives exactly the same pages, for example after you
fix a typo. Clear the seed to get a new variation every time.

### 5. Formatted notes with Markdown

Turn on **Markdown formatting** under the text box to write structured notes.
**Formatting guide** next to it shows a cheat sheet.

```markdown
# Biology Notes
## Cell Structure
The **cell membrane** controls what enters and leaves the cell.
- Nucleus: holds *genetic* material
  - nested point
- ~~Golgi makes DNA~~ Golgi packages <u>proteins</u>
1. Read chapter 4
- [x] Flashcards
- [ ] Practice quiz
> Remember: structure follows function!
---
```

| Markdown | How it is written |
| --- | --- |
| `#` … `######` headings | larger (×1.7, ×1.4, ×1.2, …); no underline (an option under *More options → Other* adds a hand-drawn one to levels 1–2); no space is added around them: only the blank lines you type (turn on *Free line above big headings* under *More options → Other* for an extra line above `#`–`###`); never left alone at the bottom of a page |
| `-`, `*`, `+` bullets | written as typed: a `-` bullet is your own handwritten dash, `*` and `+` are a dot (your own `•` if you added one as a symbol). Without a sample, the dash or dot is drawn. Text is indented; indent 2 spaces per nesting level |
| `1.` / `1)` numbered | the number is written in your handwriting, text aligned after it |
| `- [ ]` / `- [x]` tasks | a hand-drawn box, ticked when done |
| `**bold**` | a second, slightly offset pen pass (heavier writing) |
| `*italic*` | your letters slanted |
| `~~strike~~` | crossed out with a hand-drawn line |
| `<u>underline</u>` (or `<ins>`) | a hand-drawn line under the text; works anywhere, e.g. inside a bullet. An unclosed tag is written as typed |
| `> quote` | indented, with a line in the margin (can contain lists/headings) |
| `---`, `***`, `___` | a hand-drawn line across the page |
| `` `code` ``, fenced code blocks | written literally, without formatting |
| `[text](url)`, `![alt](url)` | only the text / alt text is written |
| `\*` | backslash makes any symbol literal |

Differences from standard Markdown, chosen for handwritten notes:

- every line break is kept (standard Markdown would merge lines into
  paragraphs);
- spacing is exactly as typed: every blank line skips one line (three blank
  lines skip three), including around headings. Only blank lines at the very
  end are dropped. An empty `>` line closing a quote is a plain gap. With
  *Free line above big headings* on, a big heading's extra space is shared
  with any blank line already above it;
- `<u>underline</u>` is the only HTML that is interpreted;
- tables, footnotes and other HTML are not interpreted. They are written as typed
  text, so nothing is lost.

With the toggle off (the default), the text is written exactly as typed,
including any `#`, `*` or `-` characters. The missing-character check
understands the mode: with Markdown on, syntax characters are not reported
as missing.

### Settings

Choose the output resolution (also on the Write page), see where data is stored, clear caches, and read
the privacy notes.
Resolutions are 150, 200, 300 or 600 DPI. 300 is the default and prints
crisply. 600 stays sharp when zoomed in, especially with samples scanned at
600 DPI, but files are about 4x larger and up to 7 pages are generated at
once.

---

## How the rendering works

- **Exact content.** Every visible character of the input is placed exactly
  once, in order. Only the appearance changes. Line endings are normalised
  (`\r\n` → `\n`) and Unicode is NFC-composed. Neither changes any visible
  character. In Markdown mode, the same guarantee applies to the text that
  remains after the Markdown syntax is interpreted.
- **Documents.** Input becomes a list of blocks (`document.py`). Plain text is
  one paragraph per line. Markdown (`markdown.py`, a small built-in parser
  with no extra dependency) adds headings, list items, quotes, rules and
  styled spans. One renderer handles both.
- **Variants.** Each occurrence of a character uses one of its samples. By
  default variants are dealt from a shuffled "deck", so all of them are used
  before any repeats and the same sample is never used twice in a row
  (`eeeeee` cycles naturally).
- **Proportional widths.** Advance width comes from the real glyph image (an `i`
  takes less space than an `m`), plus configurable letter spacing.
- **Baselines.** Glyphs sit on a common baseline. Descenders (`g j p q y`,
  `,` `;`), brackets, and floating marks (`-` `=` `'` `"` `^` `*`) are placed by
  character class. Sheet imports also record where you wrote relative to the
  printed baseline ticks, which is used when it is plausible. This logic lives
  in `handwriting/baseline.py` so it can be improved independently.
- **Natural variation.** Independent per-letter jitter alone looks "neat but
  shaky", so variation works at four levels:
  - **letters**: clipped-normal rotation, baseline, size and spacing jitter,
    plus a smooth random distortion field that reshapes every copy of a
    sample;
  - **words**: all letters of a word share an offset, size, tilt, slant,
    letter spacing (some words are written looser, others cramped) and pen
    pressure, and size drifts slowly as a random walk;
  - **lines**: each line has its own slope (partly a document-wide habit,
    as real writers drift consistently), a gentle baseline wave, an offset
    from the rule and a start position. Letters rotate to follow the line,
    and underlines and strike-throughs follow it too;
  - **layout**: each nesting level's indent wanders slowly from item to item
    (capped well below one nesting step, so the hierarchy stays clear),
    wrapped lines start a little left or right of the text above (never left
    of their bullet), the gap after a bullet or number varies, and each line
    may end some way before the right margin, as writers move on before
    reaching the edge. A word is never split because of this;
  - **fatigue**: all of this grows towards the end of the text.

  Line and indent shifts are reserved during word wrapping, so the right
  margin is respected. Each effect has its own seeded random stream, so moving one
  slider does not reshuffle everything else.
- **Word wrapping.** Lines break only between words. A word is split only if it
  is wider than the whole line. Manual line breaks and blank lines are kept,
  and text flows onto as many pages as needed.
- **Paper.** Paper is drawn programmatically with Pillow: no image downloads.
  Ruled paper uses standard spacings (narrow 1/4", college 9/32", wide
  11/32"). College and wide ruled have an optional red margin line; narrow
  ruled has none, so text starts at the left page margin. Handwriting
  baselines sit on the rules.
- **Ink.** Glyphs store ink coverage in their alpha channel, so recolouring to
  any colour keeps anti-aliasing and pen-pressure shading intact.
- **Missing characters.** Default: a red outlined box containing the missing
  character, so nothing silently disappears. Alternatives: typeset it with a
  regular font in the ink colour, or refuse to render. Typographic quotes and
  dashes (`’ “ ” – —`) automatically use your `'`, `"` and `-` samples when you
  have no dedicated samples for them.

## How extraction works

For both sheet cells and manual uploads (`handwriting/preprocessing.py`):

1. Convert to grayscale (transparent images are placed on white).
2. Normalise illumination by dividing by an estimate of the paper brightness.
   This removes shadows and lighting gradients.
3. Threshold with Otsu's method, clamped to sane limits so faint strokes survive.
4. Remove straight lines that touch the crop edge (box borders, notebook rules)
   and tiny specks.
5. Find the bounding box of the remaining ink and crop to it (2 px margin).
6. Build a **soft alpha channel** from the normalised darkness. Stroke edges keep
   their anti-aliasing, and strokes are never smoothed, thinned or vectorised.
7. Save as an RGBA PNG with a transparent background.

Sample-sheet pages are flattened at the resolution of the scan or photo (300
to 600 DPI). Each sample records that resolution, so samples captured at
different resolutions still come out the same physical size.

When writing, each letter is resampled only twice: one area-averaged resize
to its final size, then a single step for slant, rotation and shape
variation, with sub-pixel positioning. PDF pages are stored as
high-quality JPEG (quality 95, full colour resolution) to avoid blocky
artifacts around strokes.

---

## Where data is stored

```
data/
  profiles/
    my-handwriting/
      profile.json              # metadata: character → samples, sizes, metrics
      glyphs/
        lower_a/lower_a_001.png # transparent RGBA glyph images
        upper_a/upper_a_001.png
        question/question_001.png
        u00e9/u00e9_001.png     # é (any Unicode character is supported)
  trash/                        # deleted profiles and samples (restorable)
```

- Folder names use **safe character IDs** (`lower_a`, `upper_a`, `slash`,
  `u00e9`, …) instead of the characters themselves. This avoids illegal file
  names and collisions on case-insensitive disks (macOS and Windows treat `a`
  and `A` as the same name).
- `profile.json` maps each real Unicode character to its files and metrics, and
  is written atomically, so a crash cannot corrupt it.
- To store data elsewhere, set the environment variable
  `HANDWRITING_DATA_DIR=/path/to/folder` before starting the app.
- To back up your handwriting, copy the `data/profiles` folder.
- Everything in `data/` (profiles and `trash/`) is listed in `.gitignore`, so
  your handwriting is never committed. Only the empty `data/profiles/`
  folder is part of the repository. Images, PDFs and zips saved directly in
  the project folder are ignored too.

---

## Project architecture

```
app.py                      Streamlit entry point (top navigation only)
ui/                         Streamlit UI; thin layer, no image logic
  common.py                 shared helpers (error display, thumbnails)
  shell.py                  shared page header, profile switcher, welcome screen, styling
  pages/                    one tiny script per page, calling the modules below
  assets/                   logo
  generate_tab.py           Write page: text → pages, settings, preview, downloads
  profile_tab.py            My handwriting page: coverage, upload, browse/delete, manage
  sheet_tab.py              Sample sheet page: print, import, review
  about_tab.py              Settings page: resolution, storage, privacy
handwriting/                core library: no Streamlit dependency
  settings.py               all rendering defaults, presets, page/paper/ink settings
  charset.py                supported characters, safe character IDs, aliases
  models.py                 profile metadata dataclasses, GlyphSet
  errors.py                 user-facing exception hierarchy
  utils.py                  image + PDF page decoding, recolouring, fonts, helpers
  preprocessing.py          OpenCV glyph extraction pipeline
  baseline.py               heuristic vertical metrics (isolated, replaceable)
  sample_store.py           JSON + PNG profile storage with an image cache
  layout.py                 page geometry, word wrapping, line-to-baseline assignment (pure math)
  document.py               block/span model shared by plain text and Markdown
  markdown.py               small Markdown parser for notes (no dependency)
  decorations.py            hand-drawn bullets, checkboxes, rules, underlines, strikes
  paper.py                  programmatic paper backgrounds
  renderer.py               plan → layout → compose pipeline
  template.py               printable sample sheets with fixed geometry
  sheet_import.py           marker detection, perspective correction, cell extraction
  export.py                 PNG / multi-page PDF / ZIP (Pillow + stdlib only)
tests/                      pytest suite with programmatically generated images
data/                       your profiles and trash (git-ignored)
.streamlit/config.toml      localhost-only, telemetry off
```

The `handwriting` package can be used without the UI, for example from a script:

```python
from handwriting import ProfileStore, RenderSettings, render_text

store = ProfileStore()                         # uses ./data
glyphs = store.load_glyph_set("my-handwriting")
result = render_text("Hello, world!", glyphs, RenderSettings(seed=42))
result.pages[0].save("hello.png")
```

---

## Testing

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pytest
```

The suite (193 tests, about 21 seconds) needs no personal handwriting. All
test images are generated on the fly. It covers:

- Markdown parsing (block types, nesting, inline styles, escapes, code, links,
  symbols that must stay literal) and formatted rendering (heading sizes,
  indents, bullets, checkboxes, bold/strike ink, ruled-line spacing,
  determinism, and that syntax is never written or reported as missing);
- word wrapping, manual line breaks, blank lines, indentation, long-word
  splitting, and a check that every character is placed once and in order;
- multi-page overflow, page dimensions for Letter/A4 at several DPIs, and
  margins;
- deterministic output for a fixed seed, and changed appearance with the same
  content for different seeds;
- variant selection (all variants used, no immediate repeats);
- layout: lines that stop short of the margin never split a word, list
  indents wander while nested items stay clearly nested, wrapped lines never
  start left of their bullet, bullets are written as typed, letters touch
  only as far as allowed, headings are not underlined unless asked, narrow
  ruled paper (1/4" lines, no margin line), and every character survives
  the loosest layout;
- messiness: every preset keeps the text exact and reproducible, presets get
  progressively messier, line slope and wander, words moving as units, the
  right margin respected with ragged line starts, shape distortion, pen
  pressure, and fatigue;
- proportional widths, descender placement, and ink recolouring with
  anti-aliasing;
- missing-glyph handling (placeholder, typed fallback, error policy, listing);
- new symbols: keyboard symbols accepted, emoji and lone accents rejected,
  accented letters normalised, size and position rules (bullets, degree
  signs, accents, cedillas), and writing them in text;
- glyph extraction: transparency, tight cropping, uneven lighting, speck and
  ruled-line removal, corrupted/unsupported/truncated files, EXIF rotation,
  and blank images;
- profile persistence across store instances, safe file names, rename,
  duplicates, deletion to trash, clearing all samples (and restoring from the
  trash backup), invalid or unsafe metadata, corrupted sample files, and
  unwritable storage;
- sample-sheet geometry, page-code encoding, and import of simulated phone
  photos (rotation, perspective, uneven light, noise, upside-down, A4), plus
  detection failure;
- PNG/PDF/ZIP export (page count and order, physical size);
- PDF uploads: page order and resolution, corrupted PDFs, page limits, a
  scanned multi-page sample-sheet PDF, and uploading a PDF in the app;
- resolution: sheets read at the scan's resolution (capped at 600 DPI),
  consistent letter size across capture resolutions, PDF scans rasterised at
  their own DPI, 600 DPI output, page limits per resolution, the single-step
  letter transform, and PDF export quality;
- end-to-end headless runs of the Streamlit app using Streamlit's `AppTest`:
  first-run welcome → create profile → manual upload → sheet import →
  generate, paging through a multi-page result, *Keep this look*, settings
  that survive switching pages, and adding a new symbol with pictures.

---

## Limitations

- **Characters are composed individually.** Connected cursive is not
  reproduced: letters do not join, and there are no ligatures or
  context-dependent letter shapes. Print-style handwriting gives the most
  convincing results.
- **Baseline and size heuristics are approximate.** Unusual habits (e.g. a
  descending `f`, very large capitals) may sit slightly off, especially for
  manually uploaded samples, which have no shared size reference. The sample
  sheet gives the best results.
- **Sheet detection is robust, not perfect.** It handles moderate perspective,
  rotation in any direction, shadows and uneven lighting, but not strongly
  curled or folded paper, heavy blur, very dark photos, or pages with corner
  markers cut off. In those cases the app explains the problem so you can
  retake the photo. Writing that touches or crosses the box borders may be
  clipped.
- **Ink extraction assumes dark ink on light paper.** Very light pencil or
  coloured paper may need a cleaner scan.
- **One ink colour per document.** Original pen colour and texture are replaced
  by the chosen colour (pressure shading is kept).
- **No kerning.** Spacing comes from glyph widths plus your spacing settings.
- **Size limits.** Up to 30 pages per generation at 300 DPI, or 7 at 600 DPI,
  to bound memory use.
- **Letter detail is limited by the samples.** A higher output resolution
  makes edges smoother but cannot add detail that was not captured. For
  sharper letters, re-import your sheet from a 600 DPI scan.
- **Markdown is a practical subset.** Tables, footnotes, HTML and setext
  (underlined) headings are written as plain text. Bold is simulated with a
  second pen pass and italics with a slant, so they look natural but are not
  separately written samples.
- **PDF uploads** are limited to 30 pages per file. Password-protected PDFs
  cannot be opened; remove the password first.
- **HEIC photos** (iPhone default) are not readable by Pillow without an extra
  plugin. Export or share the photo as JPEG/PNG instead (iPhone: Settings →
  Camera → Formats → *Most Compatible*).

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Could not find the sample sheet" | Make sure all four black corner squares are visible, the photo is sharp, and the page fills most of the frame. |
| "Page code could not be read" | Tick **Identify pages manually** and pick paper size, samples per character and page number. |
| Some extracted characters look cut off | Untick them in the review, or re-write those characters further from the box borders. |
| A character renders as a red box | It has no samples. Add some on the **My handwriting** page. |
| Letters look soft when zoomed in | Set 600 DPI on the **Settings** page. For more detail, re-import the sample sheet from a 600 DPI scan (the preview on screen is always downscaled; downloads are full resolution). |
| Output looks too neat or uniform | Pick **Messy** or **Rushed notes** under *Messiness*, and raise *Uneven indents*, *Uneven line ends*, *Line slope* or *Word tilt* under **More options → Fine-tune**. Match your real notes: most people write smaller than the default (the default **Size** is 2.2 mm; try 1.8–2.6 mm on ruled paper), and with your own letter spacing (**Letter spacing** under *Spacing*; raise *Touching letters* under *Fine-tune* if your letters often run together). For messier letter shapes, fill in a new sheet quickly with 5–8 samples per character. |
| "Cannot write to the data directory" | Check folder permissions, or set `HANDWRITING_DATA_DIR` to a writable folder. |
| Warning that a profile was skipped | Its `profile.json` is damaged. Restore it from a backup or from `data/trash/`. |

---

## License

Released under the [MIT License](LICENSE).
