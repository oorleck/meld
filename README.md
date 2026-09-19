# meld

Find phone videos of the same concert, line them up on one timeline using their audio, and fuse them into a
single long video and audio track. Personal tool; runs locally.

## Setup

```
uv sync
```

ffmpeg comes bundled (via `imageio-ffmpeg`), nothing else to install.

## Windows app (no terminal needed)

A simple window for people who don't use a terminal: type a band and a year (or a date), choose how many videos
and the picture quality, press **Start**. It searches, downloads, lines the videos up, mixes the sound and cuts the
video, showing the step it is on, and finishes with buttons to play the result or open its folder. **Cancel**
stops it; videos already downloaded are kept. Run it from source with `uv run meld gui`.

The picture quality (1080p, 720p or the smallest, 480p) caps how large the videos are downloaded and is also the size
of the finished video, so 480p uses the least disk space and time; it is fine for lining the videos up and for a
watchable result, but not for sharp footage.

To build an installer you can hand to someone (`dist-installer\Meld-Setup-<version>.exe`, about 60 MB):

```
powershell -File packaging\build.ps1
```

This freezes the app with PyInstaller, self-tests it (sync, mixing, video cutting and a real YouTube search),
builds the installer with [Inno Setup](https://jrsoftware.org/isdl.php) (install it first; the script says so if
it's missing) and then installs, runs and uninstalls the installer in a scratch folder to make sure it isn't
corrupt. The installer needs no administrator rights, installs per user, adds a Start Menu shortcut (and an
optional desktop one) and shows a personal-use notice. An installed copy can be checked any time with
`Meld.exe --selftest report.txt [--online]`.

- **Several concerts in one search.** A search like "oasis 2025" finds clips of many different nights, and only one
  night can be combined. When the results are from more than one, Meld shows a list ("4 Jul 2025 · Cardiff Principality
  Stadium, 14 videos", ...) and only downloads the one you pick (or all of them, if you're not sure). Nights are told
  apart by the dates in the titles (4/7/25, July 4th 2025, 04.07.25, ...), and a title with no date joins a night when
  its venue and city words point at it. Videos that give no clue are tried with whichever night you pick and dropped
  by the audio match if they don't fit. If the titles state no dates, Meld can't tell nights apart and uses everything,
  as before. On the command line, `meld fetch --dry-run` / `meld auto --dry-run` list the concerts it found.
- **The video and the sound are saved** straight into `Documents\Meld` (or the folder you choose) as `<name>.mp4` and
  `<name>.wav`. The name is made of the words most of the YouTube titles have in common, for example
  `Coldplay at Wembley Stadium 2022`; if the titles give nothing to go on, it is what you typed. An existing file is
  never overwritten: the next one becomes `<name> (2)`.
- **Only those two files are kept.** When a run finishes, the downloaded videos and all working files are deleted
  (only Meld's own; anything else in the folder stays). Turn on **Keep the downloaded videos** to keep them in
  `<folder>\<search>\`. A run that is cancelled or fails always keeps them, so trying again does not download them again.
- **The YouTube downloader (yt-dlp) updates itself**: YouTube breaks old versions every few weeks, so the app keeps
  its own copy in `%LOCALAPPDATA%\Meld` and refreshes it at start-up when it's older than two weeks
  (the "Update the YouTube downloader" link at the bottom of the window does it on demand). Uninstalling removes that folder but never your videos.
- The installed app does not include 3D reconstruction (see below); it needs PyTorch and several GB of model files.
- The bundled ffmpeg is a GPL build; see `packaging\NOTICE.txt` if you share the installer.

## Usage

### One command

```
uv run meld auto "coldplay yellow wembley 16 august 2022"
```

Searches YouTube with several variants of the query (live, fan video, front row, crowd, 4K, full song), merges
and de-duplicates the results, downloads up to `--max-clips` (default 500) in parallel, then runs sync, audio and
video. The project goes in `projects/<query>` unless you pass `-p`. Useful flags: `--max-clips 50`,
`--limit 40` (results kept per search query, default 250), `--max-duration 900`, `--max-height 720`, `--require wembley` (title must
contain the word; repeatable), `--dry-run` (just list what it would download, and what it dropped and why),
`--size 1280x720`.

Search is strict by default (`fetch -s` too). A result is kept only if every meaningful word of the query is in
its title or channel name, and, when the query contains a date or year, its title doesn't name a different one
(`13.08.22`, `21 August`, `20 Agosto`, `2022-08-21`, `2008`, ... in several languages).

What happens to titles that state no date at all depends on the query: for a bare year (`metallica 2003`)
they are dropped, since they could be from any year; for a full date they are kept, because the date is often
only in the description. Override with `--require-date` (always drop) or `--allow-undated` (always keep).
`--loose` switches all of this off. The more words in your query, the narrower the results: leave out the
venue or date if you want to catch more clips.

Only concert footage is kept. A title needs a live signal (live, concert, festival, stadium, arena, fancam,
a dated performance like `(Madrid, Spain - June 22, 2003)`, ...) and must not look like an interview,
commercial, audition, rehearsal, studio session, lesson, cover or tribute. This is judged from the title alone,
so a real concert whose title says nothing of the kind (`Coldplay Yellow Wembley 2022`) is dropped too; add
`--any-video` to keep those. `--dry-run` shows every dropped title and the reason, which is the quickest way to
see what a query does.

### Step by step

Each concert lives in a project directory (default `projects/default`) with `clips/`, `cache/` and `out/`.

```
# 1. collect clips: search YouTube and/or pass URLs (or just drop files into projects/<name>/clips)
uv run meld fetch -p projects/show -s "artist venue 2024 live" --dry-run
uv run meld fetch -p projects/show -s "artist venue 2024 live"
uv run meld fetch -p projects/show https://www.youtube.com/watch?v=...

# 2. align everything on a shared timeline (writes timeline.json, rejects clips that don't match)
uv run meld sync -p projects/show

# 3. fused audio  -> out/fused_audio.wav
uv run meld audio -p projects/show

# 4. multicam video -> out/fused.mp4
uv run meld video -p projects/show --size 1920x1080

# or steps 2-4 at once
uv run meld run -p projects/show
```

Search results are only candidates. The sync step keeps clips whose audio matches the rest and rejects
the others (listed with a reason in `timeline.json`), so a wide search is fine.

The defaults are generous (up to 250 results per query and 500 clips), so a big run downloads a lot. Sync stays tractable because each clip is compared against at most `--max-compare` (default 25) aligned clips, longest first; raise it if clips you expect to match are being rejected.

## 3D reconstruction (experimental)

```
uv sync --extra recon                                   # PyTorch + CUDA, about 5 GB in .venv
uv run --extra recon meld reconstruct -p projects/show [--at 120]
```

Because the clips are synced, the frame at one instant from every phone that was filming shows the same moment.
`reconstruct` picks the moment when the most phones overlap (or `--at SECONDS`), feeds those frames to
[VGGT](https://github.com/facebookresearch/vggt) (camera poses and dense depth in one pass) and writes
`out/recon_<t>s/`: `scene.ply` (coloured point cloud), `cameras.json`, `viewer.html` (orbit the scene in a
browser and jump to each phone's viewpoint; needs internet for three.js), `swing.mp4` (a virtual camera swinging
around the scene) and the source frames. The first run downloads about 4.7 GB of model weights to
`~/.cache/huggingface` and fetches VGGT's source to `~/.cache/meld`. Use `uv run --extra recon ...` (a plain
`uv sync` removes the extra). The model and its weights are under Meta's own research licence.

**It does not work on the concert footage tried so far.** It works well on sharp, well-lit photos (VGGT's sample
scenes give a coherent model with median confidence about 12), but on real Wembley clips, 13 phones at one instant,
the model's confidence was a flat 1.0 for every pixel, and classic structure-from-motion (COLMAP) could not verify
a single camera pair either. Dark, saturated, low-resolution video with very different zoom levels between phones is
too hard. Rather than write a smeared point cloud, `reconstruct` reports the model's confidence and refuses to
write anything when it is low (`--force` overrides). It may work for brighter, sharper, higher-resolution footage
with several phones at similar zoom, so it is left in and checked automatically.

## How it works

- **sync**: GCC-PHAT cross-correlation of the audio. Starts from the longest clip and grows the aligned set;
  a clip joins when its best correlation peak is `--min-z` standard deviations above the noise.
- **audio**: every 0.5 s block of every clip is scored (clipping, level, agreement with the other clips'
  spectra); the output crossfades toward the best sources instead of summing mics, which would comb-filter.
- **video**: clips are scored for sharpness and exposure; a greedy editor cuts between angles with a minimum
  and maximum shot length. Vertical clips are fitted over a blurred background. Cuts land on exact frame
  boundaries so picture stays locked to the audio.
- Time nobody recorded is skipped in both audio and video.

## Limitations (v1)

- Assumes constant offsets: phone clock drift over a long clip (tens of ms per 10 min) is not corrected.
- Clips from different nights of the same tour can sync just as confidently as same-night ones when the show
  plays to backing tracks (seen in testing with Coldplay at Wembley: 17/20/21 Aug clips aligned with the 16 Aug
  ones). `sync` prints each clip's title so you can prune `clips/` and re-run.
- 3D reconstruction is experimental and did not work on the real concert footage tried (see above); the fused
  video itself only switches between angles.
- Downloading from YouTube is against its ToS and concert footage is usually copyrighted; keep it personal.

## Tests

```
uv run pytest
```
