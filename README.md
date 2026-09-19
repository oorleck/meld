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
and the picture quality, press **Start**. It searches, downloads small audio-only previews, sorts them by their
sound, downloads the videos of one group, lines them up, mixes the sound and cuts the video, showing the step it is
on, and finishes with buttons to play the result or open its folder. **Cancel** stops it; videos already downloaded
are kept. Run it from source with `uv run meld gui`.

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
  night can be combined. So the candidates are first downloaded as small audio-only previews (a fraction of the size of
  the video; twice as many as videos wanted, up to 250) and sorted into groups by their sound. If there are several
  groups of three or more videos, a "Which group?" window lists them ("Group 1: Cardiff Principality Stadium · 92 min,
  14 videos", ...). **Only the videos of the group you pick are downloaded in full**, up to the number you asked for
  (the longest first); each is then lined up with its preview, which is one comparison per video and not a second sync.
  With a single group there is nothing to ask. The previews are kept for the next run of the same search (see below). On the command line,
  `meld auto` still downloads every candidate in full, and `meld fetch --dry-run` lists the concerts it can tell apart
  from the titles.
- **The video and the sound are saved** straight into `Documents\Meld` (or the folder you choose) as `<name>.mp4` and
  `<name>.wav`. The name is made of the words most of the YouTube titles have in common, for example
  `Coldplay at Wembley Stadium 2022`; if the titles give nothing to go on, it is what you typed. An existing file is
  never overwritten: the next one becomes `<name> (2)`.
- **The videos are only kept if you ask.** When a run finishes, the downloaded videos and the working files made from
  them are deleted (only Meld's own; anything else in the folder stays), leaving the finished `.mp4` and `.wav`. Turn on
  **Keep the downloaded videos** to keep them in `<folder>\<search>\`. A run that is cancelled or fails keeps what it had
  downloaded, so trying again does not download it again; they go at the end of the next run that finishes.
- **Running the same search again is quick.** Small things are kept in `<folder>\<search>\` after a run: the audio
  previews (what YouTube sent, a few MB each) and, in `preview\saved\`, what was worked out from them: which videos the
  search found (used for a week), how they sort into groups, and the score of every pair of clips compared. So a second run
  of the same search needs no search, no preview downloads and no sorting, and you can pick another group in the
  "Which group?" window at once; only the videos of that group are downloaded, as they were deleted. If the search
  changes a little (more videos asked for, a new upload) only the clips that are new are downloaded and compared, and a
  run you cancelled goes on where it stopped. The answer is always the one a fresh run would give. What is kept goes with the
  version of Meld: a newer Meld works out again what an older one saved (the previews are what YouTube sent, and stay). To keep the
  disk tidy, Meld clears the previews of all but your five most recent searches, and of any not run for 30 days;
  **Saved searches...** next to the save folder says how much is kept and clears it (kept videos and finished
  files are never touched). On the command line `meld sync` does the same in the project folder (`--no-cache` turns it off).
- **Watch the matching.** When the videos start being sorted by their sound, a second window, *Meld - the matching*,
  opens beside the main one (on a small screen the main window slides to the left edge and the new one overlaps only
  its right part, so Start, Cancel and the steps stay visible) and shows it happening. Each group of videos that line
  up is a timeline with a bar per video, placed where it falls in the group and labelled with its title (without the
  words most titles share, such as the band and the year). All timelines share one time scale, so a bigger group
  looks bigger; videos playing at the same time are stacked. The video being sorted waits in a tray at the bottom
  and a dotted line runs from it to the one it is being compared with, with the score it got; a match slides the bar
  into its group, and a video that links two groups makes them one. A middling score (z 10 to 25) is only believed if
  the same lag holds in every part of the overlap: unrelated clips reach z 10 by chance now and then, and one such
  false match would put songs on top of each other, so the window says "not confirmed" and the pair is not used. When sorting is over the group that is used is
  outlined. Hover a bar for its whole title. Close the window any time; **Show matching** in the footer brings it back.
- **Strict search (optional).** By default a video is kept if its title (or channel name) has each word you typed, at
  the start of a word, and does not name a different date; words such as "live" and "show" are not asked for, and a title
  with no date is kept. Tick **Strict: the title must have every word I typed** and it is much fussier: every word must
  be a whole word of the title itself (only connectors such as "the" and "at" are skipped, and the channel name does not
  count), and the date, month and year you typed must be in the title too. A month can be said in words or in a date, so
  "coldplay august wembley 2025" keeps "Wembley 16/08/2025" and "16 Aug 2025" and drops "12/06/2025", "September 2025" and
  titles that give no month. You get fewer videos, each one that says outright it is what you asked for; if that is too
  few, untick it. On the command line it is `--all-words` on `fetch` and `auto`.
- **YouTube login (optional).** When YouTube answers "Sign in to confirm you're not a bot", downloads fail whatever
  the video. The **YouTube login** link at the bottom of the window lets Meld use the login already saved in one of
  your browsers (Firefox, Edge, Chrome or Brave, whichever are installed): sign in to YouTube in that browser, choose
  it there and press Save. It is read once when a run starts, held in memory and never written to disk, to the log
  or anywhere but YouTube. With a login, downloads run 2 at a time with a short pause between them and at most 100
  candidates are previewed, because heavy automatic use can get a Google account limited by Google (a spare account
  is an option). If it can't be read, Meld says what to do: Firefox is the most reliable; Chrome and Edge must be
  closed and may not work at all, as they encrypt their cookies in a way that is not always readable. On the command
  line it is `--cookies-from-browser firefox` (also `edge`, `chrome`, `brave`) on `fetch` and `auto`.
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

# 2. align everything on a shared timeline (writes timeline.json; clips that line up with each other form a group,
#    the biggest group is used, the others are listed; --cluster 2 takes the second biggest)
uv run meld sync -p projects/show

# 3. fused audio  -> out/fused_audio.wav
uv run meld audio -p projects/show

# 4. multicam video -> out/fused.mp4
uv run meld video -p projects/show --size 1920x1080

# or steps 2-4 at once
uv run meld run -p projects/show
```

Search results are only candidates. The sync step sorts the clips into groups that line up with each other (usually
one group per concert or night) and fuses the biggest; the clips of the other groups are kept in `timeline.json`
(`other_groups`) and clips that match nothing are listed there with a reason, so a wide search is fine. `--cluster N`
fuses the N-th biggest group instead, and `--no-clusters` goes back to growing a single group from the longest clip.
Comparisons run several at a time in separate processes (up to 4, for jobs of 8 or more clips of about 20 s or
longer); `--match-workers N` sets the number, `1` being one at a time. The result is exactly the same either way,
only the time differs (about 2x faster on a run of 24 clips of 2.5 to 4 minutes).

The defaults are generous (up to 250 results per query and 500 clips), so a big run downloads a lot. Sync stays tractable because each clip is compared against at most `--max-compare` (default 25) clips that are already sorted, biggest group and longest clips first; raise it if clips you expect to match are being left out.

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

- **sync**: GCC-PHAT cross-correlation of the audio. Clips are sorted into groups that line up with each other, all
  growing at the same time. Longest first, each clip is compared with the clips already sorted (the bigger a group,
  the more of the clip's comparisons it gets) and joins the group it matches best, when its best correlation peak is
  `--min-z` standard deviations above the noise; a clip that matches nothing starts a group of its own; a clip that
  matches two groups is the bridge that makes them one. Clips that matched nothing then get a second look at each
  other. The biggest group is fused, so it no longer matters which clip is the longest or which concert it is from. In
  the window, when several groups of three or more clips turn up, you are asked which one to use. Comparisons are
  spread over worker *processes*, not threads: run side by side in one process, the FFT gave a wrong answer about
  once in a hundred comparisons, while separate processes gave exactly the one-at-a-time answers every time.
- **audio**: every 0.5 s block of every clip is scored (clipping, level, agreement with the other clips'
  spectra); the output crossfades toward the best sources instead of summing mics, which would comb-filter.
- **video**: clips are scored for sharpness and exposure; a greedy editor cuts between angles with a minimum
  and maximum shot length. Vertical clips are fitted over a blurred background. Cuts land on exact frame
  boundaries so picture stays locked to the audio.
- Time nobody recorded is skipped in both audio and video.

## Limitations (v1)

- Assumes constant offsets: phone clock drift over a long clip (tens of ms per 10 min) is not corrected.
- YouTube sometimes refuses downloads from a connection ("Sign in to confirm you're not a bot"), typically after many
  requests. Meld says so plainly instead of calling every video unavailable, stops asking after 6 refusals in a row
  and carries on with whatever did download; if nothing did, the run stops with an explanation. It usually passes after
  some hours, or on another network (a phone hotspot often works at once), or with the optional YouTube login above.
  Meld only uses a login if you choose one.
- Clips from different nights of the same tour can sync just as confidently as same-night ones when the show
  plays to backing tracks (seen in testing with Coldplay at Wembley: 17/20/21 Aug clips aligned with the 16 Aug
  ones). `sync` prints each clip's title so you can prune `clips/` and re-run.
- A group is only as connected as its overlaps: two stretches of the same show that no clip bridges stay separate
  groups (a clip needs 5 s of overlap), and only one group is fused into a video.
- 3D reconstruction is experimental and did not work on the real concert footage tried (see above); the fused
  video itself only switches between angles.
- Downloading from YouTube is against its ToS and concert footage is usually copyrighted; keep it personal.

## Tests

```
uv run pytest
```
