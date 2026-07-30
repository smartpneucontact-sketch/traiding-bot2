# SETUP_NORGATE_VM.md — Norgate Data export from a Mac, step by step

Written for someone who has never used Windows or a virtual machine.
Goal: get survivorship-free US stock data out of Norgate (whose updater is
Windows-only) and onto this Mac as frozen parquet files, one time, then
refresh monthly / per-experiment.

Total effort: ~1 afternoon of setup + several hours of unattended downloads.
Everything except clicking through installers is scripted (`norgate_export.py`).

---

## 0. What you are building (30 seconds)

```
Windows 11 VM (inside UTM on this Mac)
  └─ Norgate Data Updater (NDU)  ← downloads the database, must be running
  └─ python norgate_export.py    ← writes parquet to a shared folder
        │
        ▼  (shared folder = a Mac directory the VM can see)
macOS: /Users/arsenkhanguieldyan/Documents/Trading/Traiding 11/data_norgate_raw/
        └─ bars/*.parquet, membership/*.parquet, delistings.parquet, manifest.json
```

The Mac never talks to Norgate directly. It only reads the frozen files.
This is the plan's "one-time export, never a live connection" architecture.

## 1. Prerequisites

- [ ] Norgate Data **US Platinum** subscription active ($346.50/6mo) —
      norgatedata.com. Platinum is required: it is the level with delisted
      securities ("Current & Past" watchlists) and index-constituent history.
- [ ] ~100 GB free disk on the Mac (check: Apple menu → About This Mac →
      Storage). The VM disk takes most of it; the exported parquet is ~2–6 GB.
- [ ] Apple Silicon Mac (M1/M2/M3/M4). NDU is an x64 app; it runs under
      Windows 11 ARM's built-in x64 emulation — community-proven with
      Norgate, though not formally supported by them.

## 2. Install UTM (free virtual machine app)

1. Download UTM from **https://mac.getutm.app** (the free download button —
   the $9.99 Mac App Store version is identical, just a donation).
2. Open the .dmg, drag UTM to Applications, launch it.

## 3. Create the Windows 11 ARM virtual machine

1. In UTM: **Create a New Virtual Machine → Virtualize → Windows**.
2. Tick **"Install Windows 10 or higher"** and **"Download and Mount ISO
   for you"** if offered — recent UTM versions can fetch the Windows 11 ARM
   installer themselves (via the CrystalFetch helper app). If not offered,
   download the ISO with **CrystalFetch** (free, Mac App Store), then point
   UTM at the .iso.
3. Settings for the VM:
   - Memory: **8 GB** (4 GB minimum), CPU cores: 4.
   - Drive size: **80–100 GB** (Norgate's US database + Windows need room;
     the disk file grows as used, it does not take 100 GB immediately).
4. Before first boot, in the VM's settings enable a **Shared Directory**
   → select the repo folder
   `/Users/arsenkhanguieldyan/Documents/Trading/Traiding 11`
   (VirtFS/"Shared Directory" mode is fine; it appears inside Windows as a
   network drive once the guest tools are installed).
5. Boot the VM and click through Windows Setup. Tips for first-timers:
   - If Setup says it needs internet and finds none, the network driver
     is missing: in UTM the installer ISO includes it, or install the
     **SPICE guest tools** (see next step) from the mounted "utm-guest-tools"
     drive; the offline bypass is pressing Shift+F10 and typing
     `OOBE\BYPASSNRO` if it refuses to proceed (well-documented trick).
   - Create a **local account** when possible; no Microsoft 365 needed.
6. Inside Windows, open the mounted **UTM guest tools** drive and run
   `utm-guest-tools-*.exe` — this installs drivers and makes the shared
   directory show up (usually as drive `Z:` or under
   `\\Network\UTM shared`). Verify you can see the Mac's `Traiding 11`
   folder from Windows Explorer before continuing.

## 4. Install Norgate Data Updater (NDU) + first database download

1. In the Windows VM's browser, log in at **norgatedata.com** and download
   the **Norgate Data Updater** installer. Run it (it runs fine under x64
   emulation; if Windows asks, allow it).
2. Launch NDU, log in with the subscription credentials.
3. In NDU's settings/first-run, select the **US Stocks** database including
   **delisted/past** data, and start the initial download.
   **This takes hours (potentially most of a day) and tens of GB** — leave
   the VM running with the Mac plugged in and set Windows/UTM to not sleep.
4. Done when NDU shows the database up to date. NDU must be left RUNNING
   whenever the export script runs — the Python package talks to it locally.
5. Optional but recommended: open the **Norgate Data Viewer** (installed
   with NDU) once and confirm you can chart a delisted symbol (type `SIVB`
   — it should appear with a `-202303`-style suffix).

## 5. Install Python inside the VM

1. In the VM: download Python 3.11+ from **https://python.org/downloads/**
   (the standard Windows 64-bit installer works under emulation).
   IMPORTANT: tick **"Add python.exe to PATH"** on the installer's first
   screen.
2. Open **Command Prompt** (Start menu → type `cmd`) and run:

   ```
   pip install norgatedata pandas pyarrow
   ```

## 6. Run the export

1. Copy `norgate_export.py` from the shared folder to the Windows desktop
   (or run it in place from the share — either works):
   in Windows Explorer, the shared `Traiding 11` folder → copy
   `norgate_export.py` to `C:\Users\<you>\Desktop`.
2. Sanity checks first, in Command Prompt (with NDU running):

   ```
   python Desktop\norgate_export.py --selftest
   python Desktop\norgate_export.py --out "Z:\data_norgate_raw" --limit 25
   ```

   (`Z:` = however the shared `Traiding 11` folder is mounted; adjust.
   The `--limit 25` smoke run should finish in a minute or two and create
   `bars/`, `membership/`, `delistings.parquet`, `manifest.json`.)
   - If it aborts saying an index returned ZERO constituents: open the
     Norgate Data Viewer, find the exact index names (e.g. whether it says
     "S&P MidCap 400" vs "S&P 400"), fix the `INDEX_NAMES` constant at the
     top of `norgate_export.py`, re-run. The candidate names shipped in the
     script must be verified against the Viewer — this check is deliberate.
3. Full export:

   ```
   python Desktop\norgate_export.py --out "Z:\data_norgate_raw"
   ```

   Expect **~11–12k symbols, several hours** (bars pass + membership pass),
   and **~2–6 GB** of parquet. Progress prints every 500 symbols. If the VM
   or script dies mid-run, just re-run the same command — it is
   resume-safe (already-written bar files are skipped; membership,
   delistings and the manifest rebuild at the end). Use `--refresh` only
   when you want to re-pull everything (e.g. the monthly refresh).
4. Because `--out` pointed at the shared folder, the data is already on the
   Mac at:

   ```
   /Users/arsenkhanguieldyan/Documents/Trading/Traiding 11/data_norgate_raw/
   ```

   If you exported to a local Windows path instead, drag the whole
   `data_norgate_raw` folder onto the shared drive now. Integrity is
   verified on the Mac side against `manifest.json`'s per-file sha256s
   (the N0 loader's lossless-transfer gate), so a flaky copy gets caught.

## 7. After the copy — Mac side

- Do NOT rename or edit anything inside `data_norgate_raw/` — the sha256
  manifest is the proof the transfer was lossless.
- The macOS loader (`data_norgate.py`, separate task) reads this tree and
  builds the normalized `data_norgate/` cache with `.meta.json` sidecars.
- Monthly / per-experiment refresh: boot the VM, let NDU update, re-run
  step 6.3 with `--refresh`.

## 8. Licensing — read this once, it matters

Norgate data is licensed to the subscriber's machines only. The exported
parquet **must never leave your machines and must never enter any shared or
public repository**. `data_norgate_raw/` and `data_norgate/` are already in
this repo's `.gitignore` (added together with this guide) — do not remove
those lines, and never force-add (`git add -f`) anything under them. The
license allows 2 machines: the VM (updater) and this Mac (analysis) is the
intended pairing.

## Troubleshooting quick list

| Symptom | Fix |
| --- | --- |
| `norgatedata` import error on the Mac | Expected — the script only runs in the VM (only `--selftest` works on the Mac). |
| Script errors "NDU not running" / connection refused | Start Norgate Data Updater and log in, then re-run. |
| Watchlist returns no symbols | Subscription level: "Current & Past" watchlists need US Platinum; verify the watchlist name spelling in the Viewer. |
| Index gives zero constituents | Fix `INDEX_NAMES` to the Viewer's exact string (step 6.2). |
| VM very slow | Normal under x64 emulation; the export is API/IO-bound and finishes anyway. Give the VM 8 GB RAM. |
| Shared folder not visible in Windows | Install UTM guest tools inside the VM (step 3.6), then reboot the VM. |
