Below is a **project playbook** that matches (a) what you’ve already done, (b) the **actual SMN4Lang dataset conventions**, and (c) the “boxes” architecture you want end-to-end.

I’m going to be **painfully explicit** about: **file locations, timebases, offsets, special-case runs, and what should be saved at each step**—because those are the places these projects usually go off the rails.

---

# 0) Quick status check against SMN4Lang ground truth

Based on your log + what SMN4Lang actually provides:

* ✅ **Good:** You downloaded **stimuli + annotations + preprocessed MEG (sensor-level)** and you’re running theta-band windowed processing.
* ✅ **Good:** You implemented the known **missing start-trigger** fix for `sub-08_run-16` and `sub-09_run-7`. That’s explicitly required. ([GitHub][1])
* ⚠️ **Important simplification:** **Story IDs are consistent with run IDs** (i.e., run-01 corresponds to story 01, etc.). ([PMC][2])
  → Your `run_to_story.csv` from duration-matching is a nice validation tool, but it **shouldn’t be the primary mapping** (duration collisions can happen; run IDs are the ground truth).
* ⚠️ **Critical timing fact:** the alignment timestamps in `stimuli/time_align` have already been shifted **+10.65 s** (because the fMRI scan starts 10.65 s before the audio). ([GitHub][1])
  → If you’re aligning MEG to words using those onset times, you’ll almost certainly want to **subtract 10.65 s** to get “audio-onset time.”
* ⚠️ **Critical MEG fact:** the paper reports (1) **0.1–40 Hz** bandpass applied to their denoised MEG, and (2) a stable **39.5 ms** delay between event timing and actual auditory presentation. ([PMC][2])
  → This affects both “gamma plans” and fine alignment.

---

# 1) Project layout & reproducibility (do this once)

## 1.1 Pin dataset version (you already did)

You’re on ds004078 **v1.2.1**; keep this invariant for the whole project.

**Write this into a project-level `CONFIG.yaml` or `config.json`:**

* dataset: `ds004078`
* version: `1.2.1`
* git commit (if you have the repo): `git rev-parse HEAD`
* code commit: your analysis repo commit hash

## 1.2 Canonical folder conventions (strongly recommended)

Inside your dataset root (or adjacent), create:

```
project_root/
  config/
    config.yaml
  metadata/
    runs_index.csv
    timebase_notes.md
  outputs/
    meg_plv/
    meg_features/
    meg_axes/
    fmri_semspace/
    anchoring/
    llm_adapter/
    steering_eval/
  analysis/
    (your scripts)
```

You can keep scripts inside the dataset folder like you’re doing, but **separating outputs/metadata from raw dataset folders** reduces accidental commits / accidental deletions.

## 1.3 Save “sidecar metadata” for every output artifact

For every `.npy` you write, also write a `.json` with:

* subject/run/story_id
* band definitions
* window_length/window_step
* sfreq / decim
* channel selection details
* timebase definition (see section 3)
* git hash of code
* timestamp

This becomes priceless once you have 1000+ files.

---

# 2) Ground-truth dataset structure you should assume

The dataset and paper describe:

* Subject folders: `sub-*/anat`, `sub-*/dwi`, `sub-*/func`, `sub-*/meg` ([PMC][2])
* Stimulus audio lives in: `stimuli/` ([PMC][2])
* Linguistic resources live in:

  * `stimuli/time_align` (word/char times) with **+10.65 s shift** ([GitHub][1])
  * `stimuli/frequency` ([GitHub][1])
  * `stimuli/embeddings` (Word2Vec, BERT, GPT2 embeddings) ([GitHub][1])
  * `stimuli/syntactic_annotations` ([GitHub][1])
* Preprocessed neuroimaging is in: `derivatives/preprocessed_data` (MEG sensor-level; fMRI in CIFTI & MNI) ([PMC][2])

> If your on-disk v1.2.1 layout differs slightly (it happens), treat the above as conceptual truth and just “find the actual paths once,” then bake them into config.

---

# 3) The “single timebase” system (most important part)

You have **three time concepts** in this project:

1. **MEG sample time** (in seconds from MEG file start)
2. **Audio-onset time** (seconds from the moment the story audio actually starts)
3. **fMRI scan time** (seconds from scan start; audio starts 10.65 s later)

SMN4Lang gives you key constants:

* The pre-audio screen sequence is **8 s blank + 2.65 s instruction = 10.65 s**, explaining the fMRI/audio offset. ([PMC][2])
* `stimuli/time_align` onset/offset times are already **increased by 10.65 s** to align with fMRI image time. ([GitHub][1])
* MEG preprocessing included bandpass 0.1–40 Hz and an **auditory delivery delay of 39.5 ms** between event timing and actual audio reaching participant. ([PMC][2])

## 3.1 Define canonical conversion functions (write these into `timebase_notes.md`)

I recommend you adopt **audio-onset time as the canonical semantic time**:

### A) Convert time_align timestamps to audio-onset time

`stimuli/time_align` provides times in fMRI scan time:

[
t_{\text{audio}} = t_{\text{align}} - 10.65
]
because times were shifted +10.65 s. ([GitHub][1])

### B) Convert MEG sample time to audio-onset time

Let:

* `t_meg` = MEG seconds from beginning of MEG file
* `t_trig_start` = time of stimulus-start trigger (in MEG seconds)
* `delay_audio` = 0.0395 s (optional correction)

Then:
[
t_{\text{audio}} = t_{\text{meg}} - (t_{\text{trig_start}} + \text{delay_audio})
]
Use `delay_audio = 0.0395` if you want “audio at ear” alignment. ([PMC][2])

### C) Convert audio-onset time to fMRI scan time (for anchoring)

[
t_{\text{fmri}} = t_{\text{audio}} + 10.65
]

## 3.2 The two special MEG runs with missing start triggers

For:

* `sub-08_run-16`
* `sub-09_run-7`

The dataset states stimulus-start triggers weren’t recorded; the first trigger is the stimulus-end trigger, and:

[
t_{\text{trig_start}} = t_{\text{trig_end}} - \text{duration(audio)}
]
([GitHub][1])

You already implemented this—keep it in the canonical run table (next section).

---

# 4) Build a single “runs index” table (the spine of everything)

Create `metadata/runs_index.csv` with one row per `(subject, run)`.

Minimum columns:

### Identity

* `subject` (01–12)
* `run` (01–60)
* `story_id` (01–60) **(default: story_id = run)** ([PMC][2])

### Files

* `meg_preproc_path` (preferred; sensor-level)
* `meg_raw_path` (optional)
* `audio_path` (`stimuli/...wav`)
* `time_align_path` (word/char timing)
* `embeddings_path` (for GPT2/BERT/Word2Vec)

### Timing

* `sfreq`
* `trig_start_sample` (or `trig_start_time_s`)
* `trig_end_sample` (needed for the missing-trigger cases)
* `audio_duration_s`
* `auditory_delay_s = 0.0395` (constant; document if you apply it) ([PMC][2])
* `notes` (e.g., “missing start trigger, computed from end trigger”)

### Validation-only (optional)

* `duration_match_diff_s` (your duration-matching residual)
* `mapping_confidence`

**Why this matters:** every downstream script should only need:

* `(subject, run)` → row in this table
  No more “special-case hacks” scattered across code.

---

# 5) BOX 1 — MEG → PLV Brain Atlas (core object)

You want:

* **PLV(t) ∈ ℝᴰ** time-indexed brain state vectors
* plus “network-level PLV” and band specificity

## 5.1 Decide what “D” is (don’t skip this decision)

Sensor-level PLV over 306 sensors is *huge* if you store full edges per window.

Pick one of these “D definitions”:

### Option A (fast, scalable): **node-wise synchrony vector**

For each window, compute per-sensor synchrony summary:

* (D = N_{\text{sensors}}) (e.g., mean PLV of each sensor to all others)

Pros: cheap storage, stable; still captures global geometry.
Cons: loses pairwise detail.

### Option B (rich, heavy): **edge vector**

* (D = N(N-1)/2) (≈ 46k edges for N=306)

Pros: maximal info.
Cons: big compute, big storage; more leakage artifacts.

### Option C (best compromise for “atlas”): **compressed edge space**

Compute edge vector internally, but store only:

* top PCs (e.g., 50–200 dims)
* or graph embeddings / diffusion-map coordinates

Pros: keeps geometry; makes axis discovery and LLM adapter much easier.

> For your later LLM adapter (Box 5), Option C is usually the sweet spot.

## 5.2 Band definitions (match the dataset’s own band conventions)

The paper’s MEG validation uses:

* delta: 1–4 Hz
* theta: 4–8 Hz
* alpha: 8–13 Hz
* beta: 13–30 Hz ([PMC][2])

They also report preprocessing bandpass **0.1–40 Hz**, meaning “gamma” above 40 isn’t available from their preprocessed MEG. ([PMC][2])

So your **Box 1** should be explicitly:

* delta/theta/alpha/beta (and optionally 30–40 “low gamma”)
* **high gamma requires reprocessing raw MEG** (different branch)

## 5.3 Windowing rules (make them band-aware)

Your current theta setup (2.0s window, 0.5s step) is reasonable.

General rule:

* choose window length as **~6–12 cycles per band**

Examples:

* theta (4–8 Hz): 1.5–3 s
* alpha (8–13 Hz): 1–2 s
* beta (13–30 Hz): 0.5–1 s

Keep the step at 25–50% of window length unless you have a reason not to.

## 5.4 Channel selection (avoid silent mixing artifacts)

Elekta Neuromag has magnetometers + gradiometers. Decide and document:

* use only gradiometers, or only magnetometers, or compute separately and concatenate.

Mixing them without scaling can bias PLV geometry.

## 5.5 PLV computation (stable definition)

For a window with analytic phases (\phi_i(t)):

[
\mathrm{PLV}_{ij} = \left|\frac{1}{T}\sum_t e^{i(\phi_i(t)-\phi_j(t))}\right|
]

Implementation tips:

* bandpass filter → Hilbert transform → phase extraction
* if you decimate: decimate **after** lowpass/bandpass to avoid aliasing
* store float32

## 5.6 Leakage / field spread guardrail (critical if you publish)

Plain PLV is susceptible to spurious synchrony due to field spread/common sources.

You have two reasonable paths:

* **Sensor-space + leakage-robust metric**: wPLI / imaginary coherence style measures (not PLV proper)
* **Source-space PLV with leakage correction** (longer-term path requiring anat + forward model)

You can keep PLV for your internal axis discovery now, but I’d bake a note into the guide: “final claims should use leakage-robust connectivity.”

## 5.7 Outputs for Box 1 (what to save)

Per `(subject, run, band)` save:

1. `plv_state_windows.npy`

* shape depends on your “D choice”
* plus `window_centers_audio_s.npy` (time vector in audio-onset seconds)

2. `plv_qc.json`

* channel counts, dropped channels
* sfreq, effective sfreq after decim
* basic stats (mean PLV, variance)

If you want to keep storage small:

* store compressed coordinates: `plv_pca_coords.npy` and PCA basis in `plv_pca_basis.npy`

---

# 6) BOX 3 — Semantic Annotation Layer (S(t), no manual reading)

Your mantra: **language is handled here; you don’t manually interpret it.**

## 6.1 Use what the dataset already gives you

SMN4Lang provides:

* word/character times in `stimuli/time_align` (+10.65 already applied) ([GitHub][1])
* frequencies in `stimuli/frequency` ([GitHub][1])
* embeddings (Word2Vec, BERT, GPT2) in `stimuli/embeddings` ([GitHub][1])
* syntactic annotations in `stimuli/syntactic_annotations` ([GitHub][1])

## 6.2 Convert everything to your window grid

For each run, you already have:

* window centers `t_audio[k]`

For each semantic signal, produce a vector:

* `S_k[k] = semantic_value_at_window_k`

Standard aggregation:

* mean/median of token-level values whose onset/offset overlaps window
* optionally weighted by overlap duration

**Important:** before you join with MEG, convert `time_align` times:

* `t_audio = t_align - 10.65` ([GitHub][1])

## 6.3 Feature families that are “cheap and powerful”

Start with a handful that are robust and interpretable:

### Embedding dynamics

* `embedding_change`: L2 distance between consecutive contextual embeddings (your current feature)
* `embedding_PC1..PCk`: principal components of embeddings across story
* `semantic_shift_rate`: smoothed derivative magnitude of embedding trajectory

### Lexical predictability proxies (if you don’t compute true surprisal yet)

* frequency (word/char log freq)
* contextual novelty (distance from running mean embedding)

### Syntax

* constituency tree depth
* dependency length / arc count
* POS entropy per window (distribution over tags)

### Discourse boundaries

* boundary density: number of word boundaries per second
* pause duration proxies from audio (optional)

## 6.4 True information-theoretic signals (surprisal/entropy)

Embeddings alone don’t give you surprisal. If you want:

* surprisal(t)
* entropy(t)

you must run a language model to get token probabilities.

This can be a separate step later. For now, your embedding-change proxy is a reasonable “semantic update signal.”

## 6.5 Outputs for Box 3

Per `(run)` save:

* `semantic_features_windows.npy` (shape `[n_windows, K]`)
* `feature_names.json`
* `window_centers_audio_s.npy` (same as MEG)
* plus optional diagnostic: coverage (% windows with >=1 token)

---

# 7) BOX 4 — Axis Discovery (where “more axes” come from)

This is the box that turns:

* PLV atlas coordinates
* semantic signals
  into many reliable axes.

## 7.1 Define an “axis object” explicitly

For each axis (a_k), store:

* `axis_name`
* `band`
* `plv_space_definition` (node-wise / edges / PCA-space)
* `a_k` vector (direction)
* `thresholds` (how you defined high/low)
* `train_split` description
* reliability metrics

## 7.2 How to compute an axis (robust version)

For a given semantic feature (S_k(t)):

1. **Choose training windows**

* exclude the first/last few seconds to avoid filter edge effects
* optionally exclude windows with no tokens

2. **(Recommended) residualize confounds**
   Before splitting high/low, regress out:

* audio envelope energy (if you include it)
* boundary density / word rate

3. **Define high vs low**

* top and bottom quantiles (e.g., 20% / 20%)
* or z-thresholding

4. **Compute axis direction**
   [
   a_k = \mathbb{E}[\mathrm{PLV}\mid S_k \text{ high}] - \mathbb{E}[\mathrm{PLV}\mid S_k \text{ low}]
   ]

5. **Fix the sign**
   Define sign so that:
   [
   \mathrm{corr}(\langle \mathrm{PLV}(t), a_k\rangle, S_k(t)) > 0
   ]

6. **Compute axis coordinate time series**
   [
   z_k(t)=\langle \mathrm{PLV}(t), a_k\rangle
   ]
   Save `z_k(t)` per run.

## 7.3 Validation (don’t skip)

### Within-subject

* split-half across stories (train on 30 stories, test on 30)
* reliability: correlation of axis vectors or of projected time series

### Across-subject

* compute (a_k^{(s)}) per subject
* assess cosine similarity across subjects (after sign alignment)

### Null test

* permute time windows within run (blockwise) → should destroy effect

## 7.4 Practical note about your current script outputs

Right now you’re writing `sub-01_run-1_theta_axis.npy`, etc.

Two acceptable interpretations:

* **Interpretation A:** That file is the *axis direction* (a_k) computed within that run.
* **Interpretation B:** That file is the *axis coordinate time series* (z_k(t)).

In either case: add a sidecar JSON to disambiguate and record shapes:

* if it’s a vector of length D → it’s (a_k)
* if it’s length `n_windows` → it’s (z_k(t))

---

# 8) BOX 2 — fMRI → Semantic Brain Space (anchor, not optimizer)

This box is “slow, spatial, integrative.”

SMN4Lang provides **preprocessed fMRI in CIFTI and MNI** under derivatives. ([PMC][2])
(They used HCP minimal preprocessing pipelines.) ([GitHub][1])

## 8.1 Decide your representation: CIFTI vs MNI

* **CIFTI** is great if you want surface+subcortex in standard space and plan cross-subject alignment.
* **MNI volumetric** is simpler if your tooling is voxel-based.

If you want a robust semantic space, CIFTI is usually nicer.

## 8.2 Align fMRI time to stimulus

fMRI is slow and delayed; you need one of:

### Option A: SRM/hyperalignment on raw story-evoked time series

* align subjects by shared response (stimulus-locked signals)
* then reduce dimensionality

This gives you a latent “semantic response space” without committing to a specific semantic model.

### Option B: Build a semantic design matrix and derive fMRI semantic coords

* build (X(t)) from embeddings/features
* convolve with HRF
* fit encoding model
* use learned low-dimensional components as coordinates

This is closer to “semantic coordinates,” but it uses language features directly.

Given your “anchor, not optimizer” framing, Option A is truest.

## 8.3 Output for Box 2

Per story/run, save:

* `fmri_semantics_coords.npy` shape `[n_TR, d]`
* `TR_times_fmri_s.npy` (time from scan start)
* `alignment_model.pkl` (if SRM/hyperalignment)
* `d` chosen via held-out reconstruction performance

---

# 9) Box 4 (continued): use fMRI space to anchor/select MEG axes

Now you have:

* MEG axis coordinates (z_k(t)) at fast sampling (windowed)
* fMRI semantic coords (y(t)) at TR sampling

## 9.1 Put MEG and fMRI on the same sampling grid

For each story:

1. Convert MEG axis coordinate time series into **fMRI timebase**

   * `t_fmri = t_audio + 10.65`
2. Downsample MEG axis coordinates into TR bins:

   * average (z_k(t)) in each TR window
3. Optionally apply an HRF-like smoothing to MEG or allow lags.

## 9.2 Anchoring tests (what “anchoring” should mean)

For each axis (k):

* does (z_k(\text{TR})) correlate with one or more dimensions of (y(\text{TR}))?
* can you predict (y(\text{TR})) from the set of MEG axes with a simple linear model?
* does this relationship generalize across stories and subjects?

Axes that are reliable in MEG *and* relate consistently to fMRI semantic space are your “kept axes.”

---

# 10) BOX 5 — LLM Projection Adapter (projection, not optimization)

Goal: frozen LLM states → brain axes (no LLM training).

## 10.1 Choose the modeling unit and align to time

You need a mapping:

* LLM hidden states (h(t))
* brain axis coords (z(t)) (MEG-derived, possibly anchored)

Key alignment challenge: text tokens vs audio-aligned words/characters.

**The dataset gives character- and word-level timestamps.** ([GitHub][1])
So you can do:

* **Character-time alignment → token alignment**

  * tokenize the transcript
  * map tokens to spans of characters
  * assign token timestamps by aggregating character timestamps

Then:

* aggregate token hidden states into your MEG window grid (same window centers)

## 10.2 Adapter form (keep it simple)

Start with ridge regression:

[
\hat{z}(t) = W h(t) + b
]

Train on:

* some stories (train set)
  Test on:
* held-out stories (generalization)

No fine-tuning LLM weights.

## 10.3 Outputs

* `adapter_W.npy`, `adapter_b.npy`
* train/test metrics per axis
* story-level generalization plots

---

# 11) BOX 6 — Reading + Steering

## 11.1 Reading

Given a new text or transcript:

* run LLM, extract hidden states
* project to brain axes via adapter
* you now have a “brain-axis trajectory” of the model

Compare to humans:

* compare model trajectory to MEG axis trajectory for the same story
* compare distributions and event-locked responses (boundaries, surprises)

## 11.2 Steering (mathematically clean way with a linear adapter)

If:
[
z = W h + b
]

To increase axis (k), you want to add (\Delta h) that increases (z_k):

* gradient direction is (W_{k:}^\top)

A practical steering update:
[
h' = h + \alpha \cdot \frac{W_{k:}^\top}{|W_{k:}|}
]

Or, if you want to impose a multi-axis change (\Delta z), use pseudoinverse:
[
\Delta h = W^{+}\Delta z
]

Then inject (\Delta h) into:

* specific layers
* specific token positions (e.g., at sentence boundaries)

## 11.3 Evaluation suite (don’t skip)

For steering, you need to measure:

* semantic shift along target axis (your own classifier or embedding-based score)
* fluency / perplexity change
* “axis specificity” (other axes shouldn’t drift too much)
* human-judged coherence (optional)

---

# 12) Your pipeline, rewritten as an actionable step-by-step “runbook”

Below is the exact “do this, then this” sequence I’d recommend **from where you are now**.

## Step A — Build/lock `runs_index.csv`

1. Create rows for all `(sub, run)`.
2. Set `story_id = run` by default (ground truth). ([PMC][2])
3. For each run:

   * record audio path, audio duration
   * extract MEG trigger start/end samples
   * apply missing-trigger fix for the two runs ([GitHub][1])
4. Add `auditory_delay_s = 0.0395` (document if applied). ([PMC][2])
5. Compute your duration-matching diff and store as validation only.

**Deliverable:** `metadata/runs_index.csv` + `metadata/timebase_notes.md`

## Step B — Standardize a single “window grid generator”

Given `(band, window_length, window_step)`:

* generate window centers in **audio-onset seconds**
* store as `window_centers_audio_s.npy` per run

**Deliverable:** `outputs/meg_plv/windows/sub-XX_run-YY_band-..._times.npy`

## Step C — MEG PLV state extraction (Box 1)

For each run:

1. load preprocessed sensor-level MEG
2. apply band filter if you aren’t using band-specific prefiltered signals
3. (optional) decimate (theta can tolerate; paper used downsample to 50 Hz for entrainment analysis) ([PMC][2])
4. compute PLV per window
5. store either:

   * PLV(t) vectors, or
   * compressed PLV(t) coordinates

**Deliverable:** `outputs/meg_plv/plv_state_*.npy` (+ sidecars)

## Step D — Semantic features (Box 3)

For each run:

1. load `stimuli/time_align` (remember: subtract 10.65 to get audio time) ([GitHub][1])
2. load embeddings from `stimuli/embeddings` ([GitHub][1])
3. compute features (embedding_change, etc.)
4. aggregate to your window grid

**Deliverable:** `outputs/meg_features/features_*.npy` (+ feature_names.json)

## Step E — Axis discovery (Box 4, MEG-only first)

For each feature (S_k):

1. compute axis direction (a_k) on training stories
2. compute projected coordinate (z_k(t)) on held-out stories
3. compute reliability and keep only stable axes

**Deliverable:** `outputs/meg_axes/axis_<name>_<band>.npy` + `axis_metrics.json`

## Step F — Build fMRI semantic brain space (Box 2)

1. download preprocessed fMRI (`derivatives/preprocessed_data/.../func`) when ready ([PMC][2])
2. compute shared latent space (SRM/hyperalignment)
3. save `fmri_semantics(t)`

**Deliverable:** `outputs/fmri_semspace/*.npy`

## Step G — Anchor MEG axes to fMRI space

1. downsample MEG axis coordinate to TR
2. quantify alignment (correlation/CCA/RSA)
3. refine axis set

**Deliverable:** `outputs/anchoring/axis_selection.json`

## Step H — Train LLM projection adapter (Box 5)

1. compute LLM hidden state windows (h(t))
2. fit (W) mapping to brain axes
3. evaluate held-out stories

**Deliverable:** `outputs/llm_adapter/W.npy`, metrics

## Step I — Reading + steering (Box 6)

1. reading: project model trajectories
2. steering: inject Δh and evaluate

**Deliverable:** `outputs/steering_eval/*`

---

# 13) Two targeted improvements to your current run (low effort, high payoff)

## 13.1 Replace duration-based run→story mapping with ID-based mapping + duration check

Given story IDs are consistent with run IDs, treat:

* `story_id = run_id` as the mapping ([PMC][2])
  and keep your duration matcher as:
* a QC measure to flag runs (like the two missing-trigger runs) rather than as the mapping itself.

## 13.2 Add the “10.65 / 39.5ms” timing decisions into code as explicit flags

In your CLI, make these parameters explicit:

* `--align-timebase {audio, fmri}`
* `--apply-auditory-delay {0,1}` (0.0395 s) ([PMC][2])
* `--time_align_shift_s 10.65` (subtract when going to audio time) ([GitHub][1])

Then every output knows exactly what timeline it lives on.


[1]: https://github.com/OpenNeuroDatasets/ds004078 "GitHub - OpenNeuroDatasets/ds004078: OpenNeuro dataset available at https://openneuro.org/datasets/ds004078"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9525723/ "
            A synchronized multimodal neuroimaging dataset for studying brain language processing - PMC
        "
