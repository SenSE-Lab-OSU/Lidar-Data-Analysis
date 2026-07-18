#!/usr/bin/env python3
"""
LiDAR-Camera Calibration Suite — unified interactive launcher
================================================================
A single entry point that wraps three companion tools:

  manual_calibrate.py      interactive point-picking + solver
  evaluate_calibration.py  reprojection-error evaluation of one or more transforms
  visualize_calibration.py projects a transform onto an image / point cloud

This launcher is a text-based menu (TUI) — the only GUI elements are the
optional native file/folder picker dialogs used when choosing files (a
text-based file browser is used automatically as a fallback when no
graphical dialog is available).

Run with no arguments and follow the prompts:

    python calib_suite.py

A session JSON file (default ./calib_session.json) records your data
folder, every transform and correspondences file you've registered, and
recent history, so you can quit at any time and resume later with the
same session file.

Requirements
------------
manual_calibrate.py, evaluate_calibration.py, and visualize_calibration.py
must live in the same folder as this script (or pass --scripts-dir).

Arguments
---------
This tool is fully interactive; the only supported flags are:

  --session PATH       Session JSON file to load/create (skips the startup prompt)
  --scripts-dir DIR     Folder containing the three companion scripts
                        (default: the folder this script lives in)
  -h, --help            Show this help and exit
"""

import argparse
import datetime
import importlib.util
import json
import os
import sys
import traceback

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# Loading the three companion tools as library modules
# ══════════════════════════════════════════════════════════════════════════

REQUIRED_SCRIPTS = [
    'manual_calibrate.py',
    'evaluate_calibration.py',
    'visualize_calibration.py',
]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tools(scripts_dir):
    missing = [f for f in REQUIRED_SCRIPTS
               if not os.path.isfile(os.path.join(scripts_dir, f))]
    if missing:
        print('ERROR: could not find the following required script(s) in:')
        print(f'  {scripts_dir}\n')
        for m in missing:
            print(f'  missing: {m}')
        print('\nPlace manual_calibrate.py, evaluate_calibration.py, and '
              'visualize_calibration.py next to this script, or pass '
              '--scripts-dir /path/to/folder.')
        sys.exit(1)

    try:
        manual = _load_module(os.path.join(scripts_dir, 'manual_calibrate.py'),
                              'manual_calibrate')
        evaluate = _load_module(os.path.join(scripts_dir, 'evaluate_calibration.py'),
                                'evaluate_calibration')
        visualize = _load_module(os.path.join(scripts_dir, 'visualize_calibration.py'),
                                 'visualize_calibration')
    except ImportError as e:
        print(f'ERROR: a required Python package is missing: {e}')
        print('Make sure numpy, opencv-python (cv2) and scipy are installed '
              '(matplotlib is optional, needed only for evaluate plots).')
        sys.exit(1)

    return {'manual': manual, 'evaluate': evaluate, 'visualize': visualize}


# ══════════════════════════════════════════════════════════════════════════
# Session persistence
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_SESSION_PATH = './calib_session.json'


def now_iso():
    return datetime.datetime.now().isoformat(timespec='seconds')


def new_session(path):
    return {
        'session_file': os.path.abspath(path),
        'created': now_iso(),
        'updated': now_iso(),
        'data_dir': None,
        'output_root': None,
        'correspondence_sets': {},   # label -> {path, data_dir, added}
        'active_correspondences': None,
        'transforms': {},            # label -> {path, source, added}
        'last_dirs': {},             # category -> last browsed directory
        'last_evaluate': None,
        'last_visualize': None,
        'history': [],
    }


def load_session(path):
    with open(path) as f:
        data = json.load(f)
    data.setdefault('correspondence_sets', {})
    data.setdefault('active_correspondences', None)
    data.setdefault('transforms', {})
    data.setdefault('last_dirs', {})
    data.setdefault('history', [])
    data.setdefault('output_root', None)
    data.setdefault('data_dir', None)
    data['session_file'] = os.path.abspath(path)
    return data


def save_session(session):
    session['updated'] = now_iso()
    path = session['session_file']
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(session, f, indent=2)
    os.replace(tmp, path)


def log_event(session, action, **details):
    session['history'].append({'time': now_iso(), 'action': action, **details})


def get_output_root(session):
    if not session.get('output_root'):
        base = os.path.dirname(session['session_file'])
        stem = os.path.splitext(os.path.basename(session['session_file']))[0]
        session['output_root'] = os.path.join(base, f'{stem}_outputs')
    os.makedirs(session['output_root'], exist_ok=True)
    return session['output_root']


def _unique_label(existing, base):
    base = base or 'item'
    if base not in existing:
        return base
    i = 2
    while f'{base}_{i}' in existing:
        i += 1
    return f'{base}_{i}'


# ══════════════════════════════════════════════════════════════════════════
# Text-mode prompt helpers
# ══════════════════════════════════════════════════════════════════════════

def prompt_text(msg, default=None):
    suffix = f' [{default}]' if default not in (None, '') else ''
    val = input(f'{msg}{suffix}: ').strip()
    return val or (default or '')


def prompt_yes_no(msg, default=True):
    d = 'Y/n' if default else 'y/N'
    val = input(f'{msg} ({d}): ').strip().lower()
    if not val:
        return default
    return val in ('y', 'yes')


def prompt_int(msg, default=None, min_v=None, max_v=None):
    while True:
        suffix = f' [{default}]' if default is not None else ''
        raw = input(f'{msg}{suffix}: ').strip()
        if not raw and default is not None:
            return default
        try:
            v = int(raw)
        except ValueError:
            print('  Please enter a whole number.')
            continue
        if min_v is not None and v < min_v:
            print(f'  Must be >= {min_v}')
            continue
        if max_v is not None and v > max_v:
            print(f'  Must be <= {max_v}')
            continue
        return v


def prompt_menu(title, options, allow_cancel=True, cancel_label='Back'):
    """options: list of (label, description). Returns 0-based index or None."""
    print(f'\n{title}')
    for i, (label, desc) in enumerate(options, start=1):
        tail = f'  — {desc}' if desc else ''
        print(f'  [{i}] {label}{tail}')
    if allow_cancel:
        print(f'  [0] {cancel_label}')
    while True:
        raw = input('  > ').strip()
        if not raw:
            continue
        if allow_cancel and raw == '0':
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx - 1
        print('  Invalid choice, try again.')


def multi_select(title, items, preselected=None):
    """items: list of (label, sublabel). Returns sorted list of selected
    indices, or None if the user cancelled."""
    selected = set(preselected or [])
    while True:
        print(f'\n{title}')
        for i, (label, sub) in enumerate(items, start=1):
            box = '[x]' if (i - 1) in selected else '[ ]'
            extra = f'  ({sub})' if sub else ''
            print(f'  {box} {i}. {label}{extra}')
        print("  Type numbers to toggle (e.g. '1 3 4'), 'a' = all, 'n' = none,")
        print("  'd' = done, 'q' = cancel.")
        raw = input('  > ').strip().lower()
        if raw == 'q':
            return None
        if raw == 'd':
            if selected:
                return sorted(selected)
            print('  Select at least one item first (or q to cancel).')
            continue
        if raw == 'a':
            selected = set(range(len(items)))
            continue
        if raw == 'n':
            selected = set()
            continue
        toks = raw.replace(',', ' ').split()
        idxs, valid = [], bool(toks)
        for t in toks:
            if not t.isdigit() or not (1 <= int(t) <= len(items)):
                valid = False
                break
            idxs.append(int(t) - 1)
        if not valid:
            print('  Invalid input.')
            continue
        for idx in idxs:
            if idx in selected:
                selected.remove(idx)
            else:
                selected.add(idx)


# ══════════════════════════════════════════════════════════════════════════
# File / directory picker — native dialog with a text-browser fallback
# ══════════════════════════════════════════════════════════════════════════

def _tk_pick(mode, title, initialdir=None, filetypes=None):
    """Returns the chosen path, '' if the user cancelled the dialog, or the
    sentinel 'no_gui' if no graphical dialog is available at all."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return 'no_gui'
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        kwargs = {'title': title}
        if initialdir and os.path.isdir(initialdir):
            kwargs['initialdir'] = initialdir
        if mode == 'dir':
            result = filedialog.askdirectory(**kwargs)
        else:
            if filetypes:
                kwargs['filetypes'] = filetypes
            result = filedialog.askopenfilename(**kwargs)
        root.destroy()
        return result or ''
    except Exception:
        return 'no_gui'


def _text_pick(mode, title, initialdir=None, filetypes=None):
    cur = os.path.abspath(initialdir or os.getcwd())
    if not os.path.isdir(cur):
        cur = os.getcwd()

    ext_filter = None
    if filetypes:
        exts = set()
        for _, pattern in filetypes:
            for tok in pattern.replace(';', ' ').split():
                if tok.startswith('*.'):
                    exts.add(tok[1:].lower())
        ext_filter = exts or None

    while True:
        print(f'\n── {title} ──')
        print(f'  Current directory: {cur}')
        try:
            entries = sorted(os.listdir(cur))
        except OSError as e:
            print(f'  Cannot list directory: {e}')
            entries = []
        dirs = [e for e in entries
                if os.path.isdir(os.path.join(cur, e)) and not e.startswith('.')]
        files = [e for e in entries
                 if os.path.isfile(os.path.join(cur, e)) and not e.startswith('.')]
        if mode == 'file' and ext_filter:
            files = [f for f in files if os.path.splitext(f)[1].lower() in ext_filter]

        listing = [('..', 'dir')] + [(d, 'dir') for d in dirs] + [(f, 'file') for f in files]
        for i, (name, kind) in enumerate(listing, start=1):
            marker = '/' if kind == 'dir' else ''
            print(f'    [{i:>3}] {name}{marker}')

        print("  Enter a number to navigate/select, '.' to choose THIS directory,")
        print("  a path to type it directly, or 'q' to cancel.")
        choice = input('  > ').strip()

        if choice.lower() == 'q':
            return None
        if choice == '.':
            if mode == 'dir':
                return cur
            print('  This selection requires a file, not a directory.')
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(listing):
                name, kind = listing[idx]
                target = os.path.dirname(cur) if name == '..' \
                    else os.path.abspath(os.path.join(cur, name))
                if kind == 'dir':
                    cur = target
                    continue
                if mode == 'dir':
                    print("  That's a file, not a directory.")
                    continue
                return target
            print('  Invalid selection.')
            continue

        expanded = os.path.expanduser(choice)
        typed = expanded if os.path.isabs(expanded) else os.path.abspath(os.path.join(cur, expanded))
        if os.path.isdir(typed):
            if mode == 'dir':
                return typed
            cur = typed
            continue
        if os.path.isfile(typed):
            if mode == 'file':
                return typed
            print("  That's a file, not a directory.")
            continue
        print('  Path not found.')


def pick_path(session, category, mode, title, filetypes=None):
    """mode: 'file' or 'dir'. Remembers the last directory used per category."""
    initialdir = session['last_dirs'].get(category) or session.get('data_dir') or os.getcwd()
    outcome = _tk_pick(mode, title, initialdir, filetypes)
    if outcome == 'no_gui':
        result = _text_pick(mode, title, initialdir, filetypes)
    elif outcome == '':
        print('  Dialog cancelled.')
        result = _text_pick(mode, title, initialdir, filetypes) \
            if prompt_yes_no('Use the text-based browser instead?', default=True) else None
    else:
        result = outcome

    if result:
        session['last_dirs'][category] = result if mode == 'dir' else os.path.dirname(result)
    return result


# ══════════════════════════════════════════════════════════════════════════
# Registry helpers (transforms / correspondence sets)
# ══════════════════════════════════════════════════════════════════════════

def _remove_item(d, kind):
    if not d:
        print(f'  No {kind}s registered.')
        return None
    labels = list(d.keys())
    idx = prompt_menu(f'Remove which {kind}?',
                      [(l, d[l]['path']) for l in labels], cancel_label='Cancel')
    if idx is None:
        return None
    label = labels[idx]
    if prompt_yes_no(f"Remove '{label}'? (the file on disk is not deleted)", default=False):
        del d[label]
        print(f"  Removed '{label}'.")
        return label
    return None


def _rename_item(d, kind):
    if not d:
        print(f'  No {kind}s registered.')
        return
    labels = list(d.keys())
    idx = prompt_menu(f'Rename which {kind}?',
                      [(l, d[l]['path']) for l in labels], cancel_label='Cancel')
    if idx is None:
        return
    old = labels[idx]
    new = prompt_text('New label', default=old)
    new = _unique_label({k: v for k, v in d.items() if k != old}, new)
    d[new] = d.pop(old)
    print(f"  Renamed '{old}' -> '{new}'.")


def _select_correspondences(session):
    sets = session['correspondence_sets']
    if not sets:
        print('  No correspondences registered yet.')
        if not prompt_yes_no('Browse for a correspondences.json now?', default=True):
            return None
        path = pick_path(session, 'correspondences', 'file',
                         'Select correspondences.json', filetypes=[('JSON', '*.json')])
        if not path:
            return None
        label = _unique_label(sets, os.path.splitext(os.path.basename(path))[0])
        sets[label] = {'path': path, 'data_dir': None, 'added': now_iso()}
        session['active_correspondences'] = label
        return label

    labels = list(sets.keys())
    options = [(lbl, sets[lbl]['path']) for lbl in labels]
    options.append(('Browse for a different file...', ''))
    idx = prompt_menu('Select a correspondences set:', options)
    if idx is None:
        return None
    if idx == len(labels):
        path = pick_path(session, 'correspondences', 'file',
                         'Select correspondences.json', filetypes=[('JSON', '*.json')])
        if not path:
            return None
        label = _unique_label(sets, os.path.splitext(os.path.basename(path))[0])
        sets[label] = {'path': path, 'data_dir': None, 'added': now_iso()}
        session['active_correspondences'] = label
        return label
    label = labels[idx]
    session['active_correspondences'] = label
    return label


def _select_transform(session, single_prompt='Select a transform:'):
    transforms = session['transforms']
    if not transforms:
        print('  No transforms registered yet.')
        if not prompt_yes_no('Register one now?', default=True):
            return None
        action_import_transform(session)
        if not session['transforms']:
            return None
        transforms = session['transforms']

    labels = list(transforms.keys())
    options = [(lbl, transforms[lbl]['path']) for lbl in labels]
    options.append(('Browse for a different file...', ''))
    idx = prompt_menu(single_prompt, options)
    if idx is None:
        return None
    if idx == len(labels):
        before = set(session['transforms'])
        action_import_transform(session)
        added = list(set(session['transforms']) - before)
        return added[0] if added else None
    return labels[idx]


# ══════════════════════════════════════════════════════════════════════════
# Header / status
# ══════════════════════════════════════════════════════════════════════════

def print_header(session):
    print('\n' + '=' * 70)
    print('  LiDAR-Camera Calibration Suite')
    print('=' * 70)
    print(f"  Session file        : {session['session_file']}")
    print(f"  Data directory      : {session['data_dir'] or '(none)'}")
    n_corr = len(session['correspondence_sets'])
    active = session['active_correspondences']
    corr_line = f'  Correspondence sets : {n_corr}'
    if active:
        corr_line += f'  (active: {active})'
    print(corr_line)
    print(f"  Registered transforms: {len(session['transforms'])}")
    print('=' * 70)


# ══════════════════════════════════════════════════════════════════════════
# Actions
# ══════════════════════════════════════════════════════════════════════════

def action_new_calibration(session, mods):
    mc = mods['manual']
    print('\n── New calibration from a data folder ──')
    print('Expected layout:  <data-dir>/camera/cam_N.png  and  <data-dir>/lidar/lidar_N.npy')

    reuse = session.get('data_dir') and prompt_yes_no(
        f"Use previously used data folder ({session['data_dir']})?", default=True)
    if reuse:
        data_dir = session['data_dir']
    else:
        data_dir = pick_path(session, 'data_dir', 'dir',
                             'Select data directory (contains camera/ and lidar/)')
        if not data_dir:
            print('  Cancelled.')
            return

    try:
        pairs_meta = mc.discover_pairs(data_dir)
    except FileNotFoundError as e:
        print(f'  ERROR: {e}')
        return

    print(f"  Found {len(pairs_meta)} matched pair(s): "
          + ', '.join(p['label'] for p in pairs_meta))
    if not prompt_yes_no('Proceed to load these pairs and launch the picking UI?', default=True):
        return

    pairs = []
    for meta in pairs_meta:
        print(f"Loading {meta['label']} ...")
        img = mc.load_camera(meta['cam_path'])
        pts3d, intensity = mc.load_lidar(meta['lidar_path'])
        pairs.append({'img': img, 'pts3d': pts3d, 'intensity': intensity,
                      'label': meta['label'], 'number': meta['number']})

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(get_output_root(session), f'manual_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nLaunching the OpenCV picking UI. Output will be saved to:\n  {out_dir}')
    print('(Press Q/ESC in either window, or close a window, when finished.)\n')

    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        try:
            mc.CalibTool(pairs).run()
        except Exception as e:
            print(f'\n  ERROR running the picking UI: {e}')
            print('  (This UI needs a graphical display / OpenCV GUI backend.)')
            return
    finally:
        os.chdir(cwd)

    corr_path = os.path.join(out_dir, 'correspondences.json')
    calib_path = os.path.join(out_dir, 'calibration_result.json')

    session['data_dir'] = data_dir

    if os.path.isfile(corr_path):
        label = f'manual_{run_id}'
        session['correspondence_sets'][label] = {
            'path': corr_path, 'data_dir': data_dir, 'added': now_iso()}
        session['active_correspondences'] = label
        print(f"  Correspondences registered as '{label}'.")
    else:
        print('  No correspondences.json produced (no points were picked).')

    if os.path.isfile(calib_path):
        with open(calib_path) as f:
            result = json.load(f)
        t_label = f'manual_{run_id}'
        session['transforms'][t_label] = {
            'path': calib_path, 'source': 'manual_calibrate', 'added': now_iso()}
        print(f"  Transform registered as '{t_label}'.")
        print(f"  translation = {result.get('translation')}")
        print(f"  quaternion  = {result.get('quaternion')}")
    else:
        print('  No calibration_result.json produced (solver was not run — press ENTER '
              'in the picking UI to solve before quitting next time).')

    log_event(session, 'manual_calibration', data_dir=data_dir, out_dir=out_dir)


def action_import_transform(session):
    print('\n── Register an existing transform ──')
    path = pick_path(session, 'transform', 'file', 'Select transform JSON (or .npy)',
                     filetypes=[('Transform files', '*.json *.npy'), ('All files', '*.*')])
    if not path:
        print('  Cancelled.')
        return

    default_label = os.path.splitext(os.path.basename(path))[0]
    label = prompt_text('Label for this transform', default=default_label)
    label = _unique_label(session['transforms'], label)
    session['transforms'][label] = {'path': path, 'source': 'imported', 'added': now_iso()}
    print(f"  Registered '{label}' -> {path}")

    if not session['correspondence_sets']:
        if prompt_yes_no(
                'No correspondences registered yet. Register a correspondences.json now '
                "(needed for 'Evaluate')?", default=False):
            cpath = pick_path(session, 'correspondences', 'file',
                              'Select correspondences.json', filetypes=[('JSON', '*.json')])
            if cpath:
                clabel = _unique_label(session['correspondence_sets'],
                                       os.path.splitext(os.path.basename(cpath))[0])
                session['correspondence_sets'][clabel] = {
                    'path': cpath, 'data_dir': None, 'added': now_iso()}
                session['active_correspondences'] = clabel
                print(f"  Registered correspondences '{clabel}'.")

    log_event(session, 'import_transform', label=label, path=path)


def action_evaluate(session, mods):
    ec = mods['evaluate']
    print('\n── Evaluate transform(s) against correspondences ──')

    corr_label = _select_correspondences(session)
    if corr_label is None:
        return
    corr_path = session['correspondence_sets'][corr_label]['path']

    if not session['transforms']:
        print('  No transforms registered yet.')
        if not prompt_yes_no('Register one now?', default=True):
            return
        action_import_transform(session)
        if not session['transforms']:
            return
    elif prompt_yes_no('Add another transform file before selecting?', default=False):
        action_import_transform(session)

    labels = list(session['transforms'].keys())
    items = [(lbl, session['transforms'][lbl]['path']) for lbl in labels]
    chosen = multi_select('Select transform(s) to evaluate:', items)
    if not chosen:
        print('  Cancelled.')
        return
    chosen_labels = [labels[i] for i in chosen]

    try:
        doc = ec.load_correspondences(corr_path)
    except Exception as e:
        print(f'  ERROR loading correspondences: {e}')
        return
    pairs = doc['pairs']
    intr = doc['intrinsics']
    total_pts = sum(len(p['correspondences']) for p in pairs)
    print(f'  {len(pairs)} pair(s) · {total_pts} total correspondences')
    K, dist = ec.build_camera_params(intr)

    all_rows, eval_labels = [], []
    for lbl in chosen_labels:
        tpath = session['transforms'][lbl]['path']
        try:
            T = ec.load_transform(tpath)
        except Exception as e:
            print(f"  ERROR loading transform '{lbl}': {e} — skipping")
            continue
        rows = ec.evaluate_transform(lbl, T, pairs, K, dist)
        all_rows.extend(rows)
        eval_labels.append(lbl)
        errs = np.array([r['error_px'] for r in rows])
        if len(errs):
            print(f'  [{lbl}] {len(errs)} pts  mean={errs.mean():.3f}px  '
                  f'std={errs.std():.3f}px  max={errs.max():.3f}px')

    if not all_rows:
        print('  No results — nothing to report.')
        return

    ec.print_summary(all_rows, eval_labels, pairs)

    out_root = get_output_root(session)
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, f'evaluate_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, 'errors.csv')
    ec.write_csv(all_rows, out_csv)

    out_plot = None
    if ec.HAS_MPL:
        if prompt_yes_no('Generate plots?', default=True):
            no_display = not prompt_yes_no(
                'Show plots interactively (in addition to saving them)?', default=False)
            out_plot = os.path.join(out_dir, 'errors')
            ec.make_plots(all_rows, eval_labels, pairs, save_path=out_plot,
                          no_display=no_display)
    else:
        print('  (matplotlib not installed — skipping plots)')

    session['last_evaluate'] = {
        'correspondences': corr_label, 'transforms': eval_labels,
        'out_csv': out_csv, 'out_plot': out_plot, 'time': now_iso()}
    log_event(session, 'evaluate', correspondences=corr_label,
             transforms=eval_labels, out_csv=out_csv)
    print(f'\n  Done. CSV -> {out_csv}')


def action_visualize(session, mods):
    vc = mods['visualize']
    print('\n── Visualize a transform ──')

    mode_idx = prompt_menu('Mode:', [
        ('Single image/LiDAR pair', 'pick one camera image + one LiDAR .npy'),
        ('Directory batch', 'scan a folder for matched image/lidar triplets'),
    ])
    if mode_idx is None:
        return

    transform_label = _select_transform(session)
    if transform_label is None:
        return
    transform_path = session['transforms'][transform_label]['path']

    try:
        vc.load_intrinsics(transform_path)
    except Exception:
        pass  # transform file has no embedded intrinsics -> defaults are used

    radius = prompt_int('Point radius in pixels', default=2, min_v=1, max_v=20)
    max_depth_raw = prompt_text('Max depth for colour scale in metres (blank = auto)', default='')
    max_depth = float(max_depth_raw) if max_depth_raw else None
    show = prompt_yes_no('Show interactive windows?', default=True)

    out_root = get_output_root(session)
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, f'visualize_{run_id}')

    if mode_idx == 0:
        cam_path = pick_path(session, 'camera_image', 'file', 'Select camera image',
                             filetypes=[('Images', '*.png *.jpg *.jpeg'), ('All files', '*.*')])
        if not cam_path:
            return
        lidar_path = pick_path(session, 'lidar_npy', 'file', 'Select LiDAR .npy',
                               filetypes=[('NumPy', '*.npy')])
        if not lidar_path:
            return
        try:
            vc.process_one(cam_path, lidar_path, transform_path, out_dir,
                           radius=radius, max_depth=max_depth, no_display=not show)
        except Exception as e:
            print(f'  ERROR: {e}')
            return
        session['last_visualize'] = {
            'mode': 'single', 'camera_image': cam_path, 'lidar_npy': lidar_path,
            'transform': transform_label, 'out_dir': out_dir, 'time': now_iso()}
    else:
        directory = pick_path(session, 'visualize_dir', 'dir',
                              'Select directory to scan for image/lidar pairs')
        if not directory:
            return
        try:
            triplets = vc.find_pairs(directory)
        except Exception as e:
            print(f'  ERROR: {e}')
            return
        print(f'  Found {len(triplets)} matched pair(s).')
        if not prompt_yes_no('Proceed?', default=True):
            return
        for i, trip in enumerate(triplets, start=1):
            label = f'{i:03d}'
            print(f'── {label} ──')
            try:
                vc.process_one(trip['image'], trip['lidar'], transform_path, out_dir,
                               radius=radius, max_depth=max_depth, no_display=not show,
                               run_label=label)
            except Exception as e:
                print(f'  ERROR processing {label}: {e}')
                continue
        session['last_visualize'] = {
            'mode': 'directory', 'directory': directory, 'transform': transform_label,
            'out_dir': out_dir, 'time': now_iso()}

    log_event(session, 'visualize', transform=transform_label, out_dir=out_dir)
    print(f'\n  Done. Outputs -> {out_dir}')


def action_manage(session):
    while True:
        print('\n── Manage session ──')
        print(f"  Session file : {session['session_file']}")
        print(f"  Output root  : {session.get('output_root') or '(default, created on first use)'}")

        print('\n  Transforms:')
        if session['transforms']:
            for lbl, meta in session['transforms'].items():
                print(f"    - {lbl}: {meta['path']}  [{meta['source']}]")
        else:
            print('    (none)')

        print('\n  Correspondence sets:')
        if session['correspondence_sets']:
            for lbl, meta in session['correspondence_sets'].items():
                active = '  (active)' if lbl == session['active_correspondences'] else ''
                print(f"    - {lbl}: {meta['path']}{active}")
        else:
            print('    (none)')

        idx = prompt_menu('Actions:', [
            ('Remove a transform', ''),
            ('Rename a transform', ''),
            ('Remove a correspondence set', ''),
            ('Set active correspondence set', ''),
            ('Change output directory', ''),
            ('View recent history', ''),
        ], cancel_label='Back to main menu')

        if idx is None:
            return
        if idx == 0:
            _remove_item(session['transforms'], 'transform')
        elif idx == 1:
            _rename_item(session['transforms'], 'transform')
        elif idx == 2:
            removed = _remove_item(session['correspondence_sets'], 'correspondence set')
            if removed and session['active_correspondences'] == removed:
                session['active_correspondences'] = None
        elif idx == 3:
            labels = list(session['correspondence_sets'].keys())
            if not labels:
                print('  None registered.')
            else:
                i = prompt_menu('Choose active set:', [(l, '') for l in labels])
                if i is not None:
                    session['active_correspondences'] = labels[i]
        elif idx == 4:
            new_root = prompt_text('New output root directory',
                                   default=session.get('output_root') or '')
            if new_root:
                session['output_root'] = os.path.abspath(new_root)
                os.makedirs(session['output_root'], exist_ok=True)
        elif idx == 5:
            if not session['history']:
                print('  No history yet.')
            for h in session['history'][-15:]:
                extra = {k: v for k, v in h.items() if k not in ('time', 'action')}
                print(f"    {h['time']}  {h['action']}  {extra}")


# ══════════════════════════════════════════════════════════════════════════
# Session startup
# ══════════════════════════════════════════════════════════════════════════

def choose_or_create_session(path_arg):
    if path_arg:
        path = os.path.abspath(path_arg)
        if os.path.isfile(path):
            print(f'Loading session: {path}')
            try:
                return load_session(path)
            except Exception as e:
                print(f'  Could not load session ({e}); starting fresh at this path.')
        s = new_session(path)
        save_session(s)
        print(f'Created new session: {path}')
        return s

    default = os.path.abspath(DEFAULT_SESSION_PATH)
    if os.path.isfile(default):
        print(f'Found existing session: {default}')
        if prompt_yes_no('Resume this session?', default=True):
            try:
                return load_session(default)
            except Exception as e:
                print(f'  Could not load session ({e}).')

    print('\nNo session loaded yet.')
    resp = prompt_text('Session file path to use', default=DEFAULT_SESSION_PATH)
    path = os.path.abspath(resp)
    if os.path.isfile(path):
        if prompt_yes_no(f"'{path}' exists. Load it?", default=True):
            try:
                return load_session(path)
            except Exception as e:
                print(f'  Could not load session ({e}); creating a new one at this path.')
    s = new_session(path)
    save_session(s)
    print(f'Created new session: {path}')
    return s


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def build_arg_parser():
    p = argparse.ArgumentParser(
        prog='calib_suite.py',
        description='Interactive launcher for manual LiDAR-camera calibration, '
                    'evaluation, and visualization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'This tool is fully interactive -- run it with no arguments and follow\n'
            'the prompts. It wraps manual_calibrate.py, evaluate_calibration.py, and\n'
            'visualize_calibration.py, which must live in the same folder as this\n'
            'script (or pass --scripts-dir).\n\n'
            'A session JSON file tracks your data folder, registered transforms, and\n'
            'correspondence sets so you can quit and resume later.'
        ),
    )
    p.add_argument('--session', metavar='PATH', default=None,
                   help='Session JSON file to load or create '
                        f'(default: prompt, offering {DEFAULT_SESSION_PATH})')
    p.add_argument('--scripts-dir', metavar='DIR', default=None,
                   help='Folder containing manual_calibrate.py / evaluate_calibration.py / '
                        'visualize_calibration.py (default: the folder this script is in)')
    return p


MAIN_OPTIONS = [
    ('New calibration from a data folder', 'run the manual point-picking UI'),
    ('Register an existing transform', 'skip picking; import a transform JSON'),
    ('Evaluate transform(s)', 'reprojection error vs. correspondences'),
    ('Visualize a transform', 'project LiDAR onto image / image onto LiDAR'),
    ('Manage session', 'transforms, correspondence sets, output dir, history'),
]


def main():
    args = build_arg_parser().parse_args()

    scripts_dir = args.scripts_dir or os.path.dirname(os.path.abspath(__file__))
    mods = load_tools(scripts_dir)

    session = choose_or_create_session(args.session)
    save_session(session)

    while True:
        print_header(session)
        try:
            idx = prompt_menu('Main menu:', MAIN_OPTIONS, cancel_label='Quit')
        except (KeyboardInterrupt, EOFError):
            print('\nExiting.')
            break

        if idx is None:
            break

        try:
            if idx == 0:
                action_new_calibration(session, mods)
            elif idx == 1:
                action_import_transform(session)
            elif idx == 2:
                action_evaluate(session, mods)
            elif idx == 3:
                action_visualize(session, mods)
            elif idx == 4:
                action_manage(session)
        except (KeyboardInterrupt, EOFError):
            print('\n  Interrupted.')
        except Exception as e:
            print(f'\n  ERROR: {e}')
            traceback.print_exc()

        save_session(session)

    save_session(session)
    print(f"\nSession saved to {session['session_file']}. Goodbye!")


if __name__ == '__main__':
    main()
