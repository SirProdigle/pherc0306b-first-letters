"""patch_fitter.py SPIRAL_DIR : add FIT_SPIRAL_STOP_AT to the old spiral fitter (fit_spiral.py), idempotently.

FIT_SPIRAL_STOP_AT=N ends the training loop after N completed iterations of the CONFIGURED horizon (default 30000) and
saves checkpoint_fitted.ckpt with completed_iterations=N, plus the fitted meshes (so the snap test can run on them).
A later run with FIT_SPIRAL_RESUME_PATH=<that ckpt> and the same horizon continues at iteration N with the optimiser,
scheduler and RNG state restored and the LR schedule realigned to the absolute step (fit_spiral.realign_optimizer_lr_schedule),
i.e. the continuation is the same trajectory as one uninterrupted 30k run (up to sampling RNG restore).
Without the env var the fitter behaves exactly as before.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'fit_spiral.py'
s = p.read_text()
MARK = '# FIT_SPIRAL_STOP_AT (x_vast_plan/patch_fitter.py)'
if MARK in s:
    print('fit_spiral.py already patched'); sys.exit(0)
old_loop = """        progress.begin(
            'optimizing', 'Optimizing',
            step=0, total_steps=max(0, self.num_training_steps - self.start_iteration),
            unit='iterations')
        for iteration in tqdm(
                range(self.start_iteration, self.num_training_steps),"""
new_loop = """        """ + MARK + """
        _stop_at = int(os.environ.get('FIT_SPIRAL_STOP_AT', '0') or 0)
        self._loop_end = min(self.num_training_steps, _stop_at) if _stop_at > 0 else self.num_training_steps
        progress.begin(
            'optimizing', 'Optimizing',
            step=0, total_steps=max(0, self._loop_end - self.start_iteration),
            unit='iterations')
        for iteration in tqdm(
                range(self.start_iteration, self._loop_end),"""
old_save = """            self._save_model(suffix, self.num_training_steps)
            if self.config.get('output_save_png_visualizations', False):"""
new_save = """            self._save_model(suffix, getattr(self, '_loop_end', self.num_training_steps))
            if self.config.get('output_save_png_visualizations', False):"""
for a, b in ((old_loop, new_loop), (old_save, new_save)):
    if s.count(a) != 1:
        sys.exit(f'patch anchor not found exactly once ({s.count(a)}): {a[:60]!r}')
    s = s.replace(a, b)
if '\nimport os' not in s and '\nimport os,' not in s and 'import os\n' not in s:
    s = 'import os\n' + s
p.write_text(s)
print('fit_spiral.py patched: FIT_SPIRAL_STOP_AT')
