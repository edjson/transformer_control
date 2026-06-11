# Stabilization failure — diagnostic checkpoint

Checkpoint: cartpole_cos_sin_theta/aa341f9c-.../checkpoint_epoch1_step91000.pt
Result: 0/50 success on in-distribution systems (eval_cartpole.py)
Symptom: stabilization-phase forces ~3x expert magnitude; representational
         collapse toward saturated forces. Swing-up mode-switching ~perfect.
Commit:  <git rev-parse HEAD>
Config:  <link or path to the training config used>
