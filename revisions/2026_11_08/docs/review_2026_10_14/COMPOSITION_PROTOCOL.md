# Composition and support protocol

Frozen before execution on 14 October 2026. The analysis reads completed 20% target Morgan-MLP predictions, endpoint y values, manifests and descriptor/Morgan group scores. No model is retrained.

For every seed/regime report test n, realised fraction, val/train n, positive fraction (classification) or target mean/SD/range/IQR (regression), group count/largest group, and fragment/charge/metal/acyclic fractions. Regression baseline predicts the training median; skill is 1−model MAE/baseline MAE and is undefined for a zero baseline denominator. CYP2D6 additionally reports AUROC. These measures describe composition; they do not make selected populations interchangeable.

Report descriptor/Morgan test Jaccard for each seed and group-score Spearman correlation. Maximum training Tanimoto is calculated in ECFP4 (radius2,2048 bits) and atom-pair (2048 bits), without using the scaled SVD coordinates. Atom-pair is not used by either allocation geometry; ECFP4 remains related to the Morgan family and is not described as wholly representation-independent. Preserve per-molecule similarities, source hashes, protocol and script hash under results/review_2026_10_14/composition/.
