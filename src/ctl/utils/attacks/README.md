You can extend the classification framework with your own custom attacks or integrate existing attacks from the Adversarial Robustness Toolbox (ART).

ART provides several attack categories:
   - [`Evasion attacks`](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/art/attacks/evasion) – adversarial examples generated at inference time 
   - [`Inference (privacy) attacks`](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/art/attacks/inference) – attacks targeting sensitive information leakage
   - [`Poisoning attacks`](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/art/attacks/poisoning) – attacks that manipulate the training data

To use ART attacks, wrap your model with the appropriate ART estimator and configure the attack within your experiment settings.
A working integration example using PGD is already [available](pgd_art.py) in the repository and can be used as a reference implementation.
