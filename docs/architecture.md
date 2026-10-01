# Architecture and training flow

The generator maps `[batch, 100, 1, 1]` noise tensors to `[batch, 3, 64, 64]` using five transposed-convolution stages. The critic downsamples the image through five convolution stages to a scalar score per image. The critic uses InstanceNorm, while the generator uses BatchNorm and a Tanh output.

The critic objective is `mean(fake scores) - mean(real scores) + lambda * gradient_penalty`. The penalty computes the mean squared deviation of interpolated-image input-gradient norms from one. The generator minimizes the negative mean critic score. Defaults retain the original learning rate (1e-4), batch size (64), critic iterations (5), gradient weight (10), Adam betas (0, 0.9) and epoch count (50).

The original training script imports `Critic` although the supplied class is `CriticNetwork`, and imports `weight_penality` although the supplied function is `calculate_gradient_penalty`. It also passes a mismatched device keyword, runs the Deep Lake download/training at module import time and assumes a full-size batch. The cleaned module uses matching names, local ImageFolder data, an explicit `main`, detached critic-training fakes, interpolation gradients and the observed batch length.

No results from the original source can be inferred from successful syntax checks. Dataset distribution, seed variation, quality metrics and runtime behavior still need evaluation in an environment with PyTorch. All original files remain byte-identical under `archive/original-code`.
