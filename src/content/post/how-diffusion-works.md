---
title: How Diffusion Works
category: Diffusion
tags: [how-it-works]
created_at: 2025-01-05
updated_at: 2025-01-05
language: en
---

# How Diffusion Works

**Motivation**: How can we understand images, video, and audio so thoroughly that we can create entirely new, realistic examples from scratch? Diffusion models approach this by learning to denoise data that has been corrupted step-by-step, so they can eventually start from pure noise and generate original samples. 

### Introduction
Generative modeling aims to learn an approximation of an unknown probability distribution $\large p(x)$ (e.g., images of cars) given observed data samples. The ultimate goal is to generate new samples $\large x$ that are statistically similar to the original data ($\large x \sim p(x)$).

Diffusion models achieve this via a two-step process: a forward process and a reverse process. The forward process starts with a clean data sample $\tiny x_0$ (i.e. image of a car) and progressively adds Gaussian noise over a series of steps $ \large t = 0, 1, \dots, T$. This process can be described mathematically as: $\large q$($\tiny x_t$ $\mid$ $\tiny x_{t-1}$) = $\mathcal{N}$($\tiny x_t$; $\sqrt{1-\beta_t} \,$ $\tiny x_{t-1}$, $\beta_t \, \mathbf{I}$), where the term $\beta_t$ is part of a variance schedule (often increasing with $\large t$ and is a constant in practice) that determines how much noise is added at each step. In simpler terms, the mean $\sqrt{1-\beta_t} \, x_{t-1}$ ensures that $x_t$ is a slightly noisier version of $x_{t-1}$, while the variance $\beta_t \, \mathbf{I}$ adds random noise to the sample. By the final step ($\large t = T$), the sample has been transformed into (nearly) pure Gaussian noise, drawn from a distribution that closely approximates a standard Gaussian. 

This structured corruption at a step $\tiny x_t$ can be arbitrarily generated given $\tiny x_0$ using $\large q$($\tiny x_t$ | $\tiny x_0$) = $\mathcal{N}$($\tiny x_t$; $\sqrt{\prod_{s=1}^t (1 - \beta_s)}$ $\tiny x_0$, $(1 - \prod_{s=1}^t (1 - \beta_s)) \mathbf{I}$), which is used to train the reverse process. In the variance term $\prod_{s=1}^{t} (1 - \beta_s)$ represents the cumulative fraction of the original data (i.e. signal) that survives through all $t$ steps and $1 - \prod_{s=1}^{t} (1 - \beta_s)$ then gives the complement, which corresponds to the cumulative variance in the system. At first impression, the variance term may seem like it should be just $\prod_{s=1}^{t} \beta_s$, but this does not reflect that the the total variance depends on how much of the original signal is still present.

The reverse process is defined as a learned Markov chain and begins with a noisy sample taken from a standard Gaussian $\tiny x_T$ $\large \sim \mathcal{N}$($0, I$) and iteratively removes noise, step by step, to recover a sample from the target distribution $\tiny x_0$ $\large \sim p(x)$. At each time $\large t$, the goal is to transform a sample drawn from $\large p$($\tiny x_{t}$) into one that follows $\large p$($\tiny x_{t-1}$). Note that $\large p$($\tiny x_{t}$) represents the probability distribution over possible paths that the original sample $\tiny x_0$ could have transitioned through ($\tiny{x}_1$, $\dots$, $\tiny{x}_{t-1}$) to end up at $\tiny x_t$. In standard implementations, each sample follows the same $\large T$-step trajectory. 

![Overview](src/content/post/images/diffusion_1.png)
<div style="text-align: center;">
  <p>Basic diagram of diffusion</p>
</div>

### Gaussian Approximation

The reverse process subproblem is modeled as: $\tiny p_{\theta}$($\tiny x_{t-1}$ | $\tiny x_{t}$). The reverse process is trying to guess the less noisy version of $\tiny x_t$ (i.e. $\tiny x_{t-1}$) by assigning probabilities to all possible values $\tiny x_{t-1}$ might take. This likelihood is modeled as a probability distribution, centered around a predicted value $\tiny \mu_\theta$($\tiny x_t$, $\large t$), with some uncertainty measured by the variance $\tiny \Sigma_\theta$($\tiny x_t$, $\large t$) and can be well-approximated by a Gaussian because:
1. The forward process gradually adds Gaussian noise to the data, and the reverse process denoises it step by step. Under small variance conditions, each step involves only small changes and a Gaussian approximation is effective.
2. Using Bayes' rule, the reverse process is modeled as $\tiny p_{\theta}$ ($\tiny x_{t-1}$ $\mid$ $\tiny x_t$) $\propto \large q$ ($\tiny x_t$ $\mid$ $\tiny x_{t-1}$) $\tiny p_{\theta}$($\tiny x_{t-1}$), where $\large q$($\tiny x_t$ $\mid$ $\tiny x_{t-1}$), defined by the Gaussian forward process, dominates the combined terms, although the prior $\tiny p_{\theta}$ ($\tiny x_{t-1}$)  may represent a complex learned distribution.

3. To parametrize $\tiny p_{\theta}$, a neural network is trained to predict $\tiny \mu_\theta$($\tiny x_t$, $\large t$) and $\tiny \Sigma_\theta$($\tiny x_t$, $\large t$), which are conditioned on $\tiny x_t$. These parameters $\theta$ can vary flexibly across different states $\tiny x_t$, allowing the overall model to capture non-Gaussian structure in the data.  
4. They are favored for their simplicity and tractability.

 Thus, the reverse process assumes Gaussian transition distributions, such that $\tiny p_{\theta}$ ($\tiny x_{t-1}$ $\mid$ $\tiny x_t$) $\approx$ $\mathcal{N}$($\tiny x_{t-1}$; $\tiny \mu_{\theta}$($\tiny x_t$, $\large t$), $\tiny \Sigma_{\theta}$($\tiny x_t$, $\large t$)). 

### Learning the Reverse Process
Using samples generated by $\large q$($\tiny x_t$ | $\tiny x_0$), a neural network with shared parameters $\tiny f_{\theta}$($\cdot$) can be trained to approximate either $\tiny \tilde{\mu}_t$ (the true conditional mean) or $\large \epsilon \sim \mathcal{N}(0, I)$ (the Gaussian noise added to $\tiny x_{t-1}$). Training the model to predict $\large \epsilon$ is preferred because this simplifies the variational lower bound into a training objective that resembles denoising score matching and Langevin dynamics, which is grounded in sampling techniques and makes training more efficient. Strictly speaking, $\tiny \Sigma_{\theta}$($\tiny x_t$, $\large t$) = $\tiny \sigma^2_t$ $I$ = $\tiny \beta_t$ can be fixed to a constant or learned.

When the model learns to predict $\tiny \tilde{\mu}_t$, the loss is the MSE between the prediction and the actual mean of $\large q$($\tiny x_{t-1}$ $\mid$ $\tiny x_t$, $\tiny x_0$) (which is tractable when conditioned on $\tiny x_0$: $\tiny \tilde{\mu}_t$($\tiny x_t$, $\tiny x_0$) $=$ $\tiny \left( \frac{\sqrt{\prod\limits_{i=1}^{t-1} (1 - \beta_i)} \beta_t}{1 - \prod\limits_{i=1}^{t} (1 - \beta_i)} \right)$ $\tiny x_0$ $+$ $\tiny \left( \frac{\sqrt{\scriptstyle{1 - \beta_t}} \left( 1 - \prod\limits_{i=1}^{t-1} (1 - \beta_i) \right)}{1 - \prod\limits_{i=1}^{t} (1 - \beta_i) } \right)$ $\tiny x_t$). Intuitively, when $\tiny x_t$ is highly corrupted (high noise $\tiny \beta_t$), it relies more on the cleaner reference $\tiny x_0$, whereas when $\tiny x_t$ retains more signal (low $\tiny \beta_t$), it contributes more to the reverse process as it is closer to $\tiny x_{t-1}$. For the in-depth derivation of the forward process posterior mean, see the DDPM paper (Ho et al. 2020). The loss is approximately (removing constants) $\tiny L_{t-1}$ $=$ $\mathbb{E} \|$ $\tiny \mu_\theta$($\tiny x_t$, $\large t$) - $\tilde{\mu}_t$($\tiny x_t$, $\tiny x_0$)$\tiny \|_2^2$. 

An equal and more empirically stable approach is to re-parameterize the model to predict the $\tiny \epsilon_{\theta}$ ($\tiny x_t$, $\large t$) that corrupts $\tiny x_{t-1}$ to $\tiny x_{t}$. This is done by rewriting $\large q$($\tiny x_t$ | $\tiny x_0$) as $\large q$($\tiny x_t$ | $\tiny x_0$, $\large \epsilon$) = $\sqrt{\prod\limits_{i=1}^{t} (1 - \beta_i)}$ $\tiny x_0$ + $\sqrt{1 - \prod\limits_{i=1}^{t} (1 - \beta_i)}$ $\large \epsilon$ where $\large \epsilon \large \sim \mathcal{N}$($0, I$) and using in place of sampling from a distribution $\tiny x_t$. A similar (approximate) loss function is used here: $\tiny L_{t-1}$ $=$ $\mathbb{E} \|$ $\tiny \epsilon_\theta$($\tiny x_t$, $\large t$) - $\tilde{\epsilon} \tiny \|_2^2$. Once $\tiny \epsilon_{\theta}$ ($\tiny x_t$, $\large t$) is predicted, the implied mean is $\tiny \mu_\theta$($\tiny x_t$, $\large t$) = $\tiny \frac{1}{\sqrt{1 - \beta_t}} \left( \tiny x_t \normalsize - \tiny \frac{\tiny \beta_t}{\tiny \sqrt{1 - \prod\limits_{i=1}^{t} (1 - \beta_i)}} \, \tiny \epsilon_\theta \normalsize(\tiny x_t \normalsize, t) \right)$. 

Next, having approximated the conditional mean, both approaches follow the same Gaussian sampling step: $\tiny x_{t-1}$ = $\mu_\theta$($\tiny x_t$, $\large t$) + $\tiny \sigma_t \normalsize z$, where $\large z \large \sim \mathcal{N}$($0, I$). 

### Code
This code is based on the [Pytorch DDPM implementation](https://github.com/lucidrains/denoising-diffusion-pytorch). 

First, we define upsampling/downsampling and some helper functions. 
```python
# Checks whether a given object is None
def exists(x):
    return x is not None

# Returns 'val' if it's not None, otherwise returns 'd' if 'val' is None
def default(val, d):
    return val if exists(val) else (d() if isfunction(d) else d)

# Divides 'num' elements into groups of size 'divisor'; handles remainders
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    group_list = [divisor] * groups
    if remainder > 0:
        group_list.append(remainder)
    return group_list
    
# Upsampling module: scales resolution by 2 using NN-interpolation and a convolution
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

# Downsampling module: rearranges spatial dimensions and applies a 1x1 convolution  
class Rearrange(nn.Module):
    #  p1 = 2, p2 = 2: divides each block of 2x2 pixels into 4 mini-channels 
    #  ex. [ A B ]
    #      [ C D ]
    # turns to [ [A, B, C, D] ]
    def __init__(self, pattern, p1=2, p2=2):
        super().__init__()
        self.pattern = pattern
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        return rearrange(x, self.pattern, p1=self.p1, p2=self.p2)

def Downsample(dim, dim_out=None):
    # Steps
    # 1. Rearrange() packs pixels into channels (downsample)
    # 2. 1x1 conv layer reduces number of channels
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
```
Next, we introduce a Residual module which is a general-purpose wrapper. It does not perform any specific computation by itself other than adding the input to the function's output to create skip connections (which improves gradient flow and prevents information loss). This makes sure the network doesn’t lose track of the original information while learning new features. 
```python
# A module that creates a skip connection by adding the input to the output
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
```
Below, this module encodes a single step number (i.e. t=1, t=2, etc.) in a richer way for the model to detect when it is operating. The model learns to associate specific embeddings with the correct output behavior. For example, at time steps where there are low noise levels, the model focuses on detail reconstruction and at high noise levels, it focuses on denoising. This method was originally proposed in the Transformer paper (Vaswani et al., 2017).
```python
# A module that converts time steps to sinusoidal embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # dim: [time_n]
        device = time.device
        half_dim = self.dim // 2 # first half of dim is sine values, other is cosine
        log_factor = math.log(10000) / (half_dim - 1) # frequency spacing
        exponent = torch.exp(torch.arange(half_dim, device=device) * -log_factor)
        expanded = time[:, None] * exponent[None, :]
        return torch.cat((expanded.sin(), expanded.cos()), dim=-1) # [time_n, self.dim]
```
The ResnetBlock module is a specific implementation of a residual block used in ResNet architectures. It includes two convolutional layers (Block modules) with normalization and activation. It conditions its behavior on time embeddings generated by the previous module. 
```python
# A module that extracts features from the input
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # filters learn to detect noise patterns
    def forward(self, x, scale_shift=None):
        x = self.proj(x) # apply convolution, filter weights are learned during training
        x = self.norm(x) # apply normalization

        # adapts features based on step
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x) # apply activation
        return x

# A module that combines feature extraction (using Blocks) with residual connections
class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups) 
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        # if time embeddings provided, learns to adjust features based on step
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        # first block learns low-level features (edges, noise)
        h = self.block1(x, scale_shift=scale_shift)
        # second block combines those features into high-level (denoised sections)
        h = self.block2(h)
        return h + self.res_conv(x)
```
Attention compares all parts of the input to identify relevance for a task. It uses three components: the query (Q), which represents what the model is currently searching for (i.e. features related to a car); the key (K), which represents how each part of the sample is described (i.e. patch of pixels' label); and the value (V), which represents the actual information in parts of the sample (i.e. detailed pixel features). By comparing each query to all the keys (calculating similarity scores), the network produces attention weights (relevance scores) used to combine the values and highlight on the most relevant region for a task. 

The Attention module enables full interaction between input elements (dense attention), while LinearAttention offers a more efficient approach, suitable for large inputs.

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # 1x1 conv to find q, k, v
        self.to_out = nn.Conv2d(hidden_dim, dim, 1) # 1x1 conv to restore orig dimension

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv]
        q = q * self.scale

        # similarity score between Q and K
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # use attention weights to calculate weighted sum (V)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # GroupNorm for stability
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv]

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
```
This is the U-Net (neural network) that predicts noise (or alternatively, conditional mean, however this is less efficient) from noisy images at different timesteps.
The structure is: 

1. Initial convolution on noisy input (and optional self-conditioning).
2. Several downsampling blocks (ResNet + attention + downsample).
3. A middle portion with more ResNet + attention.
4. Several upsampling blocks symmetric to the downsampling steps.
5. Final ResNet block and a 1x1 convolution to predict noise.

```python 
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_cls = partial(ResnetBlock, groups=resnet_block_groups)

        # Time-embedding MLP: position embeddings -> linear -> GELU -> linear
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        total_resolutions = len(in_out)

        for idx, (dim_in, dim_out) in enumerate(in_out):
            last_layer = (idx >= (total_resolutions - 1))
            self.downs.append(
                nn.ModuleList([
                    block_cls(dim_in, dim_in, time_emb_dim=time_dim),
                    block_cls(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not last_layer else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # Middle ResNet + attention
        mid_dim = dims[-1]
        self.mid_block1 = block_cls(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_cls(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling layers
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            final_layer = (idx == (len(in_out) - 1))
            self.ups.append(
                nn.ModuleList([
                    block_cls(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_cls(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not final_layer else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_cls(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        residual_copy = x.clone()

        time_embed = self.time_mlp(time)
        saved = []

        # Downsampling
        for (block1, block2, attn, downsample) in self.downs:
            x = block1(x, time_embed)
            saved.append(x)
            x = block2(x, time_embed)
            x = attn(x)
            saved.append(x)
            x = downsample(x)

        # Mid network
        x = self.mid_block1(x, time_embed)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_embed)

        # Upsampling
        for (block1, block2, attn, upsample) in self.ups:
            x = torch.cat((x, saved.pop()), dim=1)
            x = block1(x, time_embed)
            x = torch.cat((x, saved.pop()), dim=1)
            x = block2(x, time_embed)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, residual_copy), dim=1)
        x = self.final_res_block(x, time_embed)
        return self.final_conv(x)
```
 We now define the forward diffusion process that progressively corrupts images by adding noise at each step, which is governed by a variance schedule. For instance, a linear schedule can start at 1e-4 and end at 0.02, or a cosine schedule can be used. We also define a helper method called 'extract' to pick out the relevant terms for each example in a batch.
 ```python 
 def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine-based variance schedule introduced in:
    https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    start, end = 0.0001, 0.02
    return torch.linspace(start, end, timesteps)

def quadratic_beta_schedule(timesteps):
    start, end = 0.0001, 0.02
    return torch.linspace(start**0.5, end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    start, end = 0.0001, 0.02
    rng = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(rng) * (end - start) + start

timesteps = 300
betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    bsz = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(bsz, *((1,) * (len(x_shape) - 1))).to(t.device)

# samples x_t given x_0 to create training pairs (x_noisy, noise)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_coef = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_coef * x_start + sqrt_one_minus * noise
 ```
 The training objective is to match the predicted noise to the actual noise dded at each timestep. This can be L1, L2, or the Huber loss. In practice, L1 or Huber can provide stable training.
 ```python
 def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        return F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        return F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        return F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
 ```
Finally, we can set up a typical training loop. We define a simple Adam optimizer and then, for each batch, we sample random timesteps and compute the loss (comparing actual noise with predicted noise). We optionally save and sample images to monitor the training progress.
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 6

for epoch in range(epochs):
    for step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        real_images = batch_data["pixel_values"].to(device)
        bsz = real_images.shape[0]
        t = torch.randint(0, timesteps, (bsz,), device=device).long()

        loss = p_losses(model, real_images, t, loss_type="huber")
        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            # we sample 4 images at once
            group_sizes = num_to_groups(4, bsz)
            all_samples = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), group_sizes))
            final_images = torch.cat([torch.from_numpy(item[-1]) for item in all_samples], dim=0)
            final_images = (final_images + 1) * 0.5
            save_image(final_images, str(results_folder / f"sample-{milestone}.png"), nrow=6)
```
 After training, we reverse the diffusion process by starting from Gaussian noise. and iteratively denoising using our trained model. This is summarized as Algorithm 2 in the DDPM paper. You can generate new samples by calling: `samples = sample(model, image_size=image_size, batch_size=64, channels=channels)`
 ```python
 def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t, x.shape)

    # Predicted mean from the noise model
    model_mean = sqrt_recip_alpha_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        var_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(var_t) * noise

from tqdm import tqdm

def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    all_imgs = []

    for i in tqdm(reversed(range(timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        all_imgs.append(img.cpu().numpy())
    return all_imgs

def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
 ```
Train your own model and run inference on your own checkpoints! I hope this guide was useful for getting a high level overview of diffusion.   
### References
[1] J. Ho, A. Jain, and P. Abbeel. “Denoising Diffusion Probabilistic Models.” NeurIPS, 2020.

[2] N. Rogge and K. Rasul. "The Annotated Diffusion Model." Hugging Face Blog.



---
