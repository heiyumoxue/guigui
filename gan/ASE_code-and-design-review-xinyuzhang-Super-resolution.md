# My Project Title
Super Resolution————gan

## Abstract
TecoGAN is adversarial training is very successful in single image super-resolution tasks because it results in realistic, highly detailed outputs. Therefore, current optimal video super-resolution methods still support the simpler paradigm as the adversarial loss function. The nature of averaging direct vector parametric as a loss function can easily lead to temporal smoothness and coherence, but the generated images lack spatial detail. This study proposes an adversarial training method for video super-resolution that allows the resolution to have temporal coherence without loss of spatial detail.


## Detailed Description
The GAN can initially only generate images with a small resolution, such as a 32x32 size MNIST handwritten digital image. Because the training of both discriminators and generators is inherently difficult, it is often necessary for both to be on par with each other and not to have any momentum to overwhelm each other in order to achieve the confrontation process, and to have a better confrontation in order to have the final ideal balance. The larger the image, the more complex the feature space, the more challenging the task, and the more difficult it is to get the two to confront each other properly simultaneously.

So it was the classical DCGAN that took the lead in alleviating this problem, proposing the removal of pooling layers, fully connected layers, using BNs and so on. However, the improvement is limited, and only a few larger images can be generated. Moreover, LAPGAN introduces a kind of pyramidal pipeline, which also seems to generate only 96x96 resolution.

Best-buddy GAN (Beby-GAN) then proposes a novel best-buddy loss function, an improved one-to-many MAE loss, to allow flexible funding and use of HR-supervised signals by exploiting the self-similarity prevalent in natural images. It also proposes a region-aware adversarial learning strategy to solve the ringing problem of the generated images. Finally, and breaking the 2K resolution limit of current SISR datasets, we provide ultra-high resolution 4K (UH4K) image datasets with different classes.

TecoGAN wants to reconstruct the missing image details based on the current image information. Video super-resolution techniques are more complex, requiring the generation of detail-rich frames and the maintenance of coherence between images.
Natural image super-resolution is one of the classic challenges in image and video processing. For single image super-resolution (SISR), deep learning-based approaches achieve the best peak signal-to-noise ratio (PSNR) available, while GAN-based architectures achieve significant improvements in perceptual quality.
In video super-resolution (VSR) tasks, existing methods mainly use standard loss functions, such as mean squared loss, rather than adversarial loss functions. Similarly, the evaluation of results remains focused on vector paradigm-based metrics such as PSNR and Structural Similarity (SSIM) metrics. In contrast to SISR, the main difficulty with VSR is obtaining precise results without unnatural artefacts. For example, based on the loss of mean squared error, recent VSR tasks use multiple frames from low-resolution inputs or reuse previously generated results to improve temporal coherence.

BSRGAN, on the other hand, is based on the core idea of y = (x times k) decreasing s plus n. The order of execution of each factor is randomised around the three factors of the degradation model described above: K is the fuzzy kernel, S is the downsampling kernel, and N is the noise (e.g. KSN, NKS, SNK, SKN, NSK, KNS). At the same time, there are different methods for each factor (e.g. the downsampling kernel S can be done in any of the following ways: bicubic, nearest neighbour, bilinear, etc.), and one of these methods can be chosen at random for each factor. At this point, the degenerate model can then be constructed by two stochastic processes.



### Datasets
1.DIV 2K
https://data.vision.ee.ethz.ch/cvl/DIV2K/
2.Urban100: 4x upscaling
https://deepai.org/dataset/urban100-4x-upscaling
3.other datasets create by me
pictures that I find and so on

### Arcitecture Proposal
The VSR architecture consists of a recurrent generator, a stream estimation network and a Spatio-temporal discriminator. Generator G generates high-resolution video frames cyclically based on low-resolution inputs. The stream estimation network F learns frame-to-frame dynamic compensation to aid the generator and the temporal discriminator D_s,t.

The generator and stream estimator are trained together to spoof the Spatio-temporal discriminator D_s,t. This discriminator is the core component as it considers both spatial and temporal factors and penalises results where there is ideological temporal incoherence. This way, G is needed to generate high-frequency details that are continuous with the previous frames. Once training is complete, the additional complexity of D_s,t will have little effect unless G and F training models are required to infer the new super-resolution video output.

I put the images of the architecture in the pictures folder(2.1-2.4)


## References
1.Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network（SRGAN）
2.ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
3.Designing a Practical Degradation Model for Deep Blind Image Super-Resolution（BSRGAN）
4.Best-Buddy GANs for Highly Detailed Image Super-Resolution（AAAI 2022）
5.Designing a Practical Degradation Model for Deep Blind Image Super-Resolution
6.Temporally Coherent GANs for Video Super-Resolution (TecoGAN)
7.Image super-resolution using deep convolutional networks（SRCNN）
8.Wang, Zhihao, et al. Deep Learning for Image Super-Resolution: A Survey. arXiv, 7 Feb. 2020. arXiv.org, https://doi.org/10.48550/arXiv.1902.06068.