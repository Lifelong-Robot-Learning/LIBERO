# Policy Architectures
We provide three vision-language models for encoding spatial-temporal information in robot learning for LLDM.

<div class="container">
  <div class="row">
    <div class="col-md-16 col-lg-12">
      <figure>
        <div class="image" >
          <img src="../../_images/arch.png" class="img-fluid">
        </div>
        <figcaption class="figcaption-study">Three Vision-Language Policy Networks</figcaption>
      </figure>
    </div>
  </div>
  <div class="row">
    <div class="center-div">
    <div class="col-md-10 col-lg-8">
      <figure>
        <div class="image" >
          <img src="../../_images/encoder.png" class="img-fluid">
        </div>
        <figcaption class="figcaption-study">How Sentence Embedding is Injected</figcaption>
      </figure>
    </div>
  </div>
  <div>
</div>

### BCRNNPolicy (ResNet-LSTM)
(See [Robomimic](https://arxiv.org/abs/2108.03298))

The visual information is encoded using a ResNet-like architecture, then the temporal information is summarized by an LSTM. The sentence embedding of the task description is added to the network via the [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layer.


### BCTransformerPolicy (ResNet-Transformer)
(See [VIOLA](https://arxiv.org/abs/2210.11339))

The visual information is encoded using a ResNet-like architecture, then the temporal information is encoded by a temporal transformer that uses the visual encoded representations as tokens. The sentence embedding of the task description is added to the network via the [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layer.


### BCViLTPolicy (ViT-Transformer)
(See [VilT](https://arxiv.org/abs/2102.03334))

The visual information is encoded using a ViT-like architecture, where the images are patchified. Then the temporal information is summarized by another transformer. The sentence embedding of the task description is treated as a token into the spatial ViT.
