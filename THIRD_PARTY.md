# Third-party software and weights

- [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), Apache-2.0. Only the Small checkpoint is used; the larger upstream variants have different terms.
- [Transformers](https://github.com/huggingface/transformers), Apache-2.0.
- [OpenCV](https://github.com/opencv/opencv), Apache-2.0.

The model revision is pinned in `predict.py` and cached into the container at
build time. The adapter code in this repository is MIT licensed.

