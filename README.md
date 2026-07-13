# appnz-depth-vfx

Turn one image into a depth map, normal map, heatmap, or subtle looping
parallax shot. This is a small, auditable [Cog](https://github.com/replicate/cog)
adapter around the Apache-2.0 Depth Anything V2 Small checkpoint.

## Why this exists

Hosted image APIs are good at generating pixels. Depth VFX is the composable
post-production step: generate anywhere, then self-host predictable geometry
passes and camera motion on app.nz. The 25M-parameter checkpoint is baked into
the image, loaded once, uses FP16/channels-last on CUDA, and never downloads
weights during scale-up.

## Run

```bash
cog predict -i image=@photo.jpg -i effect=parallax -i strength=0.55
docker run --rm --gpus all -p 5000:5000 ghcr.io/lee101/appnz-depth-vfx:latest
app cogs create --name depth-vfx --image ghcr.io/lee101/appnz-depth-vfx:latest --hardware gpu-rtx3090
app apps deploy demo --app depth-vfx-demo
```

Effects: `parallax` (MP4), `depth`, `heatmap`, and `normals` (PNG). Inputs are
bounded to 20 MP and parallax clips to six seconds.

## API

```json
{"input":{"image":"https://example.com/photo.jpg","effect":"normals","strength":0.55,"seconds":3}}
```

See `THIRD_PARTY.md` for exact model provenance and terms. Adapter: MIT.
