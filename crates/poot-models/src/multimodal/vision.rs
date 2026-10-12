//! Image preprocessing for idefics3 / SmolVLM: a pure transform from a decoded RGB image to the
//! channel-first frames the vision tower ([`super::idefics3`]) encodes.

/// The image-preprocessing parameters poot reads from a model's `preprocessor_config.json` + `vision_config`.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessConfig {
    /// The square side the vision encoder expects (vision_config `image_size`, e.g. 512). Also the tile side
    /// (`max_image_size.longest_edge`) for the idefics3 splitting pipeline.
    pub image_size: usize,
    /// The longest edge the source image is first resized to, preserving aspect (`size.longest_edge`, e.g.
    /// 2048), before splitting into `image_size`-tiles.
    pub longest_edge: usize,
    /// Multiply raw pixel values by this before normalizing (`rescale_factor`, e.g. `1/255`).
    pub rescale_factor: f32,
    /// Per-channel normalization mean / std (`image_mean` / `image_std`, e.g. `0.5`).
    pub mean: [f32; 3],
    /// Per-channel normalization standard deviation (`image_std`).
    pub std: [f32; 3],
}

// --- idefics3 / SmolVLM image splitting (card vlm-image-preprocessing-fidelity) ---
// The reference `Idefics3ImageProcessor` (transformers, `resample=LANCZOS`, `do_image_splitting=true`) turns one
// image into a sequence of `image_size x image_size` frames, not a single squished square. The geometry is a port
// of `image_processing_idefics3.py`. Three steps per source:
//   1. `get_resize_output_image_size`: scale the longest edge to `longest_edge` preserving aspect, then clamp
//      below MAX_IMAGE_SIZE (4096, inert for normal sizes). This upscales small images too.
//   2. `resize_for_vision_encoder`: round each edge up to a multiple of `tile` (image_size), preserving aspect
//      via the long side.
//   3. `split_images`: if either edge > `tile`, crop `ceil(h/tile) x ceil(w/tile)` non-overlapping `tile`-tiles
//      and append one global thumbnail (the whole image resized to `tile x tile`); else a single frame.

/// idefics3 MAX_IMAGE_SIZE (`image_processing_idefics3.py`): the upper bound the longest-edge resize is clamped below.
const IDEFICS3_MAX_IMAGE_SIZE: usize = 4096;

/// Truncate-to-int (matches Python `int(f)` for non-negative `f`).
fn trunc(f: f64) -> usize {
    f as usize
}

/// idefics3 `get_resize_output_image_size`: scale the longest edge to `longest`, preserving aspect and rounding
/// the other edge to an even value, then clamp below [`IDEFICS3_MAX_IMAGE_SIZE`]. Returns `(height, width)`.
/// Ports `_resize_output_size_rescale_to_max_len` + `_resize_output_size_scale_below_upper_bound`.
pub fn idefics3_resize_longest(h: usize, w: usize, longest: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64; // width / height
    let (mut oh, mut ow);
    if w >= h {
        ow = longest;
        oh = trunc(ow as f64 / aspect); // int(width / aspect_ratio) = int(longest * h / w)
        if oh % 2 != 0 {
            oh += 1;
        }
    } else {
        oh = longest;
        ow = trunc(oh as f64 * aspect); // int(height * aspect_ratio) = int(longest * w / h)
        if ow % 2 != 0 {
            ow += 1;
        }
    }
    oh = oh.max(1);
    ow = ow.max(1);
    // scale_below_upper_bound(MAX_IMAGE_SIZE): only shrinks if a side exceeds the bound (inert <= 4096).
    let aspect2 = ow as f64 / oh as f64;
    if ow >= oh && ow > IDEFICS3_MAX_IMAGE_SIZE {
        ow = IDEFICS3_MAX_IMAGE_SIZE;
        oh = trunc(ow as f64 / aspect2);
    } else if oh > ow && oh > IDEFICS3_MAX_IMAGE_SIZE {
        oh = IDEFICS3_MAX_IMAGE_SIZE;
        ow = trunc(oh as f64 * aspect2);
    }
    (oh.max(1), ow.max(1))
}

/// idefics3 `resize_for_vision_encoder`: round each edge up to a multiple of `tile`, preserving aspect via the
/// long side. Returns `(height, width)`. The `aspect_ratio` is the input `w/h`, computed once.
pub fn idefics3_resize_for_encoder(h: usize, w: usize, tile: usize) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let ceil_mul = |x: usize| x.div_ceil(tile) * tile;
    if w >= h {
        let ow = ceil_mul(w);
        let oh = ceil_mul(trunc(ow as f64 / aspect)); // int(width / aspect_ratio) then ceil-to-multiple
        (oh, ow)
    } else {
        let oh = ceil_mul(h);
        let ow = ceil_mul(trunc(oh as f64 * aspect));
        (oh, ow)
    }
}

/// idefics3 `split_images` geometry: given the post-[`idefics3_resize_for_encoder`] dims, return `(rows, cols)`
/// of `tile`-tiles. `(0, 0)` means no split (a single frame, no global thumbnail). When rows/cols > 0 the total
/// frame count is `rows * cols + 1` (the trailing global thumbnail).
pub fn idefics3_split_geometry(h: usize, w: usize, tile: usize) -> (usize, usize) {
    if h > tile || w > tile {
        (h.div_ceil(tile), w.div_ceil(tile))
    } else {
        (0, 0)
    }
}

/// The result of the idefics3 splitting pipeline: the ordered channel-first frames (tiles row-major, then the
/// global thumbnail last, matching the reference `replace_image_token` prompt order) plus the tile grid.
pub struct Idefics3Frames {
    /// Each frame is `[3, image_size, image_size]`, rescaled + normalized (ready for the vision encoder).
    pub frames: Vec<Vec<f32>>,
    /// Tile grid; `(0, 0)` when the image was not split (`frames` holds a single frame).
    pub rows: usize,
    /// Tile grid columns; `(0, 0)` rows and columns when the image was not split.
    pub cols: usize,
}

/// Resample channel `c` of an interleaved-RGB image with a separable Lanczos-3 filter (idefics3 `resample=1` = PIL
/// LANCZOS), widening the filter support by the scale ratio on downscale (PIL's convention). Works in f32 (PIL
/// rounds to u8 between passes, so this is close to but not bit-exact with PIL). `src` is `sh x sw` interleaved
/// RGB; returns `oh x ow` interleaved RGB.
pub fn resize_lanczos(src: &[f32], sh: usize, sw: usize, oh: usize, ow: usize) -> Vec<f32> {
    let a = 3.0f64;
    // Horizontal pass: sw -> ow, producing an sh x ow interleaved buffer.
    let wx = axis_weights(sw, ow, a);
    let mut mid = vec![0.0f32; sh * ow * 3];
    for y in 0..sh {
        for (ox, taps) in wx.iter().enumerate() {
            let mut acc = [0.0f64; 3];
            for &(sx, wt) in taps {
                let p = (y * sw + sx) * 3;
                acc[0] += src[p] as f64 * wt as f64;
                acc[1] += src[p + 1] as f64 * wt as f64;
                acc[2] += src[p + 2] as f64 * wt as f64;
            }
            let q = (y * ow + ox) * 3;
            mid[q] = acc[0] as f32;
            mid[q + 1] = acc[1] as f32;
            mid[q + 2] = acc[2] as f32;
        }
    }
    // Vertical pass: sh -> oh, producing the oh x ow output.
    let wy = axis_weights(sh, oh, a);
    let mut out = vec![0.0f32; oh * ow * 3];
    for (oy, taps) in wy.iter().enumerate() {
        for x in 0..ow {
            let mut acc = [0.0f64; 3];
            for &(sy, wt) in taps {
                let p = (sy * ow + x) * 3;
                acc[0] += mid[p] as f64 * wt as f64;
                acc[1] += mid[p + 1] as f64 * wt as f64;
                acc[2] += mid[p + 2] as f64 * wt as f64;
            }
            let q = (oy * ow + x) * 3;
            out[q] = acc[0] as f32;
            out[q + 1] = acc[1] as f32;
            out[q + 2] = acc[2] as f32;
        }
    }
    out
}

/// Lanczos-`a` kernel value at `x`.
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-9 {
        1.0
    } else if ax >= a {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        a * px.sin() * (px / a).sin() / (px * px)
    }
}

/// Per-output-pixel resampling taps `(src_index, weight)` for a 1D resize from `src` to `dst` samples, using a
/// Lanczos-`a` filter with PIL-style support widening on downscale. Weights are normalized to sum 1.
fn axis_weights(src: usize, dst: usize, a: f64) -> Vec<Vec<(usize, f32)>> {
    let scale = src as f64 / dst as f64;
    let support = a * scale.max(1.0); // widen the filter on downscale to antialias.
    let inv = 1.0 / scale.max(1.0);
    (0..dst)
        .map(|d| {
            let center = (d as f64 + 0.5) * scale;
            let lo = (center - support).floor().max(0.0) as usize;
            let hi = ((center + support).ceil() as usize).min(src);
            let mut taps: Vec<(usize, f32)> = Vec::new();
            let mut sum = 0.0f64;
            for i in lo..hi {
                let w = lanczos_kernel((i as f64 + 0.5 - center) * inv, a);
                if w != 0.0 {
                    taps.push((i, w as f32));
                    sum += w;
                }
            }
            if sum != 0.0 {
                for t in taps.iter_mut() {
                    t.1 = (t.1 as f64 / sum) as f32;
                }
            } else if src > 0 {
                // degenerate (no taps) - nearest source pixel.
                taps.push((center.floor().min((src - 1) as f64) as usize, 1.0));
            }
            taps
        })
        .collect()
}

/// Rescale + per-channel normalize an interleaved-RGB `tile x tile` frame into the channel-first
/// `[3, tile, tile]` layout the vision encoder consumes: `(v * rescale - mean) / std`.
fn normalize_frame(rgb: &[f32], tile: usize, cfg: &PreprocessConfig) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * tile * tile];
    for i in 0..tile * tile {
        for c in 0..3 {
            let v = rgb[i * 3 + c];
            out[c * tile * tile + i] = (v * cfg.rescale_factor - cfg.mean[c]) / cfg.std[c];
        }
    }
    out
}

/// The full idefics3 splitting pipeline for a decoded `h x w` interleaved-RGB image (values `[0, 255]`): resize the
/// longest edge to `cfg.longest_edge` (Lanczos), round each edge up to a multiple of `cfg.image_size`, crop the
/// `rows x cols` tiles, append the global thumbnail (whole image resized to `image_size`), and rescale+normalize
/// each. Returns the ordered [`Idefics3Frames`] (tiles row-major then global); the frame count is `rows * cols + 1`,
/// or `1` when the image is not split.
pub fn idefics3_preprocess(
    rgb: &[f32],
    h: usize,
    w: usize,
    cfg: &PreprocessConfig,
) -> Result<Idefics3Frames, ImageBufferError> {
    if h.checked_mul(w).and_then(|px| px.checked_mul(3)) != Some(rgb.len()) {
        return Err(ImageBufferError {
            len: rgb.len(),
            height: h,
            width: w,
        });
    }
    let tile = cfg.image_size;
    let (rh, rw) = idefics3_resize_longest(h, w, cfg.longest_edge);
    let (eh, ew) = idefics3_resize_for_encoder(rh, rw, tile);
    // The single resize to the encoder-multiple size (Lanczos), then crop exact tiles from it.
    let resized = resize_lanczos(rgb, h, w, eh, ew);
    let (rows, cols) = idefics3_split_geometry(eh, ew, tile);
    let mut frames = Vec::new();
    if rows == 0 && cols == 0 {
        frames.push(normalize_frame(&resized, tile, cfg));
        return Ok(Idefics3Frames { frames, rows, cols });
    }
    // Tiles, row-major (n_h outer, n_w inner) - matching the reference frame/prompt order.
    for nh in 0..rows {
        for nw in 0..cols {
            let mut crop = vec![0.0f32; tile * tile * 3];
            for ty in 0..tile {
                let sy = nh * tile + ty;
                for tx in 0..tile {
                    let sx = nw * tile + tx;
                    let s = (sy * ew + sx) * 3;
                    let d = (ty * tile + tx) * 3;
                    crop[d] = resized[s];
                    crop[d + 1] = resized[s + 1];
                    crop[d + 2] = resized[s + 2];
                }
            }
            frames.push(normalize_frame(&crop, tile, cfg));
        }
    }
    // Global thumbnail LAST: the whole encoder-resized image squashed to tile x tile.
    let global = resize_lanczos(&resized, eh, ew, tile, tile);
    frames.push(normalize_frame(&global, tile, cfg));
    Ok(Idefics3Frames { frames, rows, cols })
}

/// An interleaved-RGB buffer whose length is not `height * width * 3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("an {height}x{width} RGB image has {} values, the buffer holds {len}", height.saturating_mul(*width).saturating_mul(3))]
pub struct ImageBufferError {
    /// The number of values the buffer holds.
    pub len: usize,
    /// The image height in pixels.
    pub height: usize,
    /// The image width in pixels.
    pub width: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smolvlm-256m / idefics3 defaults (rescale 1/255, mean/std 0.5, 512px tiles, 2048px longest edge).
    fn smolvlm() -> PreprocessConfig {
        PreprocessConfig {
            image_size: 512,
            longest_edge: 2048,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    #[test]
    fn preprocess_shape_and_constant_normalization() {
        // SC-001: a constant RGB image preprocesses to [3, S, S] with each channel the expected constant
        // (v*rescale - mean)/std, and all values finite.
        let cfg = PreprocessConfig {
            image_size: 8,
            longest_edge: 32,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        };
        let (h, w) = (5usize, 7usize); // a different size -> exercises the resize.
        let rgb: Vec<f32> = (0..h * w)
            .flat_map(|_| [255.0, 0.0, 128.0]) // constant R=255, G=0, B=128.
            .collect();
        let out = preprocess_image(&rgb, h, w, &cfg);
        let s = cfg.image_size;
        assert_eq!(out.len(), 3 * s * s);
        let want = |v: f32| (v / 255.0 - 0.5) / 0.5;
        for (c, &v0) in [255.0f32, 0.0, 128.0].iter().enumerate() {
            for i in 0..s * s {
                let got = out[c * s * s + i];
                assert!(got.is_finite(), "non-finite at c={c} i={i}");
                assert!(
                    (got - want(v0)).abs() < 1e-5,
                    "c={c}: {got} vs {}",
                    want(v0)
                );
            }
        }
    }

    #[test]
    fn identity_resize_preserves_a_gradient() {
        // SC-001: resizing an already-image_size image is a no-op (the half-pixel convention maps oy->oy),
        // so a per-pixel gradient round-trips through preprocessing as the normalized gradient.
        let cfg = PreprocessConfig {
            image_size: 6,
            longest_edge: 24,
            rescale_factor: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let s = cfg.image_size;
        // R channel = a horizontal ramp, G = vertical ramp, B = 100 constant.
        let mut rgb = vec![0.0f32; s * s * 3];
        for y in 0..s {
            for x in 0..s {
                let p = (y * s + x) * 3;
                rgb[p] = (x * 20) as f32;
                rgb[p + 1] = (y * 20) as f32;
                rgb[p + 2] = 100.0;
            }
        }
        let out = preprocess_image(&rgb, s, s, &cfg);
        for y in 0..s {
            for x in 0..s {
                let r = out[y * s + x];
                let g = out[s * s + y * s + x];
                let b = out[2 * s * s + y * s + x];
                assert!(
                    (r - (x * 20) as f32 / 255.0).abs() < 1e-5,
                    "R at {x},{y}: {r}"
                );
                assert!(
                    (g - (y * 20) as f32 / 255.0).abs() < 1e-5,
                    "G at {x},{y}: {g}"
                );
                assert!((b - 100.0 / 255.0).abs() < 1e-5, "B at {x},{y}: {b}");
            }
        }
    }

    #[test]
    fn idefics3_preprocess_split_grid_matches_reference_formula() {
        // Faithful to image_processing_idefics3.py: longest edge -> 2048 (upscaling small images too), each
        // edge rounded UP to a multiple of 512, then ceil(h/512) x ceil(w/512) tiles + 1 global thumbnail.
        // Expected (rows, cols, n_frames) hand-computed from the source arithmetic (h, w in that order):
        let cfg = smolvlm();
        let cases = [
            // even a tiny square upscales to 2048x2048 -> 4x4 tiles + global = 17 frames.
            ((64usize, 64usize), (4usize, 4usize, 17usize)),
            ((512, 512), (4, 4, 17)),
            // landscape 512x1024 (w>h): -> 1024x2048 -> 2x4 tiles + global = 9.
            ((512, 1024), (2, 4, 9)),
            // 600x800 (w>h, 4:3): -> 1536x2048 -> 3x4 + global = 13.
            ((600, 800), (3, 4, 13)),
            // portrait 800x600 (h>w): -> 2048x1536 -> 4x3 + global = 13.
            ((800, 600), (4, 3, 13)),
            // tall 300x100 (h>w, 1:3): -> 2048x1024 -> 4x2 + global = 9.
            ((300, 100), (4, 2, 9)),
        ];
        for ((h, w), want) in cases {
            let out = idefics3_preprocess(&vec![100.0; h * w * 3], h, w, &cfg).unwrap();
            let got = (out.rows, out.cols, out.frames.len());
            assert_eq!(got, want, "geometry for {h}x{w}");
        }
    }

    #[test]
    fn idefics3_geometry_preserves_aspect_not_squished() {
        // A non-square image is not forced to a square tile grid: a landscape image yields cols > rows and a portrait
        // rows > cols.
        let cfg = smolvlm();
        let wide = idefics3_preprocess(&vec![100.0; 512 * 1536 * 3], 512, 1536, &cfg).unwrap();
        let (lr, lc) = (wide.rows, wide.cols);
        assert!(
            lc > lr,
            "landscape must have more columns than rows: {lr}x{lc}"
        );
        let tall = idefics3_preprocess(&vec![100.0; 1536 * 512 * 3], 1536, 512, &cfg).unwrap();
        let (pr, pc) = (tall.rows, tall.cols);
        assert!(
            pr > pc,
            "portrait must have more rows than columns: {pr}x{pc}"
        );
    }

    #[test]
    fn idefics3_preprocess_extracts_the_right_frames() {
        // Small config (8px tiles, 32px longest edge) keeps the test cheap. A 4x8 (h x w) input:
        // longest->32 => 16x32, encoder-multiple(8) => 16x32, split => 2x4 tiles + global = 9 frames.
        let cfg = PreprocessConfig {
            image_size: 8,
            longest_edge: 32,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        };
        let (h, w) = (4usize, 8usize);
        // a distinguishable gradient so the crops are not trivially identical.
        let mut rgb = vec![0.0f32; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 3;
                rgb[p] = (x * 30) as f32;
                rgb[p + 1] = (y * 30) as f32;
                rgb[p + 2] = 64.0;
            }
        }
        let out = idefics3_preprocess(&rgb, h, w, &cfg).unwrap();
        assert_eq!((out.rows, out.cols), (2, 4));
        assert_eq!(out.frames.len(), 2 * 4 + 1, "8 tiles + 1 global");
        for (i, f) in out.frames.iter().enumerate() {
            assert_eq!(f.len(), 3 * 8 * 8, "frame {i} is [3,8,8]");
            assert!(f.iter().all(|v| v.is_finite()), "frame {i} finite");
        }
    }

    #[test]
    fn lanczos_resize_preserves_a_constant_and_is_finite() {
        // A constant image must resample to the same constant (weights sum to 1), up or down, with no NaN.
        let (sh, sw) = (10usize, 7usize);
        let rgb: Vec<f32> = (0..sh * sw).flat_map(|_| [200.0f32, 50.0, 123.0]).collect();
        for &(oh, ow) in &[(4usize, 3usize), (20, 14), (10, 7)] {
            let out = resize_lanczos(&rgb, sh, sw, oh, ow);
            assert_eq!(out.len(), oh * ow * 3);
            for (i, chunk) in out.as_chunks::<3>().0.iter().enumerate() {
                assert!(chunk.iter().all(|v| v.is_finite()), "nonfinite at {i}");
                assert!((chunk[0] - 200.0).abs() < 0.5, "R {} at {i}", chunk[0]);
                assert!((chunk[1] - 50.0).abs() < 0.5, "G {} at {i}", chunk[1]);
                assert!((chunk[2] - 123.0).abs() < 0.5, "B {} at {i}", chunk[2]);
            }
        }
    }

    #[test]
    fn idefics3_preprocess_handles_degenerate_and_extreme_images() {
        // idefics3 preprocessing feeds a serve endpoint (untrusted images), so it must never panic on a degenerate or
        // extreme-aspect input. Run a battery of pathological sizes through the full pipeline and assert: no panic, the
        // frame count matches the geometry, every frame is [3, S, S] and finite.
        let cfg = PreprocessConfig {
            image_size: 8,
            longest_edge: 32,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        };
        let s = cfg.image_size;
        for &(h, w) in &[
            (1usize, 1usize),
            (1, 64),
            (64, 1),
            (2, 3),
            (1, 500),
            (500, 1),
            (3, 4),
            (7, 129),
        ] {
            let rgb: Vec<f32> = (0..h * w * 3).map(|i| (i % 256) as f32).collect();
            let out = idefics3_preprocess(&rgb, h, w, &cfg).unwrap();
            let n_frames = if out.rows == 0 && out.cols == 0 {
                1
            } else {
                out.rows * out.cols + 1
            };
            assert_eq!(
                out.frames.len(),
                n_frames,
                "{h}x{w}: frame count vs tile grid"
            );
            for (i, f) in out.frames.iter().enumerate() {
                assert_eq!(f.len(), 3 * s * s, "{h}x{w} frame {i} shape");
                assert!(
                    f.iter().all(|v| v.is_finite()),
                    "{h}x{w} frame {i} has a non-finite value"
                );
            }
        }
    }

    /// Bilinear-sample channel `c` of an interleaved RGB image (`rgb[(y*w + x)*3 + c]`) at fractional
    /// `(sx, sy)`, clamping to the image bounds.
    fn sample_bilinear(rgb: &[f32], h: usize, w: usize, sx: f32, sy: f32, c: usize) -> f32 {
        let x0 = sx.floor().clamp(0.0, (w - 1) as f32) as usize;
        let y0 = sy.floor().clamp(0.0, (h - 1) as f32) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = (sx - x0 as f32).clamp(0.0, 1.0);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        let px = |x: usize, y: usize| rgb[(y * w + x) * 3 + c];
        let top = px(x0, y0) * (1.0 - fx) + px(x1, y0) * fx;
        let bot = px(x0, y1) * (1.0 - fx) + px(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Preprocess a decoded `h x w` interleaved-RGB image (values in `[0, 255]`, row-major) into the vision encoder's
    /// channel-first pixel tensor `[3, image_size, image_size]` (spec 049 slice 1, FR-001/FR-002): bilinear resize to
    /// a square, rescale by `rescale_factor`, then per-channel `(x - mean) / std`. A single resized frame, no tiling.
    fn preprocess_image(rgb: &[f32], h: usize, w: usize, cfg: &PreprocessConfig) -> Vec<f32> {
        assert_eq!(rgb.len(), h * w * 3, "rgb must be h*w*3 interleaved");
        let s = cfg.image_size;
        let mut out = vec![0.0f32; 3 * s * s];
        for oy in 0..s {
            // half-pixel-centered source coordinate (the standard resize convention).
            let sy = (oy as f32 + 0.5) * (h as f32 / s as f32) - 0.5;
            for ox in 0..s {
                let sx = (ox as f32 + 0.5) * (w as f32 / s as f32) - 0.5;
                for c in 0..3 {
                    let v = sample_bilinear(rgb, h, w, sx, sy, c);
                    out[c * s * s + oy * s + ox] =
                        (v * cfg.rescale_factor - cfg.mean[c]) / cfg.std[c];
                }
            }
        }
        out
    }
}
