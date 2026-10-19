//! Decoding an image's bytes: PNG or JPEG to RGB8, alpha dropped. Which pixels a model wants of it is
//! the model's front end's.

use poot_models::multimodal::front_end::{Image, ImageError};

/// Upper bound on a decoded image's pixel count, guarding against a decompression bomb (a tiny file
/// that decodes to huge dimensions, blowing up the `w*h*3` RGB buffer and the f32 expansion a front end
/// makes of it). About 32 megapixels is far above any real input. The `image` crate's default limits are
/// not honored by every decoder, so the header dimensions are checked directly.
const MAX_IMAGE_PIXELS: u64 = 32 * 1024 * 1024;

/// Why bytes are not an image.
#[derive(Debug, thiserror::Error)]
pub enum ImageDecodeError {
    /// The bytes are not a decodable PNG or JPEG.
    #[error("decode image: {0}")]
    Image(#[from] image::ImageError),
    /// The header's dimensions exceed the pixel cap; refused before any pixel is decoded.
    #[error("image {width}x{height} ({pixels} pixels) exceeds the {MAX_IMAGE_PIXELS}-pixel cap")]
    TooLarge {
        /// Header width in pixels.
        width: u32,
        /// Header height in pixels.
        height: u32,
        /// `width * height`, the count compared against the cap.
        pixels: u64,
    },
    /// The decoded pixels are not a valid image buffer.
    #[error(transparent)]
    Buffer(#[from] ImageError),
}

/// Decode PNG or JPEG bytes: interleaved RGB8, alpha dropped. The header's dimensions are checked
/// against the pixel cap before any pixel is decoded.
pub fn decode_image(bytes: &[u8]) -> Result<Image, ImageDecodeError> {
    let (width, height) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?
        .into_dimensions()?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels > MAX_IMAGE_PIXELS {
        return Err(ImageDecodeError::TooLarge {
            width,
            height,
            pixels,
        });
    }
    let rgb = image::load_from_memory(bytes)?.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    Ok(Image::from_rgb8(height, width, rgb.into_raw())?)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use base64::Engine as _;
    use image::ImageFormat;

    use super::*;

    fn rgb_fixture(height: usize, width: usize) -> Vec<u8> {
        (0..height * width)
            .flat_map(|pixel| {
                let value = (pixel % 180) as u8;
                [value, value + 20, value + 40]
            })
            .collect()
    }

    /// PNG and JPEG bytes decode to their pixels (a PNG exactly, a JPEG lossily), and bytes that are
    /// no image are the decoder's typed error.
    #[test]
    fn png_and_jpeg_bytes_decode_to_their_pixels() {
        let source = image::RgbImage::from_raw(28, 28, rgb_fixture(28, 28)).unwrap();
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(source.clone())
                .write_to(&mut bytes, format)
                .unwrap();
            let decoded = decode_image(bytes.get_ref()).unwrap();
            assert_eq!((decoded.height(), decoded.width()), (28, 28));
            if format == ImageFormat::Png {
                assert_eq!(decoded.rgb(), source.as_raw());
            }
        }
        assert!(matches!(
            decode_image(b"not an image"),
            Err(ImageDecodeError::Image(_))
        ));
        // Only PNG and JPEG are compiled in: a GIF is no image to this decoder.
        assert!(matches!(
            decode_image(b"GIF89a"),
            Err(ImageDecodeError::Image(_))
        ));
    }

    /// An image whose header declares more pixels than the cap is refused before it decodes, and the
    /// pixels of PNGs and a JPEG another encoder wrote are the RGB8 the decoder promises: alpha dropped,
    /// luminance replicated.
    #[test]
    fn the_header_gates_the_pixel_cap_and_other_encoders_pixels_become_rgb() {
        // Emitted by Pillow, not the image crate under test. PIL `convert("RGB")` discards alpha and
        // replicates 8-bit luminance, matching the decoded RGB8 contract.
        for (encoded, expected) in [
            (
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGNkZGJmYGZm/g8AAVQBEGAwCA8AAAAASUVORK5CYII=",
                &[1, 2, 3, 4, 5, 6][..],
            ),
            (
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAAAAADRSSBWAAAAC0lEQVR4nGNgPwEAANkA0NemIjwAAAAASUVORK5CYII=",
                &[7, 7, 7, 200, 200, 200][..],
            ),
        ] {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap();
            let decoded = decode_image(&bytes).unwrap();
            assert_eq!((decoded.height(), decoded.width()), (1, 2));
            assert_eq!(decoded.rgb(), expected);
        }

        // Pillow-emitted JPEG, so this crate's encoder is not its own oracle. JPEG pixels are lossy.
        let jpeg = base64::engine::general_purpose::STANDARD
            .decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAIDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDtv2O/+TR/hZ/2TjQ//SCCvw/PP+R1if8Ar5P/ANKZ+x5P/wAijD/9e4f+ko//2Q==")
            .unwrap();
        let decoded = decode_image(&jpeg).unwrap();
        assert_eq!((decoded.height(), decoded.width()), (1, 2));

        // A valid PNG IHDR advertising 8192x4097 with no pixel stream: the header-only gate must
        // return the cap error before any pixel allocation or missing-data error.
        let oversized_header = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAIAAAABABCAIAAACe9yv5AAAACElEQVR4nAMAAAAAAUgGidIAAAAASUVORK5CYII=")
            .unwrap();
        assert!(matches!(
            decode_image(&oversized_header),
            Err(ImageDecodeError::TooLarge {
                height: 4097,
                width: 8192,
                ..
            })
        ));
    }

    /// A header of 8192 x 8192 pixels (64 megapixels) is refused by the cap, built by rewriting a
    /// 1x1 PNG's IHDR and its chunk CRC-32 (which covers its type and data: the decoder checks it before
    /// it reports the header).
    #[test]
    fn an_oversized_header_is_refused_before_any_pixel_decodes() {
        let mut png = Vec::new();
        image::RgbImage::new(1, 1)
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        png[16..20].copy_from_slice(&8192u32.to_be_bytes());
        png[20..24].copy_from_slice(&8192u32.to_be_bytes());
        let mut crc = u32::MAX;
        for &byte in &png[12..29] {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        png[29..33].copy_from_slice(&(!crc).to_be_bytes());
        assert!(matches!(
            decode_image(&png),
            Err(ImageDecodeError::TooLarge {
                width: 8192,
                height: 8192,
                ..
            })
        ));
    }
}
