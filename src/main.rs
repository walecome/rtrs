use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};

fn get_pixel_color(x: u32, y: u32, image_width: u32, image_height: u32) -> Rgb<u8> {
    let pct_x = x as f32 / image_width as f32;
    let r = (0xFF as f32 * pct_x) as u8;

    let pct_y = y as f32 / image_height as f32;
    let g = (0xFF as f32 * pct_y) as u8;

    return Rgb([r, g, 0xFF / 4]);
}

fn main() {
    let image_width: u32 = 256;
    let image_height: u32 = 256;

    let mut image: RgbImage = ImageBuffer::new(image_width, image_height);

    let progress = ProgressBar::new((image_width * image_height) as u64);

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = get_pixel_color(x, y, image_width, image_height);
        progress.inc(1);
    }

    image.save("/tmp/image.png").unwrap();
}
