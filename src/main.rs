use glam::DVec3;
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::ProgressBar;

fn rgb(r: u8, g: u8, b: u8) -> Rgb<u8> {
    Rgb([r, g, b])
}

type Color = Rgb<u8>;
type ColorVec = DVec3;
type Vec3 = DVec3;
type Point3 = DVec3;

struct Sphere {
    center: Point3,
    radius: f64,
}

impl Sphere {
    fn new(center: &Point3, radius: f64) -> Sphere {
        Sphere {
            center: *center,
            radius,
        }
    }

    fn box_area(&self) -> f64 {
        self.radius * self.radius
    }
}

struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: &Point3, direction: &Vec3) -> Ray {
        Ray {
            origin: *origin,
            direction: *direction,
        }
    }

    fn at(&self, t: f64) -> Point3 {
        self.origin + t * self.direction
    }

    fn hit_sphere(&self, sphere: &Sphere) -> Option<f64> {
        let oc = self.origin - sphere.center;
        let a = self.direction.length_squared();
        let half_b = oc.dot(self.direction);
        let c = oc.length_squared() - sphere.box_area();
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }
        return Some((-half_b - discriminant.sqrt()) / a);
    }
}

fn ray_color(ray: &Ray) -> ColorVec {
    let sphere = Sphere::new(&Point3::new(0.0, 0.0, -1.0), 0.5);
    if let Some(t) = ray.hit_sphere(&sphere) {
        let N = (ray.at(t) - Vec3::new(0.0, 0.0, -1.0)).normalize();
        return 0.5 * (N + 1.0);
    }

    let unit_direction = ray.direction.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * ColorVec::new(1.0, 1.0, 1.0) + t * ColorVec::new(0.5, 0.7, 1.0);
}

fn double_to_color(val: f64) -> u8 {
    assert!(val >= 0.0);
    let scaled: u32 = (val * 255.999) as u32;
    assert!(scaled < 256, "scaled={}, val={}", scaled, val);
    return scaled as u8;
}

fn vec_to_image_color(color_vec: &ColorVec) -> Color {
    rgb(
        double_to_color(color_vec.x),
        double_to_color(color_vec.y),
        double_to_color(color_vec.z),
    )
}

fn main() {
    // Image
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: u32 = 400;
    let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;

    // Camera
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = Point3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

    let mut image: RgbImage = ImageBuffer::new(image_width, image_height);
    let progress = ProgressBar::new((image_width * image_height) as u64);

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let u = x as f64 / image_width as f64;
        let v = y as f64 / image_height as f64;
        let direction = lower_left_corner + u * horizontal + v * vertical - origin;
        let ray = Ray::new(&origin, &direction);
        let color_vec = ray_color(&ray);
        *pixel = vec_to_image_color(&color_vec);
        progress.inc(1);
    }

    image.save("/tmp/image.png").unwrap();
}
