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

struct HitRecord {
    point: Point3,
    normal: Vec3,
    t: f64,
}

struct Threshold {
    min: f64,
    max: f64,
}

trait Hittable {
    fn try_collect_hit_from(&self, ray: &Ray, threshold: &Threshold) -> Option<HitRecord>;
}

impl HitRecord {
    fn new(point: &Point3, normal: &Vec3, t: f64) -> HitRecord {
        HitRecord {
            point: *point,
            normal: *normal,
            t,
        }
    }
}

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

impl Hittable for Sphere {
    fn try_collect_hit_from(&self, ray: &Ray, threshold: &Threshold) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.box_area();
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if root < threshold.min || threshold.max < root {
            root = (-half_b + sqrtd) / a;
            if root < threshold.min || threshold.max < root {
                return None;
            }
        }

        let point = ray.at(root);
        let normal = (point - self.center) / self.radius;

        return Some(HitRecord::new(
            &point,
            &normal,
            root,
        ));
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
}

fn ray_color(ray: &Ray, hittable: &dyn Hittable, threshold: &Threshold) -> ColorVec {
    if let Some(hit) = hittable.try_collect_hit_from(ray, threshold) {
        let N = (ray.at(hit.t) - Vec3::new(0.0, 0.0, -1.0)).normalize();
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

    let sphere = Sphere::new(&Point3::new(0.0, 0.0, -1.0), 0.5);
    // TODO: What should these be?
    let threshold = Threshold {
        min: 0.0,
        max: 1.0,
    };

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let u = x as f64 / image_width as f64;
        let v = y as f64 / image_height as f64;
        let direction = lower_left_corner + u * horizontal + v * vertical - origin;
        let ray = Ray::new(&origin, &direction);
        let color_vec = ray_color(&ray, &sphere, &threshold);
        *pixel = vec_to_image_color(&color_vec);
        progress.inc(1);
    }

    image.save("/tmp/image.png").unwrap();
}
