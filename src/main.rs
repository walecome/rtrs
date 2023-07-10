use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::boxed::Box;
use std::ops::Neg;

use glam::DVec3;
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

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
    // front_face: bool,
}

impl HitRecord {
    fn new(point: &Point3, outward_normal: &Vec3, ray: &Ray, t: f64) -> HitRecord {
        let front_face = ray.direction.dot(*outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal.to_owned()
        } else {
            outward_normal.neg()
        };

        HitRecord {
            point: *point,
            normal,
            t,
            // front_face,
        }
    }
}

#[derive(Clone, Copy)]
struct Threshold {
    min: f64,
    max: f64,
}

impl Threshold {
    fn with_max(&self, new_max: f64) -> Threshold {
        Threshold {
            min: self.min,
            max: new_max,
        }
    }
}

trait Hittable {
    fn try_collect_hit_from(&self, ray: &Ray, threshold: &Threshold) -> Option<HitRecord>;
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
        let outward_normal = (point - self.center) / self.radius;

        return Some(HitRecord::new(&point, &outward_normal, ray, root));
    }
}

struct HittableList {
    objects: Vec<Box<dyn Hittable + Sync>>,
}

impl HittableList {
    fn new() -> HittableList {
        HittableList { objects: vec![] }
    }

    fn add(&mut self, hittable: Box<dyn Hittable + Sync>) {
        self.objects.push(hittable);
    }
}

impl Hittable for HittableList {
    fn try_collect_hit_from(&self, ray: &Ray, threshold: &Threshold) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut narrowing_threshold = threshold.clone();

        for obj in &self.objects {
            if let Some(hit) = obj.try_collect_hit_from(ray, &narrowing_threshold) {
                narrowing_threshold = narrowing_threshold.with_max(hit.t);
                closest_hit = Some(hit);
            }
        }

        return closest_hit;
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

    fn resolve_color(&self, hittable: &dyn Hittable, random: &mut Random, depth: u32) -> ColorVec {
        ray_color(self, hittable, random, depth)
    }
}

fn ray_color(ray: &Ray, hittable: &dyn Hittable, random: &mut Random, depth: u32) -> ColorVec {
    if depth == 0 {
        return ColorVec::ZERO;
    }
    let base_threshold = Threshold {
        min: 0.0,
        max: f64::INFINITY,
    };

    if let Some(hit) = hittable.try_collect_hit_from(ray, &base_threshold) {
        let target = hit.point + hit.normal + random_unit_sphere(random);
        let new_ray = Ray::new(&hit.point, &(target - hit.point));
        return 0.5 * ray_color(&new_ray, hittable, random, depth - 1);
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

struct Camera {
    // aspect_ratio: f64,
    // viewport_height: f64,
    // viewport_width: f64,
    // focal_length: f64,
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,
}

impl Camera {
    fn new(aspect_ratio: f64) -> Camera {
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);
        Camera {
            // aspect_ratio,
            // viewport_height,
            // viewport_width,
            // focal_length,
            origin,
            horizontal,
            vertical,
            lower_left_corner,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        let direction: Vec3 =
            self.lower_left_corner + u * self.horizontal + v * self.vertical + self.origin;
        Ray::new(&self.origin, &direction)
    }
}

struct ImageSpec {
    width: u32,
    height: u32,
    aspect_ratio: f64,
}

impl ImageSpec {
    fn from_aspect_ratio(width: u32, aspect_ratio: f64) -> ImageSpec {
        let height: u32 = (width as f64 / aspect_ratio) as u32;
        ImageSpec {
            width,
            height,
            aspect_ratio,
        }
    }

    fn pixel_count(&self) -> u32 {
        self.width * self.height
    }
}

fn create_world() -> impl Hittable {
    let mut world = HittableList::new();
    world.add(Box::new(Sphere::new(&Point3::new(0.0, 0.0, -1.0), 0.5)));
    world.add(Box::new(Sphere::new(
        &Point3::new(0.0, -100.5, -1.0),
        100.0,
    )));

    return world;
}

struct Random {
    rng: SmallRng,
}

impl Random {
    fn new() -> Random {
        Random {
            rng: SmallRng::from_entropy(),
        }
    }

    fn random_normalized(&mut self) -> f64 {
        self.rng.gen::<f64>()
    }

    fn random_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.random_normalized()
    }
}

fn normalize_coord_with_noise(val: u32, bound: u32, random: &mut Random) -> f64 {
    let noise = random.random_f64(0.0, 1.0);
    ((val as f64) + noise) / bound as f64
}

// fn random_vec(random: &mut Random) -> Vec3 {
//     Vec3::new(random.random_normalized(), random.random_normalized(), random.random_normalized())
// }

fn random_vec_bounded(random: &mut Random, min: f64, max: f64) -> Vec3 {
    Vec3::new(
        random.random_f64(min, max),
        random.random_f64(min, max),
        random.random_f64(min, max),
    )
}

fn random_unit_sphere(random: &mut Random) -> Vec3 {
    return loop {
        let vec = random_vec_bounded(random, -1.0, 1.0);
        if vec.length_squared() < 1.0 {
            break vec;
        }
    };
}

fn compute_pixel(
    x: u32,
    y: u32,
    samples_per_pixel: u32,
    image_spec: &ImageSpec,
    camera: &Camera,
    world: &dyn Hittable,
    max_depth: u32,
) -> ColorVec {
    let mut final_color = ColorVec::ZERO;
    let sample_scale = 1.0 / (samples_per_pixel as f64);
    let mut random = Random::new();

    for _ in 0..samples_per_pixel {
        let u = normalize_coord_with_noise(x, image_spec.width, &mut random);
        // Flip top and bottom, as guide's x=0 is bottom, but our's is top.
        let v = 1.0 - normalize_coord_with_noise(y, image_spec.height, &mut random);
        let color = camera
            .get_ray(u, v)
            .resolve_color(world, &mut random, max_depth);

        final_color += color * sample_scale;
    }

    return final_color;
}

fn main() {
    let image_spec = ImageSpec::from_aspect_ratio(400, 16.0 / 9.0);
    let samples_per_pixel = 100;

    let world = create_world();
    let camera = Camera::new(image_spec.aspect_ratio);

    let max_depth = 50;

    let result = (0..image_spec.pixel_count())
        .into_par_iter()
        .progress_count(image_spec.pixel_count() as u64)
        .map(|i| {
            let x = i % image_spec.width;
            let y = i / image_spec.width;
            let color_vec = compute_pixel(
                x,
                y,
                samples_per_pixel,
                &image_spec,
                &camera,
                &world,
                max_depth,
            );
            (x, y, vec_to_image_color(&color_vec))
        })
        .collect::<Vec<_>>();

    let mut image: RgbImage = ImageBuffer::new(image_spec.width, image_spec.height);
    result.iter().for_each(|info| {
        let (x, y, color) = *info;
        image.put_pixel(x, y, color);
    });

    image.save("/tmp/image.png").unwrap();
}
