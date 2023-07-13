use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{boxed::Box, f64::consts::PI};
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

struct HitRecord<'a> {
    point: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: &'a dyn Material,
}

impl<'a> HitRecord<'a> {
    fn new(
        point: &Point3,
        outward_normal: &Vec3,
        ray: &Ray,
        t: f64,
        material: &'a dyn Material,
    ) -> Self {
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
            front_face,
            material,
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
    material: Box<dyn Material + Sync>,
}

impl Sphere {
    fn new(center: &Point3, radius: f64, material: Box<dyn Material + Sync>) -> Sphere {
        Sphere {
            center: *center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn try_collect_hit_from(&self, ray: &Ray, threshold: &Threshold) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
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

        return Some(HitRecord::new(
            &point,
            &outward_normal,
            ray,
            root,
            self.material.as_ref(),
        ));
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
        min: 0.001,
        max: f64::INFINITY,
    };

    if let Some(hit) = hittable.try_collect_hit_from(ray, &base_threshold) {
        if let Some(scatter) = hit.material.scatter(ray, &hit, random) {
            return scatter.attenuation
                * scatter.scattered.resolve_color(hittable, random, depth - 1);
        }

        return ColorVec::ZERO;
    }

    let unit_direction = ray.direction.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * ColorVec::new(1.0, 1.0, 1.0) + t * ColorVec::new(0.5, 0.7, 1.0);
}

fn double_to_color(val: f64) -> u8 {
    (256 as f64 * val.clamp(0.0, 0.99)) as u8
}

fn vec_to_image_color(color_vec: &ColorVec) -> Color {
    rgb(
        double_to_color(color_vec.x),
        double_to_color(color_vec.y),
        double_to_color(color_vec.z),
    )
}

fn degrees_to_radians(degrees: f64) -> f64 {
    return degrees * PI / 180.0;
}

struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,
    w: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        look_from: Point3,
        look_to: Point3,
        vup: Vec3,
        vertical_fov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64
    ) -> Camera {
        let theta = degrees_to_radians(vertical_fov);
        let h = (theta / 2.0).tan();

        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_to).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let origin = look_from;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
        let lens_radius = aperture / 2.0;
        Camera {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
            w,
            u,
            v,
            lens_radius,
        }
    }

    fn get_ray(&self, s: f64, t: f64, random: &mut Random) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(random);
        let offset = self.u * rd.x + self.v * rd.y;

        let direction: Vec3 =
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset;
        Ray::new(&(self.origin + offset), &direction)
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

fn create_world_2() -> impl Hittable {
    let mut world = HittableList::new();

    let material_left = Box::new(Lambertian {
        albedo: ColorVec::new(0.0, 0.0, 1.0),
    });
    let material_right = Box::new(Lambertian {
        albedo: ColorVec::new(1.0, 0.0, 0.0),
    });

    let R = (PI / 4.0).cos();

    world.add(Box::new(Sphere::new(
        &Point3::new(-R, 0.0, -1.0),
        R,
        material_left,
    )));

    world.add(Box::new(Sphere::new(
        &Point3::new(R, 0.0, -1.0),
        R,
        material_right,
    )));

    return world;
}

fn create_world() -> impl Hittable {
    let mut world = HittableList::new();

    let material_ground = Box::new(Lambertian {
        albedo: ColorVec::new(0.8, 0.8, 0.0),
    });
    let material_center = Box::new(Lambertian {
        albedo: ColorVec::new(0.1, 0.2, 0.5),
    });
    let material_left = Box::new(Dialectric { ir: 1.5 });
    // TODO: The Hittable should probably not own the Material...
    let material_left_2 = Box::new(Dialectric { ir: 1.5 });
    let material_right = Box::new(Metal::new(ColorVec::new(0.8, 0.6, 0.2), 0.0));

    world.add(Box::new(Sphere::new(
        &Point3::new(0.0, -100.5, -1.0),
        100.0,
        material_ground,
    )));

    world.add(Box::new(Sphere::new(
        &Point3::new(0.0, 0.0, -1.0),
        0.5,
        material_center,
    )));

    world.add(Box::new(Sphere::new(
        &Point3::new(-1.0, 0.0, -1.0),
        0.5,
        material_left,
    )));

    world.add(Box::new(Sphere::new(
        &Point3::new(-1.0, 0.0, -1.0),
        -0.4,
        material_left_2,
    )));

    world.add(Box::new(Sphere::new(
        &Point3::new(1.0, 0.0, -1.0),
        0.5,
        material_right,
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

struct ScatterRecord {
    attenuation: ColorVec,
    scattered: Ray,
}

trait Material {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, random: &mut Random) -> Option<ScatterRecord>;
}

struct Lambertian {
    albedo: ColorVec,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord, random: &mut Random) -> Option<ScatterRecord> {
        let mut scatter_direction = hit.normal + random_unit_vector(random);
        if near_zero(&scatter_direction) {
            scatter_direction = hit.normal;
        }
        return Some(ScatterRecord {
            attenuation: self.albedo,
            scattered: Ray::new(&hit.point, &scatter_direction),
        });
    }
}

fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    return *v - 2.0 * v.dot(*n) * *n;
}

fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = uv.neg().dot(*n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * (*n));

    let r_out_parallell = (1.0 - r_out_perp.length_squared()).abs().sqrt().neg() * *n;
    return r_out_perp + r_out_parallell;
}

struct Dialectric {
    ir: f64,
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Use Schlick's approximation for reflectance.
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0_sqared = r0 * r0;
    return r0_sqared + (1.0 - r0_sqared) * (1.0 - cosine).powi(5);
}

impl Material for Dialectric {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, random: &mut Random) -> Option<ScatterRecord> {
        let refraction_ratio = if hit.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };
        let unit_direction = ray.direction.normalize();

        let cos_theta = unit_direction.neg().dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction = if cannot_refract
            || reflectance(cos_theta, refraction_ratio) > random.random_f64(0.0, 1.0)
        {
            reflect(&unit_direction, &hit.normal)
        } else {
            refract(&unit_direction, &hit.normal, refraction_ratio)
        };

        return Some(ScatterRecord {
            attenuation: ColorVec::ONE,
            scattered: Ray::new(&hit.point, &direction),
        });
    }
}

struct Metal {
    albedo: ColorVec,
    fuzz: f64,
}

impl Metal {
    fn new(albedo: ColorVec, fuzz: f64) -> Metal {
        Metal {
            albedo,
            fuzz: fuzz.min(1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, random: &mut Random) -> Option<ScatterRecord> {
        let reflected = reflect(&(ray.direction.normalize()), &hit.normal);
        let scattered = Ray::new(
            &hit.point,
            &(reflected + self.fuzz * random_in_unit_sphere(random)),
        );
        return if scattered.direction.dot(hit.normal) < 0.0 {
            None
        } else {
            Some(ScatterRecord {
                attenuation: self.albedo,
                scattered,
            })
        };
    }
}

fn normalize_coord_with_noise(val: u32, bound: u32, random: &mut Random) -> f64 {
    let noise = random.random_f64(0.0, 1.0);
    ((val as f64) + noise) / bound as f64
}

fn random_vec_bounded(random: &mut Random, min: f64, max: f64) -> Vec3 {
    Vec3::new(
        random.random_f64(min, max),
        random.random_f64(min, max),
        random.random_f64(min, max),
    )
}

fn random_in_unit_sphere(random: &mut Random) -> Vec3 {
    return loop {
        let vec = random_vec_bounded(random, -1.0, 1.0);
        if vec.length_squared() < 1.0 {
            break vec;
        }
    };
}

fn random_unit_vector(random: &mut Random) -> Vec3 {
    random_in_unit_sphere(random).normalize()
}

fn random_in_hemisphere(normal: &Vec3, random: &mut Random) -> Vec3 {
    let in_unit_sphere = random_in_unit_sphere(random);
    // In the same hemisphere as the normal
    return if in_unit_sphere.dot(*normal) > 0.0 {
        in_unit_sphere
    } else {
        return in_unit_sphere.neg();
    };
}

fn random_in_unit_disk(random: &mut Random) -> Vec3 {
    return loop {
        let p = Vec3::new(random.random_f64(-1.0, 1.0), random.random_f64(-1.0, 1.0), 0.0);
        if p.length_squared() < 1.0 {
            break p;
        }
    }
}

fn near_zero(vec: &Vec3) -> bool {
    let s = 1e-8;
    return vec.abs().max_element() < s;
}

fn sqrt_vec(vec: &ColorVec) -> ColorVec {
    ColorVec::new(vec.x.sqrt(), vec.y.sqrt(), vec.z.sqrt())
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
    let mut random = Random::new();

    for _ in 0..samples_per_pixel {
        let u = normalize_coord_with_noise(x, image_spec.width, &mut random);
        // Flip top and bottom, as guide's x=0 is bottom, but our's is top.
        let v = 1.0 - normalize_coord_with_noise(y, image_spec.height, &mut random);
        let color = camera
            .get_ray(u, v, &mut random)
            .resolve_color(world, &mut random, max_depth);

        final_color += color;
    }

    let sample_scale = 1.0 / (samples_per_pixel as f64);
    let sample_adjusted_color = final_color * sample_scale;
    let gamma_corrected_color = sqrt_vec(&sample_adjusted_color);
    return gamma_corrected_color;
}

fn main() {
    let image_spec = ImageSpec::from_aspect_ratio(400, 16.0 / 9.0);
    let samples_per_pixel = 100;

    let look_from = Point3::new(3.0, 3.0, 2.0);
    let look_to = Point3::new(0.0, 0.0, -1.0);
    let vup = Point3::new(0.0, 1.0, 0.0);
    let dist_to_focus = (look_from - look_to).length();
    let aperture = 0.5;

    let camera = Camera::new(
        look_from,
        look_to,
        vup,
        20.0,
        image_spec.aspect_ratio,
        aperture,
        dist_to_focus,
    );
    let world = create_world();

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
