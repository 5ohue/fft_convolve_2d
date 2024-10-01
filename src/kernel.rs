//-----------------------------------------------------------------------------
pub type Kernel = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;
//-----------------------------------------------------------------------------

/// Generate kernel from a radial function
///
/// * `size`: the pixel size of kernel
/// * `func`: the function which takes the squared length from center
pub fn generate_radial<F>(size: u32, mut func: F) -> Kernel
where
    F: FnMut(f32) -> f32,
{
    let mid = size as f32 * 0.5;

    let mut kernel = Kernel::from_fn(size, size, |x, y| {
        let f_x = x as f32 + 0.5 - mid;
        let f_y = y as f32 + 0.5 - mid;

        let l2 = (f_x * f_x) + (f_y * f_y);

        return image::Luma([func(l2)]);
    });

    normalize(&mut kernel);

    return kernel;
}

//-----------------------------------------------------------------------------

/// Normalize the kernel to have all of the values sum to 1.0
pub fn normalize(kernel: &mut Kernel) {
    // Calculate sum value
    let sum = kernel.pixels().fold(0.0, |acc, pixel| {
        return acc + pixel.0[0];
    });

    if sum < f32::EPSILON {
        return;
    }

    kernel.pixels_mut().for_each(|pixel| {
        pixel.0 = [pixel.0[0] / sum];
    });
}

//-----------------------------------------------------------------------------

/// Kernel: [1]
pub fn generate_identity() -> Kernel {
    return Kernel::from_pixel(1, 1, image::Luma([1.0]));
}

/// Gaussian blur kernel
pub fn generate_gauss(size: u32, sigma: f32) -> Kernel {
    return generate_radial(size, |l2| return f32::exp(-l2 / (2.0 * sigma * sigma)));
}

/// Exponential function kernel
pub fn generate_exp(size: u32, pow: f32, sigma: f32) -> Kernel {
    return generate_radial(size, |l2| {
        return f32::exp(-f32::powf(l2, pow) / f32::powf(sigma, pow));
    });
}

/// Polynomial function kernel
pub fn generate_poly(size: u32, pow: f32, sigma: f32) -> Kernel {
    return generate_radial(size, |l2| {
        if l2 > sigma * sigma {
            return 0.0;
        }
        return (1.0 - l2.sqrt() / sigma).powf(pow);
    });
}

pub fn generate_smoothify(size: u32, sigma: f32) -> Kernel {
    let smoothify_curve = |l2: f32| {
        const NUM_OF_GAUSES: usize = 5;
        let mut val = 0.0;

        for k in 1..=NUM_OF_GAUSES {
            let s = (1 << k) as f32 * sigma;
            val += (-l2 * 0.5 / (s * s)).exp() / (s * s) / (1 << (NUM_OF_GAUSES - k)) as f32;
        }

        return val;
    };

    return generate_radial(size, smoothify_curve);
}

//-----------------------------------------------------------------------------
