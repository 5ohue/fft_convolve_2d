//-----------------------------------------------------------------------------
use image::buffer::ConvertBuffer;
use num_complex::{Complex, ComplexFloat};
//-----------------------------------------------------------------------------

pub trait ConvolveFFT {
    /// Pixel type of the resulting image
    type ResImageType;

    /// Convolve the image with the kernel
    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType;
}

//-----------------------------------------------------------------------------

impl ConvolveFFT for image::DynamicImage {
    type ResImageType = image::DynamicImage;

    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType {
        return match self {
            Self::ImageLuma8(img) => Self::ImageLuma8(img.convolve(kernel).convert()),
            Self::ImageLumaA8(img) => Self::ImageLumaA8(img.convolve(kernel).convert()),
            Self::ImageRgb8(img) => Self::ImageRgb8(img.convolve(kernel).convert()),
            Self::ImageRgba8(img) => Self::ImageRgba8(img.convolve(kernel).convert()),
            Self::ImageLuma16(img) => Self::ImageLuma16(img.convolve(kernel).convert()),
            Self::ImageLumaA16(img) => Self::ImageLumaA16(img.convolve(kernel).convert()),
            Self::ImageRgb16(img) => Self::ImageRgb16(img.convolve(kernel).convert()),
            Self::ImageRgba16(img) => Self::ImageRgba16(img.convolve(kernel).convert()),
            Self::ImageRgb32F(img) => Self::ImageRgb32F(img.convolve(kernel)),
            Self::ImageRgba32F(img) => Self::ImageRgba32F(img.convolve(kernel)),
            &_ => todo!(),
        };
    }
}

impl<T> ConvolveFFT for image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    T: image::Primitive + Into<f32> + Copy,
    image::Luma<T>: image::Pixel<Subpixel = T>,
{
    type ResImageType = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let max: f32 = <T as image::Primitive>::DEFAULT_MAX_VALUE.into();

        // Convert to Vec<f32>
        let img_f32: Vec<f32> = self.pixels().map(|pixel| pixel.0[0].into() / max).collect();

        let kern_fft = fft_kernel(width, height, kernel);
        let buf = convolve_fft(width, height, &img_f32, &kern_fft);

        return image::ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = (y * width as u32 + x) as usize;
            return image::Luma([buf[idx].abs()]);
        });
    }
}

impl<T> ConvolveFFT for image::ImageBuffer<image::LumaA<T>, Vec<T>>
where
    T: image::Primitive + Into<f32> + Copy,
    image::LumaA<T>: image::Pixel<Subpixel = T>,
{
    type ResImageType = image::ImageBuffer<image::LumaA<f32>, Vec<f32>>;

    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let max: f32 = <T as image::Primitive>::DEFAULT_MAX_VALUE.into();

        // Convert to Vec<f32>
        let mut l_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut a_f32: Vec<f32> = Vec::with_capacity(width * height);

        self.pixels().for_each(|pixel| {
            l_f32.push(pixel.0[0].into() / max);
            a_f32.push(pixel.0[1].into() / max);
        });

        let kern_fft = fft_kernel(width, height, kernel);
        let l_buf = convolve_fft(width, height, &l_f32, &kern_fft);
        let a_buf = convolve_fft(width, height, &a_f32, &kern_fft);

        return image::ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = (y * width as u32 + x) as usize;

            let l_value = l_buf[idx].abs();
            let a_value = a_buf[idx].abs();

            return image::LumaA([l_value, a_value]);
        });
    }
}

impl<T> ConvolveFFT for image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    T: image::Primitive + Into<f32> + Copy,
    image::Rgb<T>: image::Pixel<Subpixel = T>,
{
    type ResImageType = image::ImageBuffer<image::Rgb<f32>, Vec<f32>>;

    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let max: f32 = <T as image::Primitive>::DEFAULT_MAX_VALUE.into();

        // Convert to Vec<f32>
        let mut r_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut g_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut b_f32: Vec<f32> = Vec::with_capacity(width * height);

        self.pixels().for_each(|pixel| {
            r_f32.push(pixel.0[0].into() / max);
            g_f32.push(pixel.0[1].into() / max);
            b_f32.push(pixel.0[2].into() / max);
        });

        let kern_fft = fft_kernel(width, height, kernel);
        let r_buf = convolve_fft(width, height, &r_f32, &kern_fft);
        let g_buf = convolve_fft(width, height, &g_f32, &kern_fft);
        let b_buf = convolve_fft(width, height, &b_f32, &kern_fft);

        return image::ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = (y * width as u32 + x) as usize;

            let r_value = r_buf[idx].abs();
            let g_value = g_buf[idx].abs();
            let b_value = b_buf[idx].abs();

            return image::Rgb([r_value, g_value, b_value]);
        });
    }
}

impl<T> ConvolveFFT for image::ImageBuffer<image::Rgba<T>, Vec<T>>
where
    T: image::Primitive + Into<f32> + Copy,
    image::Rgba<T>: image::Pixel<Subpixel = T>,
{
    type ResImageType = image::ImageBuffer<image::Rgba<f32>, Vec<f32>>;

    fn convolve(&self, kernel: &crate::Kernel) -> Self::ResImageType {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let max: f32 = <T as image::Primitive>::DEFAULT_MAX_VALUE.into();

        // Convert to Vec<f32>
        let mut r_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut g_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut b_f32: Vec<f32> = Vec::with_capacity(width * height);
        let mut a_f32: Vec<f32> = Vec::with_capacity(width * height);

        self.pixels().for_each(|pixel| {
            r_f32.push(pixel.0[0].into() / max);
            g_f32.push(pixel.0[1].into() / max);
            b_f32.push(pixel.0[2].into() / max);
            a_f32.push(pixel.0[3].into() / max);
        });

        let kern_fft = fft_kernel(width, height, kernel);
        let r_buf = convolve_fft(width, height, &r_f32, &kern_fft);
        let g_buf = convolve_fft(width, height, &g_f32, &kern_fft);
        let b_buf = convolve_fft(width, height, &b_f32, &kern_fft);
        let a_buf = convolve_fft(width, height, &a_f32, &kern_fft);

        return image::ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = (y * width as u32 + x) as usize;

            let r_value = r_buf[idx].abs();
            let g_value = g_buf[idx].abs();
            let b_value = b_buf[idx].abs();
            let a_value = a_buf[idx].abs();

            return image::Rgba([r_value, g_value, b_value, a_value]);
        });
    }
}

//-----------------------------------------------------------------------------

fn convolve_fft(
    width: usize,
    height: usize,
    img_buf: &[f32],
    kern_fft: &[Complex<f32>],
) -> Vec<Complex<f32>> {
    let img_fft = fft_2d(width, height, img_buf, rustfft::FftDirection::Forward);

    let mul = img_fft
        .iter()
        .zip(kern_fft)
        .map(|(x, y)| x * y / (width * height) as f32)
        .collect::<Vec<_>>();

    return fft_2d(width, height, &mul, rustfft::FftDirection::Inverse);
}

pub fn fft_2d<T>(
    width: usize,
    height: usize,
    buffer: &[T],
    direction: rustfft::FftDirection,
) -> Vec<Complex<f32>>
where
    T: Into<Complex<f32>> + Copy,
{
    fn fft_shift<T: Copy>(data: &mut [T]) {
        let mid = data.len() / 2;
        if data.len() % 2 == 0 {
            for (i1, i2) in (0..mid).zip(mid..data.len()) {
                data.swap(i1, i2);
            }
        } else {
            for (i1, i2) in (0..mid).zip(mid..data.len()) {
                data.swap(i1, i2);
                data.swap(i1, i2 + 1);
            }
        }
    }
    fn transpose<T: Copy + Default>(width: usize, height: usize, matrix: &[T]) -> Vec<T> {
        let mut transposed = vec![T::default(); matrix.len()];

        for i in 0..height {
            for j in 0..width {
                transposed[j * height + i] = matrix[i * width + j];
            }
        }

        transposed
    }

    let mut planner = rustfft::FftPlanner::<f32>::new();
    let line_fft = planner.plan_fft(width, direction);
    let col_fft = planner.plan_fft(height, direction);

    // Process lines
    let mut res: Vec<Complex<f32>> = buffer.iter().map(|&x| x.into()).collect();

    for line in res.chunks_exact_mut(width) {
        line_fft.process(line);
        fft_shift(line);
    }

    // Process columns
    res = transpose(width, height, &res);

    for col in res.chunks_exact_mut(height) {
        col_fft.process(col);
        fft_shift(col);
    }

    transpose(height, width, &res)
}

fn fft_kernel(img_width: usize, img_height: usize, kernel: &crate::Kernel) -> Vec<Complex<f32>> {
    // Extend kernel with 0
    let kern_f32 = extend_kernel_with(kernel, img_width, img_height, 0.0);

    return fft_2d(
        img_width,
        img_height,
        &kern_f32,
        rustfft::FftDirection::Forward,
    );
}

fn extend_kernel_with(
    kernel: &crate::Kernel,
    res_width: usize,
    res_height: usize,
    what: f32,
) -> Vec<f32> {
    let src_width = kernel.width();
    let src_height = kernel.height();

    let mut buf: Vec<f32> = Vec::with_capacity(res_width * res_height);

    for i in 0..res_height {
        for j in 0..res_width {
            let (mut ii, mut ij) = (i as isize, j as isize);

            ii -= (res_height / 2) as isize - (src_height / 2) as isize;
            ij -= (res_width / 2) as isize - (src_width / 2) as isize;

            if ii < 0 || ii >= src_height as isize {
                buf.push(what);
                continue;
            }
            if ij < 0 || ij >= src_width as isize {
                buf.push(what);
                continue;
            }

            buf.push(kernel.get_pixel(ii as u32, ij as u32).0[0]);
        }
    }

    return buf;
}

//-----------------------------------------------------------------------------
