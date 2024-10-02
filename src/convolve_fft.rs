//-----------------------------------------------------------------------------
use crate::{fft_2d, fft_kernel};
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

//-----------------------------------------------------------------------------
