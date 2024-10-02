//-----------------------------------------------------------------------------
pub mod convolve_fft;
pub mod kernel;
//-----------------------------------------------------------------------------
pub use convolve_fft::*;
pub use kernel::Kernel;
//-----------------------------------------------------------------------------

use num_complex::Complex;

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

    // Extend kernel with 0
    let kern_f32 = extend_kernel_with(kernel, img_width, img_height, 0.0);

    return fft_2d(
        img_width,
        img_height,
        &kern_f32,
        rustfft::FftDirection::Forward,
    );
}

//-----------------------------------------------------------------------------
