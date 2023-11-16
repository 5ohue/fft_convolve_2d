use num_complex::Complex;

//-----------------------------------------------------------------------------

pub struct ImageFloat<'a, T> {
    width:  usize,
    height: usize,
    buffer: &'a [T]
}

impl<'a, T> ImageFloat<'a, T> {
    pub fn new(
        width:  usize,
        height: usize,
        buffer: &'a [T]
    ) -> Self {
        ImageFloat { width, height, buffer }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn buffer(&self) -> &[T] {
        self.buffer
    }
}

impl<'a, T> ImageFloat<'a, T>
where
    T: Into<Complex<f32>> + From<f32> + Copy
{
    pub fn fft_2d(&self, direction: rustfft::FftDirection) -> Vec<Complex<f32>>
    where
        T: Into<Complex<f32>> + Copy
    {
        fft_2d(self.width, self.height, self.buffer, direction)
    }

    pub fn convolve(
        rgb: [&ImageFloat<'a, T>; 3],
        kernel: &Self
    ) -> [Vec<Complex<f32>>; 3] {
        let dims = rgb[0].dimensions();
        let resized_kernel_data = kernel.extend_image_with(dims.0, dims.1, 0.0.into());
        let resized_kernel = ImageFloat::new(dims.0, dims.1, &resized_kernel_data);
        let kern_fft = resized_kernel.fft_2d(rustfft::FftDirection::Forward);

        rgb.map(|img| {
            let img_fft = img.fft_2d(rustfft::FftDirection::Forward);

            let mul = img_fft.iter()
                .zip(kern_fft.iter())
                .map(|(x, y)| { x*y / (dims.0*dims.1) as f32 })
                .collect::<Vec<Complex<f32>>>();
            fft_2d(dims.0, dims.1, &mul, rustfft::FftDirection::Inverse)
        })
    }

    pub fn extend_image_with(&self, res_width: usize, res_height: usize, what: T) -> Vec<T> {
        let (src_width, src_height) = self.dimensions();
        let mut res: Vec<T> = Vec::with_capacity(res_width*res_height);

        for i in 0..res_height {
            for j in 0..res_width {
                let (mut ii, mut ij) = (i as isize, j as isize);

                ii -= (res_height / 2) as isize - (src_height / 2) as isize;
                ij -= (res_width  / 2) as isize - (src_width  / 2) as isize;

                if ii < 0 || ii >= src_height as isize {
                    res.push(what);
                    continue;
                }
                if ij < 0 || ij >= src_width as isize {
                    res.push(what);
                    continue;
                }

                res.push(self.buffer[(ii as usize)*src_width + (ij as usize)]);
            }
        }

        res
    }

    pub fn extend_image_repeat(&self, res_width: usize, res_height: usize) -> Vec<T> {
        let (src_width, src_height) = self.dimensions();
        let mut res: Vec<T> = Vec::with_capacity(res_width*res_height);

        for i in 0..res_height {
            for j in 0..res_width {
                let (mut ii, mut ij) = (i as isize, j as isize);

                ii -= (res_height / 2) as isize - (src_height / 2) as isize;
                ij -= (res_width  / 2) as isize - (src_width  / 2) as isize;

                ii = if ii < 0 { 0 } else { ii };
                ij = if ij < 0 { 0 } else { ij };

                ii = if ii >= src_height as isize { src_height as isize - 1 } else { ii };
                ij = if ij >= src_width  as isize { src_width  as isize - 1 } else { ij };

                res.push(self.buffer[(ii as usize)*src_width + (ij as usize)]);
            }
        }

        res
    }
}

//-----------------------------------------------------------------------------

pub fn split_rgb(
    img: &image::ImageBuffer<image::Rgb<f32>, Vec<f32>>
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (width, height) = img.dimensions();

    let mut r: Vec<f32> = Vec::with_capacity((width*height) as usize);
    let mut g: Vec<f32> = Vec::with_capacity((width*height) as usize);
    let mut b: Vec<f32> = Vec::with_capacity((width*height) as usize);

    for (_, _, rgb) in img.enumerate_pixels() {
        r.push(rgb[0]);
        g.push(rgb[1]);
        b.push(rgb[2]);
    }

    (r, g, b)
}

//-----------------------------------------------------------------------------

pub fn fft_2d<T>(
    width:     usize,
    height:    usize,
    buffer:    &[T],
    direction: rustfft::FftDirection
) -> Vec<Complex<f32>>
where
    T: Into<Complex<f32>> + Copy
{
    fn fft_shift<T: Copy>(data: &mut [T]) {
        let mid = data.len() / 2;
        if data.len()%2 == 0 {
            for (i1, i2) in (0..mid).zip(mid..data.len()) {
                let tmp  = data[i1];
                data[i1] = data[i2];
                data[i2] = tmp;
            }
        } else {
            for (i1, i2) in (0..mid).zip(mid..data.len()) {
                let tmp  = data[i1];
                data[i1] = data[i2];
                data[i2] = tmp;

                let tmp  = data[i1];
                data[i1] = data[i2+1];
                data[i2+1] = tmp;
            }
        }
    }
    fn transpose<T: Copy + Default>(width: usize, height: usize, matrix: &[T]) -> Vec<T> {
        let mut transposed = vec![T::default(); matrix.len()];

        for i in 0..height {
            for j in 0..width {
                transposed[j*height + i] = matrix[i*width + j];
            }
        }

        transposed
    }

    let mut planner = rustfft::FftPlanner::<f32>::new();
    let line_fft    = planner.plan_fft(width,  direction);
    let col_fft     = planner.plan_fft(height, direction);

    // Process lines
    let mut res: Vec<Complex<f32>> = buffer.iter().map(|&x| { x.into() }).collect();

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

//-----------------------------------------------------------------------------

pub fn float_to_pixel(f: f32) -> u16 {
    if f >= 1.0 { return 65535; }
    if f <= 0.0 { return 0;     }
    return (f*65535.0) as u16;
}

//-----------------------------------------------------------------------------
