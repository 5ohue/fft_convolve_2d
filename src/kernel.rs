pub const SIZE: usize = 200;
pub const SIGMA: usize = 5;

//-----------------------------------------------------------------------------

#[derive(Debug)]
pub struct BlurKernel {
    data: Box<[f32]>,
}

impl BlurKernel {
    pub fn new() -> Self {
        let mut k: Vec<f32> = Vec::with_capacity((SIZE+1)*(SIZE+1));

        let mut acc = 0.0;
        for i in 0..SIZE+1 {
            for j in 0..SIZE+1 {
                let (x, y) = (i as f32, j as f32);
                let x = x - (SIZE as f32) * 0.5;
                let y = y - (SIZE as f32) * 0.5;

                let l2 = x*x + y*y;

                let mut val = 0.0;
                for k in 1..=SIGMA {
                    let s = (1 << k) as f32;
                    val += (-l2*0.5 / (s*s)).exp() / (s*s) / (1 << (SIGMA-k)) as f32;
                }

                acc += val;

                k.push(val);
            }
        }
        k[SIZE*(SIZE+1)/2 + SIZE/2] += 1.0 / (1 << (SIGMA-1)) as f32;
        acc += 1.0 / (1 << SIGMA) as f32;

        k.iter_mut().for_each(|val| { *val = *val/acc; });

        BlurKernel { data: k.into() }
    }

    pub fn to_img<'a>(&'a self) -> crate::convolve::ImageFloat<'a, f32> {
        crate::convolve::ImageFloat::new(SIZE+1, SIZE+1, &self.data)
    }

    pub fn get_data(&self) -> &[f32] {
        &self.data
    }
}

//-----------------------------------------------------------------------------
