use num_complex::ComplexFloat;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: program_name input_image output_image");
        std::process::exit(1);
    }

    let input_image_path  = &args[1];
    let output_image_path = &args[2];

    let kern = smoothify::kernel::BlurKernel::new();
    let kern_img = kern.to_img();

    let img = image::io::Reader::open(input_image_path)
        .expect("Failed to open input image")
        .decode()
        .expect("Failed to decode input image");
    let converted = img.to_rgb32f();
    let (width, height) = converted.dimensions();

    let (r, g, b) = smoothify::convolve::split_rgb(&converted);
    let r_img = smoothify::convolve::ImageWrapper::new(width as usize, height as usize, &r);
    let g_img = smoothify::convolve::ImageWrapper::new(width as usize, height as usize, &g);
    let b_img = smoothify::convolve::ImageWrapper::new(width as usize, height as usize, &b);

    let [conv_r, conv_g, conv_b] =
        smoothify::convolve::ImageWrapper::convolve([&r_img, &g_img, &b_img], &kern_img);

    let buf = image::ImageBuffer::from_fn(width, height, move |j, i| {
        let idx = i*width + j;
        let val_r = smoothify::convolve::float_to_pixel(conv_r[idx as usize].abs());
        let val_g = smoothify::convolve::float_to_pixel(conv_g[idx as usize].abs());
        let val_b = smoothify::convolve::float_to_pixel(conv_b[idx as usize].abs());
        image::Rgb::from([val_r, val_g, val_b])
    });

    buf.save(output_image_path)
        .expect("Failed to save output image");
}
