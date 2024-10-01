//-----------------------------------------------------------------------------
use anyhow::Result;
use fft_convolve_2d::ConvolveFFT;
//-----------------------------------------------------------------------------

#[derive(Debug)]
enum KernelType {
    Gaussian,
    Exponential,
    Polynomial,
    Smoothify,
}

#[derive(Debug)]
struct ProgramData {
    input_filename: String,
    output_filename: String,
    kernel_type: KernelType,
    kernel_size: u32,
    params: [f32; 2],
}

//-----------------------------------------------------------------------------

fn parse_args() -> ProgramData {
    // Args
    let input_arg = clap::Arg::new("input")
        .long("input")
        .short('i')
        .help("Input image filename")
        .value_name("FILE")
        .required(true);

    let output_arg = clap::Arg::new("output")
        .long("output")
        .short('o')
        .help("Output image filename")
        .value_name("FILE")
        .default_value("Output.png");

    let kernel_names = ["Gaussian", "Exponential", "Polynomial", "Smoothify"];
    let kernel_type_arg = clap::Arg::new("kernel")
        .long("kernel")
        .short('k')
        .help(format!("Kernel type. Available types: {:?}", kernel_names))
        .value_name("KERNEL")
        .default_value("Smoothify");

    let kernel_size_arg = clap::Arg::new("kernel_size")
        .long("kernel-size")
        .short('s')
        .help("Pixel size for the kernel")
        .value_name("SIZE")
        .default_value("201");

    let params_arg = clap::Arg::new("params")
        .long("params")
        .short('p')
        .help("Convolve parameters")
        .value_name("PARAMS")
        .default_value("");

    // Command
    let cmd = clap::Command::new("Convolver")
        .arg(input_arg)
        .arg(output_arg)
        .arg(kernel_type_arg)
        .arg(kernel_size_arg)
        .arg(params_arg);

    let matches = cmd.get_matches();

    // Create program data
    let input_filename = matches.get_one::<String>("input").unwrap();
    let output_filename = matches.get_one::<String>("output").unwrap();

    let kernel_type_str = matches.get_one::<String>("kernel").unwrap();
    let kernel_type;
    if kernel_type_str == kernel_names[0] {
        kernel_type = KernelType::Gaussian;
    } else if kernel_type_str == kernel_names[1] {
        kernel_type = KernelType::Exponential;
    } else if kernel_type_str == kernel_names[2] {
        kernel_type = KernelType::Polynomial;
    } else {
        kernel_type = KernelType::Smoothify;
    }

    let params_str = matches.get_one::<String>("params").unwrap();
    let mut params = [1.0; 2];
    params
        .iter_mut()
        .zip(params_str.split(','))
        .for_each(|(param, str)| {
            if let Ok(value) = str.parse::<f32>() {
                *param = value;
            }
        });

    let kernel_size = matches
        .get_one::<String>("kernel_size")
        .unwrap()
        .parse::<u32>()
        .expect("Kernel size should be an integer!")
        .clamp(1, 3000);

    return ProgramData {
        input_filename: input_filename.clone(),
        output_filename: output_filename.clone(),
        kernel_type,
        kernel_size,
        params,
    };
}

//-----------------------------------------------------------------------------

fn main() -> Result<()> {
    let data = parse_args();

    let img = image::ImageReader::open(data.input_filename)
        .expect("Failed to open input image")
        .decode()
        .expect("Failed to decode input image");

    let kernel = match data.kernel_type {
        KernelType::Gaussian => {
            fft_convolve_2d::kernel::generate_gauss(data.kernel_size, data.params[0])
        }
        KernelType::Exponential => {
            fft_convolve_2d::kernel::generate_exp(data.kernel_size, data.params[0], data.params[1])
        }
        KernelType::Polynomial => {
            fft_convolve_2d::kernel::generate_poly(data.kernel_size, data.params[0], data.params[1])
        }
        KernelType::Smoothify => {
            fft_convolve_2d::kernel::generate_smoothify(data.kernel_size, data.params[0])
        }
    };

    img.convolve(&kernel).save(data.output_filename)?;

    return Ok(());
}

//-----------------------------------------------------------------------------
