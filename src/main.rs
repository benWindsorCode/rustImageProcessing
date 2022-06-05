use std::cmp::Ordering;
use image::{DynamicImage, GenericImageView};
use std::{cmp, env};
use image::imageops::contrast;
use ndarray::{array, Array2};

type ImageBuffer = image::ImageBuffer<image::Rgba<u8>, Vec<u8>>;

// Implementing functions from 'Computer Vision: Algorithms and Applications'
fn main() {
    let input1 = load_image("./images/benWindsorCodeIcon.jpg".to_string());
    let input2 = load_image("./images/houseTest.jpg".to_string());

    let edges = edge_detect(&input2);
    edges.save("./images/edgeDetected.png");

    let contrast_enhanced = contrast(&input2, 4.);
    contrast_enhanced.save("./images/contrastEnhanced.png");

    let brightness_enhanced = adjust_brightness(&input2, 50);
    brightness_enhanced.save("./images/brightnessEnhanced.png");
}


/// Edge detection by the following process:
///     1) increase image contrast
///     2) Sharpen image
///     3) Compute x and y gradients
///     4) Add x and y gradients
///
/// # Arguments
///
/// * `input`: ImageBuffer to edge detect
///
/// returns: ImageBuffer
fn edge_detect(input: &ImageBuffer) -> ImageBuffer {
    let cleaned = contrast(input, 2.);
    let sharpened = sharpen(&cleaned, 10.);

    let gradient_x = x_grad(&sharpened);
    let gradient_y = y_grad(&sharpened);

    image_add(&gradient_x, &gradient_y)
}

fn x_grad(input: &ImageBuffer) -> ImageBuffer {
    let matrix = array![
        [-1., 1.]
    ];

    apply_matrix(input, matrix)
}

fn y_grad(input: &ImageBuffer) -> ImageBuffer {
    let matrix = array![
        [1.],
        [-1.],
    ];

    apply_matrix(input, matrix)
}

/// Perform sharpening of an image by:
///     1) Perform a bilinear blur filter
///     2) Subtract the Image - Bilinear Output, to get the 'detail' of the image
///     3) Add specified multiple of detail back to original image
///
/// # Arguments
///
/// * `input`: ImageBuffer to sharpen
/// * `value`: Multiple of detail to add on
///
/// returns: ImageBuffer
fn sharpen(input: &ImageBuffer, value: f32) -> ImageBuffer {
    let filtered = bilinear_filter(input);

    let detail = image_sub(input, &filtered);
    let detail = contrast(&detail, value);

    image_add(input, &detail)
}

fn bilinear_filter(input: &ImageBuffer) -> ImageBuffer {
    let bilinear = array![
        [1./16., 2./16., 1./16.],
        [2./16., 4./16., 2./16.],
        [1./16., 2./16., 1./16.]
    ];

    apply_matrix(input, bilinear)
}

fn gaussian_blur(input: &ImageBuffer) -> ImageBuffer {
    let gaussian = array![
        [1./256., 4./256., 6./256., 4./256., 1./256.],
        [4./256., 16./256., 24./256., 16./256., 4./256.],
        [6./256., 24./256., 36./256., 24./256., 6./256.],
        [4./256., 16./256., 24./256., 16./256., 4./256.],
        [1./256., 4./256., 6./256., 4./256., 1./256.],
    ];

    apply_matrix(input, gaussian)
}

/// Apply a matrix to the input image, pixel by pixel where:
///     g(i, j) = sum f(i + k, j + l)h(k, l)
/// Where:
///     g is new pixel of image
///     f is current pixel of image
///     h is the matrix
///     k, l range over the dimensions of the matrix
///
/// # Arguments
///
/// * `input`: ImageBuffer to apply matrix to
/// * `matrix`: Matrix to apply
///
/// returns: ImageBuffer
fn apply_matrix(input: &ImageBuffer, matrix: Array2<f32>) -> ImageBuffer {
    let (input_x, input_y) = input.dimensions();
    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    let (matrix_x, matrix_y) = (matrix.shape().get(0).unwrap(), matrix.shape().get(1).unwrap());

    println!("Applying matrix of size: {}, {}", matrix_x, matrix_y);

    for (x, y, pixel) in output.enumerate_pixels_mut() {

        // Do all maths as integers then only truncate to [0, 255] right at the end
        let mut pixels_to_sum = Vec::new();

        for i in 0..*matrix_x {
            for j in 0..*matrix_y {
                let x_curr = (x as i32) + (i as i32);
                let y_curr = (y as i32) + (j as i32);

                let x_curr = cmp::min(input_x as i32 - 1, cmp::max(0, x_curr));
                let y_curr = cmp::min(input_y as i32 - 1, cmp::max(0, y_curr));

                let input_curr = input.get_pixel(x_curr as u32, y_curr as u32);
                let matrix_curr = matrix[[i, j]];

                let prod = vec![((input_curr[0] as f32) * matrix_curr) as i32, ((input_curr[1] as f32) * matrix_curr) as i32, ((input_curr[2] as f32) * matrix_curr) as i32];

                pixels_to_sum.push(prod);
            }
        }

        let mut total: Vec<i32> = vec![0, 0, 0];

        for pixel_to_sum in pixels_to_sum {
            total = vec![total[0] + pixel_to_sum[0], total[1] + pixel_to_sum[1], total[2] + pixel_to_sum[2]];
        }

        let r = cmp::min(255, cmp::max(0, total[0])) as u8;
        let g = cmp::min(255, cmp::max(0, total[1])) as u8;
        let b = cmp::min(255, cmp::max(0, total[2])) as u8;

        *pixel = image::Rgba([r, g, b, 255]);
    }

    output
}

fn median_filter(input: &ImageBuffer, window: i32) -> ImageBuffer {
    let (input_x, input_y) = input.dimensions();
    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let mut r_vals = Vec::new();
        let mut g_vals = Vec::new();
        let mut b_vals = Vec::new();

        for i in (-1*window)..(window+1) {
            for j in (-1*window)..(window+1) {
                let x_curr = (x as i32) + i;
                let y_curr = (y as i32) + j;

                let x_curr = cmp::min(input_x as i32 - 1, cmp::max(0, x_curr));
                let y_curr = cmp::min(input_y as i32 - 1, cmp::max(0, y_curr));

                let pixel_curr = input.get_pixel(x_curr as u32, y_curr as u32);
                r_vals.push(pixel_curr[0]);
                g_vals.push(pixel_curr[1]);
                b_vals.push(pixel_curr[2]);
            }
        }

        r_vals.sort();
        g_vals.sort();
        b_vals.sort();

        let r_median = median(&r_vals);
        let g_median = median(&g_vals);
        let b_median = median(&b_vals);

        let input_pixel = input.get_pixel(x, y);

        *pixel = image::Rgba([r_median, g_median, b_median, input_pixel[3]])
    }

    output
}

fn linear_blend(input_1: &ImageBuffer, input_2: &ImageBuffer, value: f32) -> ImageBuffer {
    let (input_x, input_y) = input_1.dimensions();
    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let scaled_1 = pixel_scale(*input_1.get_pixel(x,y), (1 as f32) - value);
        let scaled_2 = pixel_scale(*input_2.get_pixel(x,y), value);

        *pixel = pixel_add(scaled_1, scaled_2);
    }

    output
}

fn image_sub(input_1: &ImageBuffer, input_2: &ImageBuffer) -> ImageBuffer {
    let (input_x, input_y) = input_1.dimensions();
    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let image_1 = *input_1.get_pixel(x,y);
        let image_2 = *input_2.get_pixel(x,y);

        *pixel = pixel_sub(image_1, image_2);
    }

    output
}

fn image_add(input_1: &ImageBuffer, input_2: &ImageBuffer) -> ImageBuffer {
    let (input_x, input_y) = input_1.dimensions();
    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let image_1 = *input_1.get_pixel(x,y);
        let image_2 = *input_2.get_pixel(x,y);

        *pixel = pixel_add(image_1, image_2);
    }

    output
}

/// For each pixel, p, of an image, adjust brightness by output of:
///     p + value
///
/// # Arguments
///
/// * `input`: Image buffer
/// * `value`: Brightness addition value
///
/// returns: ImageBuffer
fn adjust_brightness(input: &ImageBuffer, value: i32) -> ImageBuffer {
    let (input_x, input_y) = input.dimensions();

    let mut output: ImageBuffer = image::ImageBuffer::new(input_x, input_y);

    for(x, y, pixel) in output.enumerate_pixels_mut() {
        *pixel = pixel_shift(*input.get_pixel(x, y), value);
    }

    output
}

/// For each pixel, p, of an image, adjust contrast by output of:
///     p * value
///
/// # Arguments
///
/// * `input`: Image buffer
/// * `value`: Contrast scale value
///
/// returns: ImageBuffer
fn adjust_contrast(input: &ImageBuffer, value: f32) -> ImageBuffer {
    let (input_x, input_y) = input.dimensions();

    let mut output: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::new(input_x, input_y);

    for(x, y, pixel) in output.enumerate_pixels_mut() {
        *pixel = pixel_scale(*input.get_pixel(x, y), value);
    }

    output
}

fn pixel_sub(pixel_1: image::Rgba<u8>, pixel_2: image::Rgba<u8>) -> image::Rgba<u8> {
    image::Rgba([
        safe_add(pixel_1[0], -1 * (pixel_2[0] as i32)),
        safe_add(pixel_1[1], -1 * (pixel_2[1] as i32)),
        safe_add(pixel_1[2], -1 * (pixel_2[2] as i32)),
        pixel_1[3]
    ])
}

fn pixel_add(pixel_1: image::Rgba<u8>, pixel_2: image::Rgba<u8>) -> image::Rgba<u8> {
    image::Rgba([
        safe_add(pixel_1[0], pixel_2[0] as i32),
        safe_add(pixel_1[1], pixel_2[1] as i32),
        safe_add(pixel_1[2], pixel_2[2] as i32),
        pixel_1[3]
    ])
}

/// Shift a pixels r,g,b values by a constant value (positive or negative)
///
/// # Arguments
///
/// * `pixel`: Single  pixel of an image
/// * `value`: Positive or negative value to shift the pixel by
///
/// returns: rgba pixel
fn pixel_shift(pixel: image::Rgba<u8>, value: i32) -> image::Rgba<u8> {
    image::Rgba([safe_add(pixel[0], value), safe_add(pixel[1], value), safe_add(pixel[2], value), pixel[3]])
}

/// Scales (multiplies) a pixels r,g,b values by a constant value
///
/// # Arguments
///
/// * `pixel`: Single  pixel of an image
/// * `value`: Scale factor of the pixel
///
/// returns: rgba pixel
fn pixel_scale(pixel: image::Rgba<u8>, value: f32) -> image::Rgba<u8> {
    image::Rgba([safe_mult(pixel[0], value), safe_mult(pixel[1], value), safe_mult(pixel[2], value), pixel[3]])
}

/// Given a u8 and an integer, perform addition in the i32 space but then clamp back to a u8
///
/// # Arguments
///
/// * `a`: u8 value
/// * `b`: i32 value
///
/// returns: u8
fn safe_add(a: u8, b: i32) -> u8 {
    let c = (a as i32) + b;
    let scaled = cmp::min(255, cmp::max(0, c));

    scaled as u8
}

/// Given a u8 and a float, multiply as floats, round to i32 then clamb pack to to a u8
///
/// # Arguments
///
/// * `a`: u8 value
/// * `b`: f32 value
///
/// returns: u8
fn safe_mult(a: u8, b: f32) -> u8 {
    let c = ((a as f32) * b) as i32;
    let scaled = cmp::min(255, cmp::max(0, c));

    scaled as u8
}

fn load_image(path: String) -> ImageBuffer {
    let input_raw = image::open(path).unwrap();

    let (input_x, input_y) = input_raw.dimensions();
    let mut input: ImageBuffer = image::ImageBuffer::new(input_x, input_y);
    for(x, y, pixel) in input.enumerate_pixels_mut() {
        *pixel = input_raw.get_pixel(x, y);
    }

    input
}

fn median(numbers: &Vec<u8>) -> u8 {
    let mid = numbers.len() / 2;

    numbers[mid]
}