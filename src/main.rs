use axum::{
    extract::Multipart,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use imageproc::contrast::{otsu_level, threshold};
use serde::Serialize;
use std::io::Write;
use tempfile::NamedTempFile;
use tower_http::cors::CorsLayer;
use uuid::Uuid;
use std::env;
use base64::{Engine as _, engine::general_purpose};

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    message: String,
}

#[derive(Serialize)]
struct ProcessingResponse {
    success: bool,
    message: String,
    threshold_value: f64,
    output_filename: String,
    processed_image_base64: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    success: bool,
    error: String,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        message: "Image processing service is running".to_string(),
    })
}

async fn process_image(mut multipart: Multipart) -> Result<Json<ProcessingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut temp_input = NamedTempFile::new()
        .map_err(|e| {
            eprintln!("Failed to create temp input file: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: "Failed to create temporary file".to_string(),
                }),
            )
        })?;

    let mut image_found = false;
    let mut original_filename = String::new();

    // Process multipart form data
    while let Some(mut field) = multipart.next_field().await
        .map_err(|e| {
            eprintln!("Multipart error: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    success: false,
                    error: "Invalid multipart data".to_string(),
                }),
            )
        })? {
        
        let field_name = field.name().unwrap_or_default();
        println!("Processing field: {}", field_name);
        
        if field_name == "image" {
            image_found = true;
            
            // Get original filename if available
            if let Some(filename) = field.file_name() {
                original_filename = filename.to_string();
                println!("Original filename: {}", original_filename);
            }
            
            let mut total_bytes = 0;
            
            while let Some(chunk) = field.chunk().await
                .map_err(|e| {
                    eprintln!("Chunk error: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            success: false,
                            error: "Error reading file chunks".to_string(),
                        }),
                    )
                })? {
                total_bytes += chunk.len();
                temp_input.write_all(&chunk)
                    .map_err(|e| {
                        eprintln!("Write error: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                success: false,
                                error: "Error writing to temporary file".to_string(),
                            }),
                        )
                    })?;
            }
            println!("Received {} bytes for image", total_bytes);
        }
    }

    if !image_found {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                success: false,
                error: "No image field found in request".to_string(),
            }),
        ));
    }

    // Flush the input file to ensure all data is written
    temp_input.flush().map_err(|e| {
        eprintln!("Flush error: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                success: false,
                error: "Error flushing file".to_string(),
            }),
        )
    })?;

    // Create output file with proper extension
    let output_extension = if original_filename.to_lowercase().ends_with(".png") {
        ".png"
    } else {
        ".jpg"
    };
    
    // Create a new temporary file with proper extension for input
    let mut input_path = env::temp_dir();
    input_path.push(format!("input_{}{}", Uuid::new_v4(), output_extension));
    
    // Copy the uploaded file to the new location with proper extension
    std::fs::copy(temp_input.path(), &input_path)
        .map_err(|e| {
            eprintln!("Failed to copy file: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: "Failed to prepare image file".to_string(),
                }),
            )
        })?;
    
    let mut output_path = env::temp_dir();
    output_path.push(format!("processed_{}{}", Uuid::new_v4(), output_extension));
    
    let input_path_str = input_path.to_str().unwrap();
    let output_path_str = output_path.to_str().unwrap();
    
    println!("Processing image from {} to {}", input_path.display(), output_path_str);

    // Process image with cv crate
    let threshold_value = process_image_cv(input_path_str, output_path_str)
        .map_err(|e| {
            eprintln!("CV error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: format!("Image processing failed: {}", e),
                }),
            )
        })?;

    // Read the processed image and convert to base64
    let processed_image_data = std::fs::read(output_path_str)
        .map_err(|e| {
            eprintln!("Failed to read processed image: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    success: false,
                    error: "Failed to read processed image".to_string(),
                }),
            )
        })?;

    let processed_image_base64 = general_purpose::STANDARD.encode(&processed_image_data);
    let mime_type = if output_extension == ".png" { "image/png" } else { "image/jpeg" };
    let data_url = format!("data:{};base64,{}", mime_type, processed_image_base64);

    let output_filename = format!("processed_{}.jpg", Uuid::new_v4());
    println!("Successfully processed image with threshold: {}", threshold_value);

    Ok(Json(ProcessingResponse {
        success: true,
        message: "Image processed successfully".to_string(),
        threshold_value,
        output_filename,
        processed_image_base64: data_url,
    }))
}

fn process_image_cv(input_path: &str, output_path: &str) -> Result<f64, Box<dyn std::error::Error>> {
    println!("Reading image from: {}", input_path);
    
    // Read image using image crate and convert to grayscale
    let img = image::open(input_path)?;
    let gray_img = img.to_luma8();
    
    println!("Image loaded successfully, size: {}x{}", gray_img.width(), gray_img.height());

    // Calculate the Otsu threshold value using imageproc
    let threshold_val = otsu_level(&gray_img);
    println!("Threshold applied: {}", threshold_val);

    // Apply the threshold to binarize the image
    let binary_img = threshold(&gray_img, threshold_val);

    // Save the binarized image
    binary_img.save(output_path)?;
    println!("Processed image saved to: {}", output_path);

    Ok(threshold_val as f64)
}



#[tokio::main]
async fn main() {
    println!("🚀 Starting image processing backend with Axum...");
    println!("📍 Server: http://localhost:8080");
    println!("📋 Available endpoints:");
    println!("   GET  /health  - Health check");
    println!("   POST /process - Process image (multipart form with 'image' field)");
    println!("");

    // Configure CORS for web client access
    let cors = CorsLayer::permissive();

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/process", post(process_image))
        .layer(cors);

    // Start server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("Failed to bind to address");
    
    println!("✅ Server listening on {}", listener.local_addr().unwrap());
    println!("🔄 Ready to process images!");
    println!("");

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
