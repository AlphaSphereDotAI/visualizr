use actix_files::NamedFile;
use actix_multipart::{
    form::{
        tempfile::{TempFile, TempFileConfig},
        MultipartForm,
    },
    Multipart,
};
use actix_web::{
    get, http::header::ContentType, middleware, post, web, App, Error, HttpResponse, HttpServer,
    Responder,
};
use std::process::Command;
use uuid::Uuid;

#[derive(Debug, MultipartForm)]
struct UploadForm {
    #[multipart(rename = "file")]
    image: TempFile,
    audio: TempFile,
}

#[post("/generate")]
async fn generate(MultipartForm(form): MultipartForm<UploadForm>) -> Result<impl Responder, Error> {
    let image_file_path = format!(
        "./tmp/{}",
        form.image.file_name.unwrap() + Uuid::new_v4().to_string().as_str()
    );
    let audio_file_path = format!(
        "./tmp/{}",
        form.audio.file_name.unwrap() + Uuid::new_v4().to_string().as_str()
    );
    log::info!("saving to {image_file_path}");
    log::info!("saving to {audio_file_path}");
    form.image.file.persist(image_file_path.clone()).unwrap();
    form.audio.file.persist(audio_file_path.clone()).unwrap();
    log::info!("Files uploaded");
    log::info!("Generating video");
    // docker run --gpus "all" --rm -v $(pwd):/host_dir wawa9000/sadtalker \
    // --driven_audio /host_dir/deyu.wav \
    // --source_image /host_dir/image.jpg \
    // --expression_scale 1.0 \
    // --still \
    // --result_dir /host_dir
    let output = Command::new("docker")
        .arg("run")
        .arg("--gpus \"all\"")
        .arg("--rm")
        .arg("-v")
        .arg(format!(
            "{}:/host_dir",
            std::env::current_dir().unwrap().to_str().unwrap()
        ))
        .arg("wawa9000/sadtalker")
        .arg("--driven_audio")
        .arg(audio_file_path.to_string())
        .arg("--source_image")
        .arg(image_file_path.to_string())
        .arg("--expression_scale")
        .arg("1.0")
        .arg("--still")
        .arg("--result_dir")
        .arg(format!(
            "{}",
            std::env::current_dir().unwrap().to_str().unwrap()
        ))
        .output()
        .expect("Failed to execute command");
    assert!(output.status.success());
    log::info!("status: {}", output.status);
    Ok(NamedFile::open_async(format!(
        "{}",
        std::env::current_dir().unwrap().to_str().unwrap()
    ))
    .await?)
}

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok()
        .content_type(ContentType::html())
        .body("Welcome to the Chatacter's Video Generator API")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    log::info!("creating temporary upload directory");
    std::fs::create_dir_all("./tmp")?;
    log::info!("starting HTTP server at http://localhost:8002");
    HttpServer::new(|| {
        App::new()
            .service(hello)
            .service(generate)
            .wrap(middleware::Logger::default())
            .app_data(TempFileConfig::default().directory("./tmp"))
    })
    .bind(("0.0.0.0", 8002))?
    .run()
    .await
}
