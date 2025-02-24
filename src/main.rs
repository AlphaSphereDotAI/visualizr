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
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, MultipartForm)]
struct UploadForm {
    #[multipart(rename = "file")]
    image: Vec<TempFile>,
    // #[multipart(field = "name")]
}

#[post("/generate")]
async fn generate(MultipartForm(form): MultipartForm<UploadForm>) -> Result<impl Responder, Error> {
    for f in form.image {
        let path = format!("./tmp/{}", f.file_name.unwrap());
        log::info!("saving to {path}");
        f.file.persist(path).unwrap();
    }

    Ok(HttpResponse::Ok()
        .content_type(ContentType::json())
        .body("File uploaded".to_string()))
}

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok()
        .content_type(ContentType::html())
        .body("Welcome to the Chatacter's Video Generator API")
}

// #[post("/generate")]
// async fn generate(params: web::Json<GenerateParams>) -> impl Responder {
//     if params.text.trim().is_empty() {
//         return Err(actix_web::error::ErrorBadRequest("Text cannot be empty"));
//     };

//     if !(0..=10).contains(&params.character_id) {
//         return Err(actix_web::error::ErrorBadRequest(
//             "Character ID must be between 0 and 10",
//         ));
//     };

//     let config: KokoroTtsConfig = KokoroTtsConfig {
//         model: "./kokoro-en-v0_19/model.onnx".to_string(),
//         voices: "./kokoro-en-v0_19/voices.bin".into(),
//         tokens: "./kokoro-en-v0_19/tokens.txt".into(),
//         data_dir: "./kokoro-en-v0_19/espeak-ng-data".into(),
//         length_scale: 1.0,
//         ..Default::default()
//     };
//     let mut tts = KokoroTts::new(config);
//     // 0->af, 1->af_bella, 2->af_nicole, 3->af_sarah, 4->af_sky, 5->am_adam
//     // 6->am_michael, 7->bf_emma, 8->bf_isabella, 9->bm_george, 10->bm_lewis
//     let audio: TtsAudio = tts
//         .create(&params.text, params.speaker_id, 1.0)
//         .map_err(actix_web::error::ErrorInternalServerError)?;

//     let filename = format!("assets/{}.wav", Uuid::new_v4());
//     if let Err(e) = write_audio_file(&filename, &audio.samples, audio.sample_rate) {
//         log::info!("Error writing audio file: {:?}", e);
//         return Err(actix_web::error::ErrorInternalServerError(format!(
//             "Error writing audio file: {:?}",
//             e
//         )));
//     }
//     Ok(NamedFile::open_async(&filename).await?)
// }

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
