use std::fmt;
use std::fmt::Display;
use std::fs;
use std::io;
use std::panic;
use std::path::Path;
use std::str;
use std::thread;
use std::time::{Duration, Instant};

use copy_dir::copy_dir;
use reqwest::blocking;
use serde_yaml;

use edgen_core::settings;
use edgen_core::settings::SettingsParams;
use edgen_server::cli;
use edgen_server::start;
use edgen_server::status;

pub const SMALL_LLM_NAME: &str = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
pub const SMALL_LLM_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";

pub const SMALL_WHISPER_NAME: &str = "ggml-distil-small.en.bin";
pub const SMALL_WHISPER_REPO: &str = "distil-whisper/distil-small.en";

pub const BASE_URL: &str = "http://localhost:33322/v1";
pub const CHAT_URL: &str = "/chat";
pub const COMPLETIONS_URL: &str = "/completions";
pub const AUDIO_URL: &str = "/audio";
pub const TRANSCRIPTIONS_URL: &str = "/transcriptions";
pub const STATUS_URL: &str = "/status";
pub const MISC_URL: &str = "/misc";
pub const VERSION_URL: &str = "/version";
pub const MODELS_URL: &str = "/models";

pub const CHAT_COMPLETIONS_BODY: &str = r#"
    {
        "model": "this is not a model",
        "messages": [
          {
            "role": "system",
            "content": "You are a helpful assistant."
          },
          {
            "role": "user",
            "content": "what is the result of 1 + 2?"
          }
        ],
        "stream": true
    }
"#;

pub const BACKUP_DIR: &str = "env_backup";
pub const CONFIG_BACKUP_DIR: &str = "config_backup";
pub const MY_MODEL_FILES: &str = "my_models";

#[derive(Debug, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    AudioTranscriptions,
}

impl Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Endpoint::ChatCompletions => "chat completions",
            Endpoint::AudioTranscriptions => "audio transcriptions",
        };
        write!(f, "{}", s)
    }
}

// Backup environment (config and model directories) before running 'f';
// restore environment, even if 'f' panicks.
pub fn with_save_env<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    println!("with save env!");

    backup_env().unwrap();

    println!("==============");
    println!("STARTING TESTS");
    println!("==============");

    let r = panic::catch_unwind(f);

    println!("===========");
    println!("TESTS READY");
    println!("===========");

    let _ = match restore_env() {
        Ok(_) => (),
        Err(e) => {
            panic!("Panic! Cannot restore your environment: {:?}", e);
        }
    };

    match r {
        Err(e) => panic::resume_unwind(e),
        Ok(_) => (),
    }
}

// Backup config only before running 'f';
// restore config, even if 'f' panicks.
pub fn with_save_config<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    println!("with save config!");

    backup_config().unwrap();

    println!("==============");
    println!("STARTING TESTS");
    println!("==============");

    let r = panic::catch_unwind(f);

    println!("===========");
    println!("TESTS READY");
    println!("===========");

    let _ = match restore_config() {
        Ok(_) => (),
        Err(e) => {
            panic!("Panic! Cannot restore your config: {:?}", e);
        }
    };

    match r {
        Err(e) => panic::resume_unwind(e),
        Ok(_) => (),
    }
}

// Start edgen before running 'f'
pub fn with_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    let _ = thread::spawn(|| {
        let mut args = cli::Serve::default();
        args.nogui = true;
        let cmd = cli::Command::Serve(args);
        start(&cli::TopLevel {
            subcommand: Some(cmd),
        })
        .unwrap();
    });

    // give the server time to start
    thread::sleep(std::time::Duration::from_secs(1));

    f();
}

// Backup environment (config and model directories)
// and start edgen before running 'f';
// restore environment, even if 'f' or edgen panick.
pub fn with_save_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    with_save_env(|| {
        with_edgen(f);
    });
}

// Backup config directories)
// and start edgen before running 'f';
// restore config, even if 'f' or edgen panick.
pub fn with_save_config_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    with_save_config(|| {
        with_edgen(f);
    });
}

pub fn test_message(msg: &str) {
    println!("=== Test {}", msg);
}

pub fn make_url(v: &[&str]) -> String {
    let mut s = "".to_string();
    for e in v {
        s += e;
    }
    s
}

#[derive(Debug)]
pub enum ConfigError {
    IOError(io::Error),
    YamlError(serde_yaml::Error),
    Utf8Error(str::Utf8Error),
}

impl From<io::Error> for ConfigError {
    fn from(e: io::Error) -> Self {
        ConfigError::IOError(e)
    }
}

impl From<str::Utf8Error> for ConfigError {
    fn from(e: str::Utf8Error) -> Self {
        ConfigError::Utf8Error(e)
    }
}

impl From<serde_yaml::Error> for ConfigError {
    fn from(e: serde_yaml::Error) -> Self {
        ConfigError::YamlError(e)
    }
}

pub fn get_config() -> Result<SettingsParams, ConfigError> {
    let path = settings::get_config_file_path();
    let buf = fs::read(path)?;
    let yaml = str::from_utf8(&buf)?;
    Ok(serde_yaml::from_str(yaml)?)
}

pub fn write_config(config: &SettingsParams) -> Result<(), ConfigError> {
    let path = settings::get_config_file_path();
    let yaml = serde_yaml::to_string(config)?;
    let buf = yaml.as_bytes().to_vec();
    fs::write(path, buf)?;
    Ok(())
}

pub fn reset_config() {
    edgen_server::config_reset().unwrap();
}

pub fn pass_always() {
    test_message("pass always");
    assert!(true);
}

pub fn config_exists() {
    test_message("config exists");
    assert!(settings::PROJECT_DIRS.config_dir().exists());
    assert!(settings::CONFIG_FILE.exists());
}

pub fn data_exists() {
    test_message("data exists");
    let data = settings::PROJECT_DIRS.data_dir();
    println!("exists: {:?}", data);
    assert!(data.exists());

    let models = data.join("models");
    println!("exists: {:?}", models);
    assert!(models.exists());

    let chat = models.join("chat");
    println!("exists: {:?}", chat);
    assert!(models.exists());

    let completions = chat.join("completions");
    println!("exists: {:?}", completions);
    assert!(completions.exists());

    let audio = models.join("audio");
    println!("exists: {:?}", audio);
    assert!(audio.exists());

    let transcriptions = audio.join("transcriptions");
    println!("exists: {:?}", transcriptions);
    assert!(transcriptions.exists());
}

// edit the config file: set another model dir for the indicated endpoint.
pub fn set_model_dir(ep: Endpoint, model_dir: &str) {
    test_message(&format!("set {} model directory to {}", ep, model_dir,));

    let mut config = get_config().unwrap();

    match &ep {
        Endpoint::ChatCompletions => {
            config.chat_completions_models_dir = model_dir.to_string();
        }
        Endpoint::AudioTranscriptions => {
            config.audio_transcriptions_models_dir = model_dir.to_string();
        }
    }
    write_config(&config).unwrap();

    println!("pausing for 4 secs to make sure the config file has been updated");
    std::thread::sleep(std::time::Duration::from_secs(4));
}

// edit the config file: set another model name and repo for the indicated endpoint.
pub fn set_model(ep: Endpoint, model_name: &str, model_repo: &str) {
    test_message(&format!("set {} model to {}", ep, model_name,));

    let mut config = get_config().unwrap();

    match &ep {
        Endpoint::ChatCompletions => {
            config.chat_completions_model_name = model_name.to_string();
            config.chat_completions_model_repo = model_repo.to_string();
        }
        Endpoint::AudioTranscriptions => {
            config.audio_transcriptions_model_name = model_name.to_string();
            config.audio_transcriptions_model_repo = model_repo.to_string();
        }
    }
    write_config(&config).unwrap();

    println!("pausing for 4 secs to make sure the config file has been updated");
    std::thread::sleep(std::time::Duration::from_secs(4));
    let url = match ep {
        Endpoint::ChatCompletions => make_url(&[BASE_URL, CHAT_URL, COMPLETIONS_URL, STATUS_URL]),
        Endpoint::AudioTranscriptions => {
            make_url(&[BASE_URL, AUDIO_URL, TRANSCRIPTIONS_URL, STATUS_URL])
        }
    };
    let stat: status::AIStatus = blocking::get(url).unwrap().json().unwrap();
    assert_eq!(stat.active_model, model_name);
}

// exercise the edgen version endpoint to make sure the server is reachable.
pub fn connect_to_server_test() {
    test_message("connect to server");
    assert!(
        match blocking::get(make_url(&[BASE_URL, MISC_URL, VERSION_URL])) {
            Err(e) => {
                eprintln!("cannot connect: {:?}", e);
                false
            }
            Ok(v) => {
                println!("have: '{}'", v.text().unwrap());
                true
            }
        }
    );
}

// spawn a thread to send a request to the indicated endpoint.
// This allows the caller to perform another task in the caller thread.
pub fn spawn_request(ep: Endpoint, body: String) -> thread::JoinHandle<bool> {
    match ep {
        Endpoint::ChatCompletions => spawn_chat_completions_request(body),
        Endpoint::AudioTranscriptions => spawn_audio_transcriptions_request(),
    }
}

pub fn spawn_chat_completions_request(body: String) -> thread::JoinHandle<bool> {
    thread::spawn(move || {
        let ep = make_url(&[BASE_URL, CHAT_URL, COMPLETIONS_URL]);
        println!("requesting {}", ep);
        match blocking::Client::new()
            .post(&ep)
            .header("Content-Type", "application/json")
            .body(body)
            .timeout(Duration::from_secs(180))
            .send()
        {
            Err(e) => {
                eprintln!("cannot connect: {:?}", e);
                false
            }
            Ok(v) => {
                println!("Got {:?}", v);
                assert!(v.status().is_success());
                true
            }
        }
    })
}

pub fn spawn_audio_transcriptions_request() -> thread::JoinHandle<bool> {
    let frost = Path::new("resources").join("frost.wav");
    thread::spawn(move || {
        let ep = make_url(&[BASE_URL, AUDIO_URL, TRANSCRIPTIONS_URL]);

        println!("requesting {}", ep);

        let sound = fs::read(frost).unwrap();
        let part = blocking::multipart::Part::bytes(sound).file_name("frost.wav");

        let form = blocking::multipart::Form::new()
            .text("model", "default")
            .part("file", part);

        match blocking::Client::new()
            .post(&ep)
            .multipart(form)
            .timeout(Duration::from_secs(180))
            .send()
        {
            Err(e) => {
                eprintln!("cannot connect: {:?}", e);
                false
            }
            Ok(v) => {
                println!("Got {:?}", v);
                assert!(v.status().is_success());
                true
            }
        }
    })
}

// Assert that a download is ongoing and download progress is reported.
pub fn assert_download(endpoint: &str) {
    println!("requesting status of {}", endpoint);

    let mut stat: status::AIStatus = blocking::get(endpoint).unwrap().json().unwrap();

    let mut tp = Instant::now();
    while !stat.download_ongoing {
        thread::sleep(Duration::from_millis(100));
        stat = blocking::get(endpoint).unwrap().json().unwrap();

        if Instant::now().duration_since(tp) > Duration::from_secs(60) {
            break;
        }
    }

    assert!(stat.download_ongoing);

    let mut last_p = 0;
    while stat.download_ongoing {
        let p = stat.download_progress;
        if p > last_p {
            last_p = p;
            tp = Instant::now();
        } else {
            assert!(Instant::now().duration_since(tp) < Duration::from_secs(90));
        }
        thread::sleep(Duration::from_millis(30));
        stat = blocking::get(endpoint).unwrap().json().unwrap();
    }
    assert_eq!(stat.download_progress, 100);
}

// Assert that *no* download is ongoing.
pub fn assert_no_download(endpoint: &str) {
    println!("requesting status of {}", endpoint);

    let mut stat: status::AIStatus = blocking::get(endpoint).unwrap().json().unwrap();

    let tp = Instant::now();
    while !stat.download_ongoing {
        thread::sleep(Duration::from_millis(100));
        stat = blocking::get(endpoint).unwrap().json().unwrap();

        if Instant::now().duration_since(tp) > Duration::from_secs(60) {
            break;
        }
    }

    assert!(!stat.download_ongoing);
}

#[derive(Debug)]
enum BackupError {
    Unfinished,
    IOError(io::Error),
    Errors(Vec<io::Error>),
}

impl From<io::Error> for BackupError {
    fn from(e: io::Error) -> Self {
        BackupError::IOError(e)
    }
}

impl From<Vec<io::Error>> for BackupError {
    fn from(es: Vec<io::Error>) -> Self {
        BackupError::Errors(es)
    }
}

fn backup_env() -> Result<(), BackupError> {
    println!("backing up");

    let backup_dir = Path::new(BACKUP_DIR);
    if backup_dir.exists() {
        let msg = format!(
            "directory {} exists!
             This means an earlier test run did not finish correctly. \
             Restore your environment manually.",
            BACKUP_DIR,
        );
        eprintln!("{}", msg);
        return Err(BackupError::Unfinished);
    }

    println!("config dir: {:?}", settings::PROJECT_DIRS.config_dir());
    println!("data   dir: {:?}", settings::PROJECT_DIRS.data_dir());

    fs::create_dir(&backup_dir)?;

    let cnfg = settings::PROJECT_DIRS.config_dir();
    let cnfg_bkp = backup_dir.join("config");

    if cnfg.exists() {
        println!("config bkp: {:?}", cnfg_bkp);
        copy_dir(&cnfg, &cnfg_bkp)?;
        fs::remove_dir_all(&cnfg)?;
    } else {
        println!("config {:?} does not exist", cnfg);
    }

    let data = settings::PROJECT_DIRS.data_dir();
    let data_bkp = backup_dir.join("data");

    if data.exists() {
        println!("data   bkp: {:?}", data_bkp);
        copy_dir(&data, &data_bkp)?;
        fs::remove_dir_all(&data)?;
    } else {
        println!("data {:?} does not exist", data);
    }

    Ok(())
}

fn restore_env() -> Result<(), io::Error> {
    println!("restoring");

    let backup_dir = Path::new(BACKUP_DIR);

    let cnfg = settings::PROJECT_DIRS.config_dir();
    let cnfg_bkp = backup_dir.join("config");

    let data = settings::PROJECT_DIRS.data_dir();
    let data_bkp = backup_dir.join("data");

    if cnfg.exists() {
        fs::remove_dir_all(&cnfg)?;
    }

    if data.exists() {
        fs::remove_dir_all(&data)?;
    }

    if cnfg_bkp.exists() {
        println!("{:?} -> {:?}", cnfg_bkp, cnfg);
        copy_dir(&cnfg_bkp, &cnfg)?;
    } else {
        println!("config bkp {:?} does not exist", cnfg_bkp);
    }

    if data_bkp.exists() {
        println!("{:?} -> {:?}", data_bkp, data);
        copy_dir(&data_bkp, &data)?;
    } else {
        println!("data bkp {:?} does not exist", data_bkp);
    }

    println!("removing {:?}", backup_dir);
    fs::remove_dir_all(&backup_dir)?;

    Ok(())
}

fn backup_config() -> Result<(), BackupError> {
    println!("backing up");

    let backup_dir = Path::new(CONFIG_BACKUP_DIR);
    if backup_dir.exists() {
        let msg = format!(
            "directory {} exists!
             This means an earlier test run did not finish correctly. \
             Restore your environment manually.",
            CONFIG_BACKUP_DIR,
        );
        eprintln!("{}", msg);
        return Err(BackupError::Unfinished);
    }

    println!("config dir: {:?}", settings::PROJECT_DIRS.config_dir());

    fs::create_dir(&backup_dir)?;

    let cnfg = settings::PROJECT_DIRS.config_dir();
    let cnfg_bkp = backup_dir.join("config");

    if cnfg.exists() {
        println!("config bkp: {:?}", cnfg_bkp);
        copy_dir(&cnfg, &cnfg_bkp)?;
        fs::remove_dir_all(&cnfg)?;
    } else {
        println!("config {:?} does not exist", cnfg);
    }

    Ok(())
}

fn restore_config() -> Result<(), io::Error> {
    println!("restoring");

    let backup_dir = Path::new(CONFIG_BACKUP_DIR);

    let cnfg = settings::PROJECT_DIRS.config_dir();
    let cnfg_bkp = backup_dir.join("config");

    if cnfg.exists() {
        fs::remove_dir_all(&cnfg)?;
    }

    if cnfg_bkp.exists() {
        println!("{:?} -> {:?}", cnfg_bkp, cnfg);
        copy_dir(&cnfg_bkp, &cnfg)?;
    } else {
        println!("config bkp {:?} does not exist", cnfg_bkp);
    }

    println!("removing {:?}", backup_dir);
    fs::remove_dir_all(&backup_dir)?;

    Ok(())
}
