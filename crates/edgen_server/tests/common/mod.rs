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

pub const SMALL_LLM_NAME: &str = "phi-2.Q2_K.gguf";
pub const SMALL_LLM_REPO: &str = "TheBloke/phi-2-GGUF";

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

pub fn with_save_env<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
    // T: Send + 'static,
    // E: std::error::Error,
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

pub fn with_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe, // -> TestResult + Send + 'static,
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

pub fn with_save_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe, // -> TestResult + Send + 'static,
{
    with_save_env(|| {
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
                true
            }
        }
    })
}

pub fn spawn_audio_transcriptions_request() -> thread::JoinHandle<bool> {
    thread::spawn(move || {
        let ep = make_url(&[BASE_URL, AUDIO_URL, TRANSCRIPTIONS_URL]);

        println!("requesting {}", ep);

        let sound = include_bytes!("../../resources/frost.wav");
        let part = blocking::multipart::Part::bytes(sound.as_slice()).file_name("frost.wav");

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
                true
            }
        }
    })
}

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
