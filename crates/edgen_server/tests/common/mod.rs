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

pub const AUDIO_TRANSCRIPTIONS_BODY: &str = "";

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

pub fn spawn_request(endpoint: String, body: String) -> thread::JoinHandle<bool> {
    thread::spawn(move || {
        println!("requesting {}", endpoint);
        match blocking::Client::new()
            .post(&endpoint)
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

pub fn observe_progress(endpoint: &str) -> bool {

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
            assert!(Instant::now().duration_since(tp) < Duration::from_secs(60));
        }
        thread::sleep(Duration::from_millis(30));
        stat = blocking::get(endpoint).unwrap().json().unwrap();
    }
    assert_eq!(stat.download_progress, 100);
    true
}

const BACKUP_DIR: &str = "env_backup";

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

    println!("config bkp: {:?}", cnfg_bkp);

    copy_dir(&cnfg, &cnfg_bkp)?;

    let data = settings::PROJECT_DIRS.data_dir();
    let data_bkp = backup_dir.join("data");

    println!("data   bkp: {:?}", data_bkp);

    copy_dir(&data, &data_bkp)?;

    fs::remove_dir_all(&cnfg)?;
    fs::remove_dir_all(&data)?;

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

    println!("{:?} -> {:?}", cnfg_bkp, cnfg);
    copy_dir(&cnfg_bkp, &cnfg)?;

    println!("{:?} -> {:?}", data_bkp, data);
    copy_dir(&data_bkp, &data)?;

    println!("removing {:?}", backup_dir);
    fs::remove_dir_all(&backup_dir)?;

    Ok(())
}
