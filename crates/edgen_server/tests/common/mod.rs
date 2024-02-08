use std::fs;
use std::io;
use std::panic;
use std::path::Path;
use std::thread;

use copy_dir::copy_dir;

use edgen_core::settings;
use edgen_server::cli;
use edgen_server::start;

pub fn test_message(msg: &str) {
    println!("=== Test {}", msg);
}

pub fn with_save_env<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
    // T: Send + 'static,
    // E: std::error::Error,
{
    println!("with save env!");

    backup_env().unwrap();

    println!("STARTING TESTS");
    println!("==============");

    let r = panic::catch_unwind(f);

    println!("===========");
    println!("TESTS READY");

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

    // give the server time to stop
    thread::sleep(std::time::Duration::from_secs(3));
}

pub fn with_save_edgen<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe, // -> TestResult + Send + 'static,
{
    with_save_env(|| {
        with_edgen(f);
    });
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
