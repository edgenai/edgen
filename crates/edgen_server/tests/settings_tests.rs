use reqwest::blocking;

use edgen_core::settings;

mod common;

#[test]
fn fake_test() {
    println!("hello fake!");
}

#[test]
fn test_battery() {
    common::with_save_edgen(|| {
        pass_always();
        config_exists();
        data_exists();
        connect_to_server_test();
        chat_completions_status_reachable();
        audio_transcriptions_status_reachable();
    });
}

fn connect_to_server_test() {
    common::test_message("connect to server");
    assert!(
        match blocking::get("http://localhost:33322/v1/misc/version") {
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

fn pass_always() {
    common::test_message("pass always");
    assert!(true);
}

fn config_exists() {
    common::test_message("config exists");
    assert!(settings::PROJECT_DIRS.config_dir().exists());
    assert!(settings::CONFIG_FILE.exists());
}

fn data_exists() {
    common::test_message("data exists");
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

fn chat_completions_status_reachable() {
    common::test_message("chat completions status is reachable");
    assert!(
        match blocking::get("http://localhost:33322/v1/chat/completions/status") {
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

fn audio_transcriptions_status_reachable() {
    common::test_message("audio transcriptions status is reachable");
    assert!(
        match blocking::get("http://localhost:33322/v1/audio/transcriptions/status") {
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
