use reqwest::blocking;

use edgen_core::settings;
use edgen_server::status;

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
        set_chat_completions_model(common::SMALL_LLM_NAME, common::SMALL_LLM_REPO);
        // set_audio_transcriptions_model
        test_ai_endpoint_with_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_with_download(Endpoint::AudioTranscriptions);
        // test_ai_endpoint_no_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_no_download(Endpoint::AudioTranscriptions);
        // update model dir
        // test_ai_endpoint_with_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_with_download(Endpoint::AudioTranscriptions);
        // test_ai_endpoint_no_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_no_download(Endpoint::AudioTranscriptions);
        // remove model dir
        // test_ai_endpoint_with_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_with_download(Endpoint::AudioTranscriptions);
        // test_ai_endpoint_no_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_no_download(Endpoint::AudioTranscriptions);
        // config reset
        // test_ai_endpoint_no_download(Endpoint::ChatCompletions);
        // test_ai_endpoint_no_download(Endpoint::AudioTranscriptions);
    })
}

fn connect_to_server_test() {
    common::test_message("connect to server");
    assert!(match blocking::get(common::make_url(&[
        common::BASE_URL,
        common::MISC_URL,
        common::VERSION_URL
    ])) {
        Err(e) => {
            eprintln!("cannot connect: {:?}", e);
            false
        }
        Ok(v) => {
            println!("have: '{}'", v.text().unwrap());
            true
        }
    });
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
    assert!(match blocking::get(common::make_url(&[
        common::BASE_URL,
        common::CHAT_URL,
        common::COMPLETIONS_URL,
        common::STATUS_URL,
    ])) {
        Err(e) => {
            eprintln!("cannot connect: {:?}", e);
            false
        }
        Ok(v) => {
            println!("have: '{}'", v.text().unwrap());
            true
        }
    });
}

fn audio_transcriptions_status_reachable() {
    common::test_message("audio transcriptions status is reachable");
    assert!(match blocking::get(common::make_url(&[
        common::BASE_URL,
        common::AUDIO_URL,
        common::TRANSCRIPTIONS_URL,
        common::STATUS_URL,
    ])) {
        Err(e) => {
            eprintln!("cannot connect: {:?}", e);
            false
        }
        Ok(v) => {
            println!("have: '{}'", v.text().unwrap());
            true
        }
    });
}

fn set_chat_completions_model(model_name: &str, model_repo: &str) {
    common::test_message(&format!("set chat completions model to {}", model_name,));

    let mut config = common::get_config().unwrap();
    config.chat_completions_model_name = model_name.to_string();
    config.chat_completions_model_repo = model_repo.to_string();
    common::write_config(&config).unwrap();

    println!("pausing for 4 secs");
    std::thread::sleep(std::time::Duration::from_secs(4));
    let stat: status::AIStatus = blocking::get(common::make_url(&[
        common::BASE_URL,
        common::CHAT_URL,
        common::COMPLETIONS_URL,
        common::STATUS_URL,
    ]))
    .unwrap()
    .json()
    .unwrap();

    assert_eq!(stat.active_model, model_name);
}

enum Endpoint {
    ChatCompletions,
    AudioTranscriptions,
}

fn test_ai_endpoint_with_download(endpoint: Endpoint) {
    let (ep, stp, body) = match endpoint {
        Endpoint::ChatCompletions => {
            common::test_message("chat completions endpoint with download");
            (
                common::make_url(&[common::BASE_URL, common::CHAT_URL, common::COMPLETIONS_URL]),
                common::make_url(&[
                    common::BASE_URL,
                    common::CHAT_URL,
                    common::COMPLETIONS_URL,
                    common::STATUS_URL,
                ]),
                common::CHAT_COMPLETIONS_BODY.to_string(),
            )
        }
        Endpoint::AudioTranscriptions => {
            common::test_message("audio transcriptions endpoint with progress");
            (
                common::make_url(&[
                    common::BASE_URL,
                    common::AUDIO_URL,
                    common::TRANSCRIPTIONS_URL,
                ]),
                common::make_url(&[
                    common::BASE_URL,
                    common::AUDIO_URL,
                    common::TRANSCRIPTIONS_URL,
                    common::STATUS_URL,
                ]),
                common::AUDIO_TRANSCRIPTIONS_BODY.to_string(),
            )
        }
    };
    let handle = common::spawn_request(ep, body);
    assert!(common::observe_progress(&stp));
    assert!(handle.join().unwrap());
}
