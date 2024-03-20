use std::fs::remove_dir_all;
use std::path;

use reqwest::blocking;

use edgen_server::types::Endpoint;

#[allow(dead_code)]
mod common;

#[test]
// The fake test serves for trying simple things out
fn fake_test() {
    println!("hello fake!");
}

#[test]
// Be aware that this test is long-running and downloads several GiB from huggingface.
// Since it tests the edgen environment (config and model directories),
// a backup of the user environment is created before the tests start
// and restored when the tests have passed (or failed).
// Should backup or restore fail, parts of your environment may have moved away.
// You find them in 'crates/edgen_server/env_backup' and may want to restore them manually.
// Run with
// cargo test --test settings_tests -- --show-output --nocapture
// (without --nocapture the output is shown only after finishing)
//
// Currently five scenarios are exercised (please update if you add new test scenarios):
// - SCENARIO 1:
//   + Start edgen server without files, they shall exist afterwards
//   + Endpoints shall be reachable
// - SCENARIO 2:
//   + Edit configuration: change should be reflected in server.
//   + AI Endpoints shall be reachable
//   + Model files shall be downloaded
//   + Once downloaded, no download is preformed again
// - SCENARIO 3:
//   + Change model directories: make sure now directories are created
//   + Model files shall be downloaded
//   + Once downloaded, no download is preformed again
// - SCENARIO 4:
//   + Remove model directories: make sure they are created again
//   + Model files shall be downloaded
//   + Once downloaded, no download is preformed again
// - SCENARIO 5:
//   + Reset configuration
//   + Model files shall not be downloaded
// - SCENARIO 6:
//   + Use custom huggingface model (not present)
//   + Model files shall be downloaded
// - SCENARIO 7:
//   + Use custom huggingface model (present)
//   + Model files shall not be downloaded
// - SCENARIO 8:
//   + Use custom user-managed model (present)
//   + Model files shall not be downloaded
fn test_battery() {
    common::with_save_edgen(|| {
        // make sure everything is right
        common::pass_always();

        // ================================
        common::test_message("SCENARIO 1");
        // ================================
        common::config_exists();
        common::data_exists();

        // endpoints reachable
        common::connect_to_server_test();

        chat_completions_status_reachable();
        audio_transcriptions_status_reachable();

        // ================================
        common::test_message("SCENARIO 2");
        // ================================
        // set small models, so we don't need to download too much
        common::set_model(
            Endpoint::ChatCompletions,
            common::SMALL_LLM_NAME,
            common::SMALL_LLM_REPO,
        );
        common::set_model(
            Endpoint::AudioTranscriptions,
            common::SMALL_WHISPER_NAME,
            common::SMALL_WHISPER_REPO,
        );

        // test ai endpoint and download
        test_ai_endpoint_with_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_with_download(Endpoint::AudioTranscriptions, "default");

        // we have downloaded, we should not download again
        test_ai_endpoint_no_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_no_download(Endpoint::AudioTranscriptions, "default");

        // ================================
        common::test_message("SCENARIO 3");
        // ================================
        let my_models_dir = format!(
            "{}{}{}",
            common::BACKUP_DIR,
            path::MAIN_SEPARATOR,
            common::MY_MODEL_FILES,
        );

        let new_chat_completions_dir = my_models_dir.clone()
            + &format!(
                "{}{}{}{}",
                path::MAIN_SEPARATOR,
                "chat",
                path::MAIN_SEPARATOR,
                "completions",
            );

        let new_audio_transcriptions_dir = my_models_dir.clone()
            + &format!(
                "{}{}{}{}",
                path::MAIN_SEPARATOR,
                "audio",
                path::MAIN_SEPARATOR,
                "transcriptions",
            );

        common::set_model_dir(Endpoint::ChatCompletions, &new_chat_completions_dir);

        common::set_model_dir(
            Endpoint::AudioTranscriptions,
            &new_audio_transcriptions_dir,
        );

        test_ai_endpoint_with_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_with_download(Endpoint::AudioTranscriptions, "default");

        assert!(path::Path::new(&my_models_dir).exists());

        test_ai_endpoint_no_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_no_download(Endpoint::AudioTranscriptions, "default");

        // ================================
        common::test_message("SCENARIO 4");
        // ================================
        remove_dir_all(&my_models_dir).unwrap();
        assert!(!path::Path::new(&my_models_dir).exists());

        test_ai_endpoint_with_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_with_download(Endpoint::AudioTranscriptions, "default");

        assert!(path::Path::new(&my_models_dir).exists());

        test_ai_endpoint_no_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_no_download(Endpoint::AudioTranscriptions, "default");

        // ================================
        common::test_message("SCENARIO 5");
        // ================================
        test_config_reset();

        common::set_model(
            Endpoint::ChatCompletions,
            common::SMALL_LLM_NAME,
            common::SMALL_LLM_REPO,
        );
        common::set_model(
            Endpoint::AudioTranscriptions,
            common::SMALL_WHISPER_NAME,
            common::SMALL_WHISPER_REPO,
        );

        // make sure we read from the old directories again
        remove_dir_all(&my_models_dir).unwrap();
        assert!(!path::Path::new(&my_models_dir).exists());

        test_ai_endpoint_no_download(Endpoint::ChatCompletions, "default");
        test_ai_endpoint_no_download(Endpoint::AudioTranscriptions, "default");

        // ================================
        common::test_message("SCENARIO 6");
        // ================================
        let chat_model = "TheBloke/phi-2-GGUF/phi-2.Q2_K.gguf";
        let audio_model = "distil-whisper/distil-medium.en/ggml-medium-32-2.en.bin";

        test_ai_endpoint_with_download(Endpoint::ChatCompletions, chat_model);
        test_ai_endpoint_with_download(Endpoint::AudioTranscriptions, audio_model);

        // ================================
        common::test_message("SCENARIO 7");
        // ================================
        test_ai_endpoint_no_download(Endpoint::ChatCompletions, chat_model);
        test_ai_endpoint_no_download(Endpoint::AudioTranscriptions, audio_model);

        // ================================
        common::test_message("SCENARIO 8");
        // ================================
        let source = "models--TheBloke--phi-2-GGUF/blobs";
        common::copy_model(source, ".phi-2.Q2_K.gguf", "chat/completions");
        test_ai_endpoint_no_download(Endpoint::ChatCompletions, ".phi-2.Q2_K.gguf");

        let source = "models--distil-whisper--distil-medium.en/blobs";
        common::copy_model(source, ".ggml-medium-32-2.en.bin", "audio/transcriptions");
        test_ai_endpoint_no_download(
            Endpoint::AudioTranscriptions,
            ".ggml-medium-32-2.en.bin",
        );
    })
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
            assert!(v.status().is_success());
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
            assert!(v.status().is_success());
            println!("have: '{}'", v.text().unwrap());
            true
        }
    });
}

fn test_config_reset() {
    common::test_message("test resetting config");
    common::reset_config();

    println!("pausing for 4 secs to make sure the config file has been updated");
    std::thread::sleep(std::time::Duration::from_secs(4));
}

fn test_ai_endpoint_with_download(endpoint: Endpoint, model: &str) {
    test_ai_endpoint(endpoint, model, true);
}

fn test_ai_endpoint_no_download(endpoint: Endpoint, model: &str) {
    test_ai_endpoint(endpoint, model, false);
}

fn test_ai_endpoint(endpoint: Endpoint, model: &str, download: bool) {
    let (statep, body) = match endpoint {
        Endpoint::ChatCompletions => {
            common::test_message(&format!(
                "chat completions endpoint with download: {}",
                download
            ));
            (
                common::make_url(&[
                    common::BASE_URL,
                    common::CHAT_URL,
                    common::COMPLETIONS_URL,
                    common::STATUS_URL,
                ]),
                common::chat_completions_custom_body(model),
            )
        }
        Endpoint::AudioTranscriptions => {
            common::test_message(&format!(
                "audio transcriptions endpoint with download: {}",
                download
            ));
            (
                common::make_url(&[
                    common::BASE_URL,
                    common::AUDIO_URL,
                    common::TRANSCRIPTIONS_URL,
                    common::STATUS_URL,
                ]),
                "".to_string(),
            )
        }
        Endpoint::Embeddings => todo!(),
    };
    let handle = common::spawn_request(endpoint, &body, model);
    if download {
        common::assert_download(&statep);
    } else {
        common::assert_no_download(&statep);
    }
    assert!(handle.join().unwrap());
}
