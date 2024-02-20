use std::path;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use futures::executor::block_on;
use reqwest::blocking;

use edgen_core::settings;
use edgen_server::model_man::{ModelDeletionStatus, ModelDesc, ModelList};

#[allow(dead_code)]
mod common;

#[test]
fn test_modelmanager() {
    common::with_save_config_edgen(|| {
        common::pass_always();

        common::config_exists();

        common::connect_to_server_test();

        let my_models_dir = format!(
            "{}{}{}",
            common::CONFIG_BACKUP_DIR,
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

        common::set_model_dir(common::Endpoint::ChatCompletions, &new_chat_completions_dir);

        common::set_model_dir(
            common::Endpoint::AudioTranscriptions,
            &new_audio_transcriptions_dir,
        );

        make_dirs();

        test_list_models();
        test_delete_model();
    })
}

// actually create the model dirs before using them
fn make_dirs() {
    let dir = block_on(async { settings::chat_completions_dir().await });
    std::fs::create_dir_all(&dir).expect("cannot create chat completions model dir");

    assert!(PathBuf::from(&dir).exists());

    let dir = block_on(async { settings::audio_transcriptions_dir().await });
    std::fs::create_dir_all(&dir).expect("cannot create audio transcriptions model dir");

    assert!(PathBuf::from(&dir).exists());
}

fn test_list_models() {
    common::test_message("list models");

    let bloke = "TheBloke";
    let the = "The";
    let r1 = "TinyLlama-1.1B-Chat-v1.0-GGUF";
    let r2 = "Bloke--TinyLlama-1.1B-Chat-v1.0-GGUF";
    let r3 = "TinyLlama--1.1B--Chat--v1.0--GGUF";
    let f1 = format!("models--{}--{}", bloke, r1);
    let f2 = format!("models--{}--{}", the, r2);
    let f3 = format!("models--{}--{}", bloke, r3);
    let f4 = "invisible".to_string();
    let f5 = "models--TheBlokeInvisible".to_string();
    let f6 = "tmp".to_string();

    let recent = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        - 2; // careful with leap seconds

    let dir1 = block_on(async { settings::audio_transcriptions_dir().await });
    let dir1 = PathBuf::from(&dir1);

    let dir2 = block_on(async { settings::chat_completions_dir().await });
    let dir2 = PathBuf::from(&dir2);

    std::fs::create_dir(dir1.join(&f1)).expect(&format!("cannot create dir {:?}", f1));
    std::fs::create_dir(dir2.join(&f2)).expect(&format!("cannot create dir {:?}", f2));
    std::fs::create_dir(dir1.join(&f3)).expect(&format!("cannot create dir {:?}", f3));
    std::fs::create_dir(dir1.join(&f4)).expect(&format!("cannot create dir {:?}", f4));
    std::fs::create_dir(dir2.join(&f5)).expect(&format!("cannot create dir {:?}", f5));
    std::fs::create_dir(dir1.join(&f6)).expect(&format!("cannot create dir {:?}", f6));

    // --- get model descriptor
    let res = blocking::get(common::make_url(&[common::BASE_URL, common::MODELS_URL]))
        .expect("models get endpoint failed");
    assert!(res.status().is_success(), "models failed");

    let v = res
        .json::<ModelList>()
        .expect("cannot converto to model list")
        .data;

    assert_eq!(v.len(), 3);

    println!("recent is {}", recent);
    for m in v {
        assert_eq!(m.object, "model");
        if m.owned_by != the {
            assert_eq!(m.owned_by, bloke);
        }
        if m.id != format!("{}/{}", bloke, r1) && m.id != format!("{}/{}", bloke, r3) {
            assert_eq!(m.id, format!("{}/{}", the, r2));
        }
        println!("{:?}", m);

        let d = m.created.checked_sub(recent).unwrap();
        assert!(d <= 3);
    }
}

fn test_delete_model() {
    common::test_message("delete model");

    let owner = "TheFaker";
    let repo = "my-faked-model-v1-GGUF";
    let model = format!("models--{}--{}", owner, repo);
    let id = format!("{}/{}", owner, repo);
    let id_url = format!("{}%2f{}", owner, repo);

    let recent = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        - 2; // careful with leap seconds

    let dir = block_on(async { settings::chat_completions_dir().await });

    let dir = Path::new(&dir).join(&model);
    std::fs::create_dir(&dir).expect(&format!("cannot create model {:?}", dir));

    // --- get model descriptor
    let res = blocking::get(common::make_url(&[
        common::BASE_URL,
        common::MODELS_URL,
        "/",
        &id_url,
    ]))
    .expect("models get endpoint failed");

    assert!(res.status().is_success(), "models failed");
    let m: ModelDesc = res.json().expect("cannot convert to model desc");

    println!("model descriptor: {:?}", m);
    assert_eq!(m.object, "model");
    assert_eq!(m.owned_by, owner);
    assert_eq!(m.id, id);
    let d = m.created.checked_sub(recent).unwrap();
    assert!(d <= 3);

    // --- delete model
    println!("delete model");
    let res = blocking::Client::new()
        .delete(common::make_url(&[
            common::BASE_URL,
            common::MODELS_URL,
            "/",
            &id_url,
        ]))
        .send()
        .expect("models delete endpoint failed");

    assert!(res.status().is_success());
    let m: ModelDeletionStatus = res.json().expect("cannot convert to model deletion status");
    assert!(m.deleted);
    assert_eq!(m.id, id);
    assert!(!dir.exists());
}
