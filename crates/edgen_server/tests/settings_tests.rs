use reqwest::blocking;

mod common;

#[test]
fn fake_test() {
    println!("hello fake!");
}

#[test]
fn test_battery() {
    common::with_save_edgen(|| {
        pass_always();
        connect_to_server_test();
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
