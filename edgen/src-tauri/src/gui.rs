/* Copyright 2023- The Binedge, Lda team. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use edgen_core::settings;
use edgen_server;
use opener;
use tauri::{
    CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu, SystemTrayMenuItem,
};
use tracing::info;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

pub fn run() {
    // here `"quit".to_string()` defines the menu item id, and the second parameter is the menu item label.
    let menu_item_show = CustomMenuItem::new("show".to_string(), "Show");
    let menu_item_quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let menu_item_edgen_chat = CustomMenuItem::new("edgenchat".to_string(), "EdgenChat");
    let menu_item_config = CustomMenuItem::new("config".to_string(), "Config");
    let menu_item_reset_config = CustomMenuItem::new("reset_config".to_string(), "Reset Config");
    // let show = CustomMenuItem::new("show".to_string(), "Show");
    let tray_menu = SystemTrayMenu::new()
        // .add_item(show)
        .add_item(menu_item_edgen_chat)
        .add_item(menu_item_config)
        .add_item(menu_item_reset_config)
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(menu_item_show)
        .add_item(menu_item_quit);

    tauri::Builder::default()
        .system_tray(SystemTray::new().with_menu(tray_menu))
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick {
                position: _,
                size: _,
                ..
            } => {
                info!("system tray received a left click");
                let window = app.get_window("main").unwrap();
                window.set_focus().unwrap();
            }
            SystemTrayEvent::RightClick {
                position: _,
                size: _,
                ..
            } => {
                info!("system tray received a right click");
            }
            SystemTrayEvent::DoubleClick {
                position: _,
                size: _,
                ..
            } => {
                info!("system tray received a double click");
            }

            SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
                "show" => {
                    let window = app.get_window("main").unwrap();
                    window.show().unwrap();
                }
                "quit" => {
                    std::process::exit(0);
                }
                "edgenchat" => {
                    if let Err(err) = opener::open("https://chat.edgen.co") {
                        eprintln!("Error opening website: {}", err);
                    }
                }
                "config" => {
                    if let Err(err) = opener::open(settings::get_config_file_path()) {
                        eprintln!("Error opening config file: {}", err);
                    }
                }
                "reset_config" => {
                    if let Err(err) = edgen_server::config_reset() {
                        eprintln!("Error resetting config: {}", err);
                    }
                }
                _ => {}
            },
            _ => {}
        })
        // keep the frontend running in the background
        .on_window_event(|event| match event.event() {
            tauri::WindowEvent::CloseRequested { api, .. } => {
                event.window().hide().unwrap();
                api.prevent_close();
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
