use std::borrow::Cow;
use std::time::Duration;

use either::Either;
use futures::StreamExt;
use rand::Rng;
use reqwest_eventsource::EventSource;
use reqwest_eventsource::{retry, Event};
use tokio::task::JoinSet;
use tokio::time::sleep;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use edgen_server::openai_shim::{ChatCompletionChunk, ChatMessage, CreateChatCompletionRequest};

const START_PROMPTS: [&str; 6] = [
    "Hello!",
    "Please give me a number between 1 and 50.",
    "Please tell me a short story.",
    "Please tell me a long story.",
    "What is the capital of Portugal?",
    "What is the current weather like in France?",
];

const CONTINUE_PROMPTS: [&str; 4] = [
    "Please continue.",
    "Tell me more.",
    "Can you give me more details?",
    "I don't understand.",
];

/// Send an arbitrary number of requests to the streaming chat endpoint.
#[derive(argh::FromArgs, PartialEq, Debug, Clone)]
pub struct Chat {
    /// the total amount of requests sent.
    #[argh(positional, default = "10")]
    pub requests: usize,

    /// the base chance that a conversation will continue.
    #[argh(option, short = 'b', default = "0.6")]
    pub continue_chance: f32,

    /// how much the chance to continue a conversation will decrease with each successive message.
    #[argh(option, short = 'd', default = "0.05")]
    pub chance_decay: f32,

    /// the minimum amount of time to wait before a request is sent.
    #[argh(option, short = 'i', default = "3.0")]
    pub min_idle: f32,

    /// the maximum amount of time to wait before a request is sent.
    #[argh(option, short = 'a', default = "10.0")]
    pub max_idle: f32,

    /// the maximum size of a received message.
    #[argh(option, short = 'l', default = "1000")]
    pub message_limit: usize,

    /// the base URL of the endpoint the requests will be sent to.
    #[argh(
        option,
        short = 'u',
        default = "String::from(\"http://127.0.0.1:33322\")"
    )]
    pub url: String,
}

#[tokio::main]
async fn main() {
    let format = tracing_subscriber::fmt::layer().compact();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
        tracing_subscriber::EnvFilter::default()
            .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into()),
    );
    tracing_subscriber::registry()
        .with(format)
        .with(filter)
        .init();

    let chat_args: Chat = argh::from_env();

    assert!(
        chat_args.min_idle < chat_args.max_idle,
        "Minimum idle time must be higher than the maximum"
    );

    let mut rng = rand::thread_rng();

    let mut request_chains = vec![];
    let mut chain: usize = 0;
    for _ in 0..chat_args.requests {
        let chance = f32::max(
            chat_args.continue_chance - chat_args.chance_decay * chain as f32,
            0.0,
        );

        chain += 1;
        if chance < rng.gen() {
            request_chains.push(chain);
            chain = 0;
        }
    }

    if chain > 0 {
        request_chains.push(chain);
    }

    let mut join_set = JoinSet::new();
    for (id, count) in request_chains.drain(..).enumerate() {
        join_set.spawn(chain_requests(chat_args.clone(), count, id));
    }

    while let Some(_) = join_set.join_next().await {}
}

async fn chain_requests(chat_args: Chat, count: usize, index: usize) {
    let client = reqwest::Client::new();
    let base_builder = client.post(chat_args.url + "/v1/chat/completions");
    let mut body = CreateChatCompletionRequest {
        messages: Default::default(),
        model: Cow::from("default"),
        frequency_penalty: None,
        logit_bias: None,
        max_tokens: Some(chat_args.message_limit as u32),
        n: None,
        presence_penalty: None,
        seed: None,
        stop: None,
        stream: Some(true),
        response_format: None,
        temperature: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        user: None,
        one_shot: None,
        context_hint: None,
    };

    body.messages.push(ChatMessage::System {
        content: Some(Cow::from("You are Edgen, a helpful assistant.")),
        name: None,
    });

    let prompt_idx = rand::thread_rng().gen_range(0..START_PROMPTS.len());
    body.messages.push(ChatMessage::User {
        content: Either::Left(Cow::from(START_PROMPTS[prompt_idx])),
        name: None,
    });

    for request in 0..count {
        let wait = rand::thread_rng().gen_range(chat_args.min_idle..chat_args.max_idle);
        sleep(Duration::from_secs_f32(wait)).await;
        info!(
            "Chain {} sending request {} of {}.",
            index,
            request + 1,
            count
        );

        let builder = base_builder.try_clone().unwrap().json(&body);
        let mut event_source = EventSource::new(builder).unwrap();
        event_source.set_retry_policy(Box::new(retry::Never));
        let mut token_count = 0;
        let mut text = "".to_string();
        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    if token_count >= chat_args.message_limit {
                        event_source.close();
                        break;
                    }
                    token_count += 1;
                    let response: ChatCompletionChunk =
                        serde_json::from_str(message.data.as_str()).unwrap();
                    text += response.choices[0].delta.content.as_ref().unwrap();
                }
                Err(reqwest_eventsource::Error::StreamEnded) => {}
                Err(err) => {
                    println!("Error: {}", err);
                    event_source.close();
                }
            }
        }

        body.messages.push(ChatMessage::Assistant {
            content: Some(Cow::from(text)),
            name: None,
            tool_calls: None,
        });

        let continue_idx = rand::thread_rng().gen_range(0..CONTINUE_PROMPTS.len());
        body.messages.push(ChatMessage::User {
            content: Either::Left(Cow::from(CONTINUE_PROMPTS[continue_idx])),
            name: None,
        });
    }
}
