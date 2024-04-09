use std::borrow::Cow;
use std::time::Duration;

use either::Either;
use futures::StreamExt;
use rand::Rng;
use reqwest_eventsource::{retry, Event};
use reqwest_eventsource::{Error, EventSource};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::{sleep, Instant};
use tracing::{debug, error, info};
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

const LARGE_CONTEXT: &str = r#"Gordon Freeman, a recently employed theoretical physicist, is involved in an experiment analyzing an unknown crystalline artifact; however, when the anti-mass spectrometer beam contacts the crystal, it creates a resonance cascade that opens a dimensional rift between Black Mesa and another world called Xen, causing monsters to swarm Black Mesa and kill many of the facility's personnel. Attempts by the Black Mesa personnel to close the rift are unsuccessful, leading to a Marine Recon unit being sent in to silence the facility, including any survivors from the science team. Freeman fights through the facility to meet with several other scientists, who decide to travel to the alien dimension to stop the aliens. On Xen, Freeman eliminates the alien "leader" and is confronted by the G-Man, who offers Freeman employment before putting him into stasis.[2] Back in Black Mesa, a second alien race begins an invasion, but is stopped when a Marine corporal, Adrian Shephard, collapses its portal in the facility. The G-Man then destroys Black Mesa with a nuclear warhead, and detains Shephard in stasis. Barney Calhoun, a security officer, also escaped from the facility with Dr. Rosenberg and two other scientists. Nearly twenty years later,[2] Half-Life 2 opens as the G-Man brings Freeman out of stasis and inserts him into a dystopian Earth ruled by the Combine, a faction consisting of human and alien members, that used the dimensional rift caused at Black Mesa to conquer Earth in the interim. In the Eastern European settlement City 17, Freeman meets surviving members of the Black Mesa incident, including Isaac Kleiner, Barney Calhoun, Eli Vance and his daughter Alyx Vance, and aids in the human resistance against Combine rule. The Xen aliens, the Vortigaunts, who have been enslaved by the Combine, also assist the resistance. When his presence is made known to former Black Mesa administrator and Combine spokesman Wallace Breen, Freeman becomes a prime target for the Combine forces. Eventually, Freeman sparks a full revolution amongst the human citizens after destroying Nova Prospekt, a major Combine base and troop-production facility. Eli Vance and his daughter are subsequently captured by the Combine, and Freeman helps the resistance forces attack the Combine's Citadel to rescue them, fighting alongside Barney. Freeman fights his way through the Citadel, making his way to Breen's office. He is temporarily captured, but freed by Dr. Mossman, along with Eli and Alyx. Breen attempts to flee in a teleporter, but is presumed dead after Freeman destroys the dark energy reactor at the Citadel's top. The story continues with Half-Life 2: Episode One, as the G-Man then arrives to extract Freeman before he is engulfed in the explosion, but is interrupted when Vortigaunts liberate Freeman from stasis and place both him and Alyx Vance at the bottom of the Citadel. Alyx then contacts her father, Eli Vance, and Isaac Kleiner, who have escaped the city into the surrounding countryside. Kleiner informs them that the reactor's core has gone critical due to the destruction of the dark energy reaction, and is at risk of exploding at any moment, an explosion which could completely destroy City 17. To delay the explosion they must enter the Citadel's now-decaying core and attempt to stabilize its primary reactor while the citizens evacuate the city from a train station. While inside, they discover that the Combine are attempting to speed up the destruction of the reactor, and use the destruction of the Citadel to call for reinforcements from the Combine's native dimension. After downloading critical data, they move through the war-torn city to the train station to take the last train out of the city. The Combine then destroy the reactor and thus both the Citadel and the city; the resulting explosion causes the train to derail. Half-Life 2: Episode Two begins as Freeman awakens in one of the wrecked train cars with Alyx outside. In the distance a forming super-portal is visible where the Citadel used to stand. They begin a journey through the White Forest to a resistance-controlled missile base in the nearby mountains. Along the way, Freeman and Alyx are ambushed and Alyx is severely injured. However, a group of Vortigaunts are able to heal her. During the healing ritual, Freeman receives word from G-Man, indicating that the Vortigaunts were keeping him at bay. G-Man demands that Freeman take Alyx to White Forest as safely as possible, saying that he cannot help as per restrictions he has agreed to. They are able to reach the resistance base and deliver the data, which contains the codes to destroy the portal as well as information on the Borealis, an enigmatic research vessel operated by Black Mesa's rival, Aperture Science; however, the ship disappeared while testing portal technology. The base then launches a satellite that is able to shut down the super-portal, cutting off the Combine from outside assistance. However, as Alyx and Freeman prepare to travel to the Arctic and investigate the Borealis, they are attacked by Combine Advisors, who kill Eli Vance, before being driven off by Alyx's pet robot, D0g."#;

const LARGE_PROMPTS: [&str; 5] = [
    "Please resume the Half-Life story.",
    "Please give a summary of the Half-Life story.",
    "Do you think Gordon's actions were correct?",
    "What was Alyx's pet robot called?",
    "Please write a story similar to Half-Life.",
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

    /// the chance that a request will start with large context.
    #[argh(option, short = 'e', default = "0.0")]
    pub large_chance: f32,

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
        chat_args.min_idle <= chat_args.max_idle,
        "Minimum idle time cannot be higher than the maximum"
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
    let (tx, mut rx) = mpsc::unbounded_channel();
    for (id, count) in request_chains.drain(..).enumerate() {
        join_set.spawn(chain_requests(chat_args.clone(), count, id, tx.clone()));
    }
    drop(tx);

    let mut first_tokens = vec![];
    let mut all_tokens = vec![];
    let mut all_tokens_nf = vec![];
    let mut token_counts = vec![];

    while let Some(stats) = rx.recv().await {
        first_tokens.push(stats.first_token);
        all_tokens.extend(&stats.all_tokens);
        all_tokens_nf.extend(&stats.all_tokens[1..]);
        token_counts.push(stats.all_tokens.len());
    }

    println!("First token times:");
    print_stats(first_tokens);
    println!("All token times:");
    print_stats(all_tokens);
    println!("All token times (without first token):");
    print_stats(all_tokens_nf);
    println!("Token counts:");
    print_token_stats(token_counts);

    while let Some(_) = join_set.join_next().await {}
}

async fn chain_requests(
    chat_args: Chat,
    count: usize,
    index: usize,
    stats_tx: mpsc::UnboundedSender<RequestStatistics>,
) {
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

    if chat_args.large_chance < rand::thread_rng().gen() {
        let prompt_idx = rand::thread_rng().gen_range(0..START_PROMPTS.len());
        body.messages.push(ChatMessage::User {
            content: Either::Left(Cow::from(START_PROMPTS[prompt_idx])),
            name: None,
        });
    } else {
        body.messages.push(ChatMessage::System {
            content: Some(Cow::from(LARGE_CONTEXT)),
            name: None,
        });

        let prompt_idx = rand::thread_rng().gen_range(0..LARGE_PROMPTS.len());
        body.messages.push(ChatMessage::User {
            content: Either::Left(Cow::from(LARGE_PROMPTS[prompt_idx])),
            name: None,
        });
    }

    for request in 0..count {
        let wait = if chat_args.min_idle != chat_args.max_idle {
            rand::thread_rng().gen_range(chat_args.min_idle..chat_args.max_idle)
        } else {
            chat_args.min_idle
        };
        sleep(Duration::from_secs_f32(wait)).await;
        info!(
            "Chain {} sending request {} of {}.",
            index,
            request + 1,
            count
        );

        let builder = base_builder.try_clone().unwrap().json(&body);

        let mut stats = RequestStatistics {
            first_token: -1.0,
            all_tokens: vec![],
        };
        let mut t = Instant::now();

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

                    let nt = Instant::now();
                    let d = (nt - t).as_secs_f32();
                    t = nt;

                    if stats.first_token == -1.0 {
                        stats.first_token = d;
                    }
                    stats.all_tokens.push(d);

                    token_count += 1;
                    debug!("Chain {index} has received token {token_count}");
                    let response: ChatCompletionChunk =
                        serde_json::from_str(message.data.as_str()).unwrap();
                    text += response.choices[0].delta.content.as_ref().unwrap();
                }
                Err(reqwest_eventsource::Error::StreamEnded) => {}
                Err(err) => {
                    match err {
                        // Error::Utf8(_) => {}
                        // Error::Parser(_) => {}
                        // Error::Transport(_) => {}
                        // Error::InvalidContentType(_, _) => {}
                        Error::InvalidStatusCode(code, response) => {
                            error!("Error {}: {}", code, response.text().await.unwrap());
                        }
                        // Error::InvalidLastEventId(_) => {}
                        Error::StreamEnded => {}
                        _ => println!("Error: {}", err),
                    }
                    event_source.close();
                }
            }
        }

        if stats.all_tokens.len() != 0 {
            stats_tx.send(stats).unwrap();
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

    info!("Chain {index} finished")
}

struct RequestStatistics {
    first_token: f32,
    all_tokens: Vec<f32>,
}

fn print_stats(mut values: Vec<f32>) {
    let mean = values.iter().map(|v| *v).reduce(|a, b| a + b).unwrap() / values.len() as f32;
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let min = values[0];
    let max = *values.last().unwrap();
    let median = values[values.len() / 2];

    println!("Mean: {mean}s ; Median: {median}s ; Min: {min}s ; Max: {max}s");
}

fn print_token_stats(mut values: Vec<usize>) {
    let mean = values.iter().map(|v| *v).reduce(|a, b| a + b).unwrap() / values.len();
    values.sort_unstable_by(|a, b| a.cmp(b));
    let min = values[0];
    let max = *values.last().unwrap();
    let median = values[values.len() / 2];

    println!(
        "Mean: {mean} tokens ; Median: {median} tokens ; Min: {min} tokens ; Max: {max} tokens"
    );
}
