

use std::io::Write as _;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use llm_engine::generator::Generator;
use llm_engine::sampler::{Greedy, Temperature, TopP};
use llm_engine::{Config, Llama};

struct Args {
    model_dir: PathBuf,
    usecase:   String,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let get = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .and_then(|i| args.get(i + 1))
                .map(|s| s.to_owned())
        };
        Self {
            model_dir: get("--model").unwrap_or_default().into(),
            usecase:   get("--usecase").unwrap_or_else(|| "text".into()),
        }
    }
}

fn load(dir: &Path) -> anyhow::Result<Generator> {
    let device = Device::Cpu;


    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    shards.sort();

    anyhow::ensure!(
        !shards.is_empty(),
        "No .safetensors files found in {}.\nRun the setup command shown in the --help output.",
        dir.display()
    );

    println!("Loading {} shard(s) from {} …", shards.len(), dir.display());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)?
    };
    let model = Llama::new(Config::llama_3_8b(), vb)?;

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    println!("Model loaded.\n");
    Ok(Generator::new(model, tokenizer, device))
}

fn text_completion(generator: &Generator) -> anyhow::Result<()> {
    let prompt = "The history of the Rust programming language began in 2006 when";

    println!("━━━ Text Completion (greedy) ━━━");
    println!("Prompt : {prompt}");
    print!("Output : {prompt}");
    std::io::stdout().flush()?;

    generator.print_streaming(prompt, 120, &Greedy)?;
    println!("\n");
    Ok(())
}

fn code_completion(generator: &Generator) -> anyhow::Result<()> {
    let prompt = r#"/// Compute the nth Fibonacci number iteratively.
///
/// # Examples
/// ```
/// assert_eq!(fibonacci(10), 55);
/// ```
pub fn fibonacci(n: u64) -> u64 {"#;

    println!("━━━ Code Completion (greedy) ━━━");
    println!("{prompt}");

    generator.print_streaming(prompt, 200, &Greedy)?;
    println!("\n");
    Ok(())
}

fn question_answering(generator: &Generator) -> anyhow::Result<()> {
    let question = "What is the difference between stack and heap memory in Rust, \
                    and when should I use Box<T>?";

    println!("━━━ Question Answering (temperature = 0.6) ━━━");
    println!("Q: {question}");
    print!("A: ");
    std::io::stdout().flush()?;

    let sampler = Temperature::new(0.6);
    let answer = generator.chat(question, None, 300, &sampler)?;
    println!("{answer}\n");
    Ok(())
}

fn summarization(generator: &Generator) -> anyhow::Result<()> {
    let document = "\
        Rust is a multi-paradigm, general-purpose programming language that emphasises \
        performance, type safety, and concurrency. It achieves memory safety without a \
        garbage collector through a system of ownership, borrowing, and lifetimes enforced \
        at compile time. Originally designed by Graydon Hoare at Mozilla Research and first \
        released in 2010, Rust has grown into a community-driven language used in operating \
        systems, web servers, embedded firmware, game engines, and WebAssembly. Its package \
        manager, Cargo, integrates dependency management, testing, and documentation. The \
        language was voted the most admired programming language in the Stack Overflow \
        Developer Survey every year from 2016 to 2023.";

    let prompt = format!(
        "Summarise the following passage in a single sentence.\n\n\
         Passage: {document}\n\n\
         Summary:"
    );

    println!("━━━ Summarization (top-p = 0.9) ━━━");
    println!("Document:\n  {document}\n");
    print!("Summary : ");
    std::io::stdout().flush()?;

    let sampler = TopP::new(0.9, 0.7);
    generator.print_streaming(&prompt, 80, &sampler)?;
    println!("\n");
    Ok(())
}

fn chat_demo(generator: &Generator) -> anyhow::Result<()> {
    let system = "You are a concise Rust programming tutor. \
                  Explain concepts clearly using short code examples.";

    let turns = [
        "What is ownership in Rust?",
        "Can you show a small example of a borrow checker error?",
    ];

    println!("━━━ Chat (top-p = 0.9, system prompt) ━━━");

    let sampler = TopP::new(0.9, 0.7);


    let mut context = format!(
        "<|begin_of_text|>\
         <|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
    );

    for user_msg in &turns {
        context.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n"
        ));

        println!("User      : {user_msg}");
        print!("Assistant : ");
        std::io::stdout().flush()?;

        let reply = generator.print_streaming(&context, 200, &sampler)?;
        println!("\n");


        context.push_str(&reply);
        context.push_str("<|eot_id|>");
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.model_dir.as_os_str().is_empty() {
        eprintln!(
            "Usage: cargo run -p llm-engine --bin generate -- \\\n\
             \t--model <path>  --usecase <text|code|qa|summarize|chat>\n\n\
             Setup: huggingface-cli download meta-llama/Meta-Llama-3-8B \\\n\
             \t--local-dir ./weights/llama-3-8b --include '*.safetensors' 'tokenizer.json'"
        );
        std::process::exit(1);
    }

    let generator = load(&args.model_dir)?;

    match args.usecase.as_str() {
        "text"      => text_completion(&generator),
        "code"      => code_completion(&generator),
        "qa"        => question_answering(&generator),
        "summarize" => summarization(&generator),
        "chat"      => chat_demo(&generator),
        other => {
            anyhow::bail!(
                "Unknown usecase '{other}'. Choose from: text, code, qa, summarize, chat"
            )
        }
    }
}
