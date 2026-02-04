#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Initializing RAG App with Rig...");
    llm_engine::forward_pass::test_candle()?;
    Ok(())
}
