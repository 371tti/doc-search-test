mod cli;
mod indexer;
pub mod tokenize;

fn main() -> Result<(), indexer::DynError> {
    cli::run()
}