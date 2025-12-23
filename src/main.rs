mod cli;
mod indexer;
pub mod tokenize;

fn main() -> Result<(), indexer::DynError> {
    let res = cli::run();
    match res {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(e)
        }
    }
}