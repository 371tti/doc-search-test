use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;


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