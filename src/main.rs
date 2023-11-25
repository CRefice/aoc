mod aoc2021;
mod aoc2022;
mod parse;
mod solutions;

use reqwest::header::COOKIE;
use solutions::Solutions;
use std::env;
use std::error::Error;
use std::io::{self, BufRead, BufReader};

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 2022)]
    year: u32,

    #[arg(long)]
    debug: bool,

    day: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv()?;

    let session = env::var("AOC_SESSION")?;
    let args = Args::parse();

    let lines: Vec<_> = if args.debug {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        reader.lines().collect::<Result<_, _>>()?
    } else {
        let url = format!("https://adventofcode.com/2022/day/{}/input", args.day);
        let client = reqwest::blocking::Client::new();
        let res = client
            .get(url)
            .header(COOKIE, format!("session={}", session))
            .send()?;

        let reader = BufReader::new(res);
        reader.lines().collect::<Result<_, _>>()?
    };

    let solution_fn = match args.year {
        2021 => aoc2021::Solutions2021::run,
        2022 => aoc2022::Solutions2022::run,
        _ => return Err("".into()),
    };

    let (part1, part2) = solution_fn(args.day, lines);
    println!("Part 1: {}", part1);
    if let Some(part2) = part2 {
        println!("Part 2: {}", part2);
    }

    Ok(())
}
