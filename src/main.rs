mod aoc2021;
mod aoc2022;
mod aoc2023;
mod parse;
mod solutions;

use is_terminal::IsTerminal;
use reqwest::header::COOKIE;
use solutions::Solutions;
use std::env;
use std::io::{self, BufRead, BufReader};

use clap::Parser;

/// Advent of code solutions
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 2023)]
    year: u32,

    #[arg(long)]
    debug: bool,

    day: usize,
}

fn main() {
    dotenv::dotenv().ok();

    let args = Args::parse();

    let solution_fn = match args.year {
        2021 => aoc2021::Solutions2021::run,
        2022 => aoc2022::Solutions2022::run,
        2023 => aoc2023::Solutions2023::run,
        year => unimplemented!("No available solutions for {year}"),
    };

    let lines: Vec<_> = if args.debug {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        reader
            .lines()
            .collect::<Result<_, _>>()
            .expect("Could not read from standard input")
    } else {
        let session =
            env::var("AOC_SESSION").expect("Could not read AOC_SESSION environment variable");
        let url = format!(
            "https://adventofcode.com/{}/day/{}/input",
            args.year, args.day
        );
        let client = reqwest::blocking::Client::new();
        let res = client
            .get(url)
            .header(COOKIE, format!("session={}", session))
            .send()
            .expect("Could not request input file from server");

        let reader = BufReader::new(res);
        reader
            .lines()
            .collect::<Result<_, _>>()
            .expect("Could not read server response")
    };

    let (part1, part2) = solution_fn(args.day, lines);
    if std::io::stdout().is_terminal() {
        println!("Part 1: {}", part1);
        if let Some(part2) = part2 {
            println!("Part 2: {}", part2);
        }
    } else {
        print!("{}", part2.unwrap_or(part1));
    }
}
