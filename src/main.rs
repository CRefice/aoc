mod aoc2021;
mod solutions;

use reqwest::header::COOKIE;
use solutions::Solutions;
use std::env;
use std::error::Error;
use std::io::{self, BufRead, BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv()?;

    let session = env::var("AOC_SESSION")?;

    let day: usize = env::args().nth(1).unwrap().parse()?;

    let debug_flag = env::args().nth(2);

    let lines: Vec<_> = if let Some("--debug") = debug_flag.as_deref() {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        reader.lines().collect::<Result<_, _>>()?
    } else {
        let url = format!("https://adventofcode.com/2021/day/{}/input", day);
        let client = reqwest::blocking::Client::new();
        let res = client
            .get(url)
            .header(COOKIE, format!("session={}", session))
            .send()?;

        let reader = BufReader::new(res);
        reader.lines().collect::<Result<_, _>>()?
    };

    let (part1, part2) = aoc2021::Solutions2021::run(day, lines);
    println!("Part 1: {}", part1);
    if let Some(part2) = part2 {
        println!("Part 2: {}", part2);
    }

    Ok(())
}
