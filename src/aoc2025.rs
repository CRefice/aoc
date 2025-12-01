use aoc::parse::{self, Parser};
use aoc::solutions::Answers;
use aoc::util::*;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::iter;

pub struct Solutions;

impl aoc::solutions::Solutions for Solutions {
    fn day1(input: Vec<String>) -> Answers {
        let parser = parse::any_char.pair(parse::uint::<i32>);
        let instructions = input.iter().map(|line| parser.parse_exact(line).unwrap());

        const DIAL_TICKS: i32 = 100;
        let mut dial = 50;
        let mut part1 = 0;
        let mut part2 = 0;
        for (direction, amount) in instructions {
            let full_turns = amount / DIAL_TICKS;
            let amount = amount % DIAL_TICKS;
            // A full turn always passes by 0
            part2 += full_turns;
            match direction {
                'R' => {
                    // Pass by 0 if reach 100 or above
                    if dial + amount >= DIAL_TICKS {
                        part2 += 1;
                    }
                    dial += amount;
                }
                'L' => {
                    // Pass by 0 if we cross from positive to non-positive
                    if dial > 0 && (dial - amount) <= 0 {
                        part2 += 1;
                    }
                    dial -= amount;
                }
                _ => unreachable!(),
            }

            dial = dial.rem_euclid(DIAL_TICKS);
            if dial == 0 {
                part1 += 1;
            }
        }
        Self::solutions(part1, part2)
    }
}
