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

    fn day2(input: Vec<String>) -> Answers {
        /// Returns true iff num consists of two repeated numbers
        fn is_invalid_part1(num: u64) -> bool {
            let digits = num.to_string();
            let (a, b) = digits.split_at(digits.len() / 2);
            a == b
        }

        /// Returns true iff num consists of at least two repeated numbers
        fn is_invalid_part2(num: u64) -> bool {
            let digits: Vec<char> = num.to_string().chars().collect();

            for length in 1..=(digits.len() / 2) {
                let part = &digits[..length];
                if digits.chunks(length).all(|chunk| chunk == part) {
                    return true;
                }
            }
            false
        }

        let ranges: Vec<(u64, u64)> = parse::uint::<u64>
            .pair('-'.right(parse::uint::<u64>))
            .interspersed(',')
            .parse_exact(&input[0])
            .unwrap();

        let mut part1 = 0;
        let mut part2 = 0;
        for (start, end) in ranges {
            for x in start..=end {
                if is_invalid_part1(x) {
                    part1 += x;
                }
                if is_invalid_part2(x) {
                    part2 += x;
                }
            }
        }
        Self::solutions(part1, part2)
    }

    fn day3(input: Vec<String>) -> Answers {
        fn join_digits(it: impl IntoIterator<Item = u32>) -> u64 {
            it.into_iter().fold(0, |acc, x| acc * 10 + x as u64)
        }

        fn max_n(arr: &[u32], n: usize) -> Vec<u32> {
            let mut maximums = arr[..n].to_vec();
            for &x in &arr[n..] {
                maximums.push(x);
                // Try to "bump" at most one element up
                for i in 1..maximums.len() {
                    if maximums[i] > maximums[i - 1] {
                        maximums[i - 1..].rotate_left(1);
                        break;
                    }
                }
                maximums.pop();
            }
            maximums
        }

        let bank_parser = parse::zero_or_more(parse::any_char.map(|c| c.to_digit(10).unwrap()));
        let banks: Vec<Vec<u32>> = input
            .iter()
            .map(|s| bank_parser.parse_exact(s).unwrap())
            .collect();

        let part1 = banks
            .iter()
            .map(|bank| join_digits(max_n(bank, 2)))
            .sum::<u64>();

        let part2 = banks
            .iter()
            .map(|bank| join_digits(max_n(bank, 12)))
            .sum::<u64>();

        Self::solutions(part1, part2)
    }

    fn day4(input: Vec<String>) -> Answers {
        fn is_accessible(pos: (usize, usize), map: &Map) -> bool {
            map.neighbors(pos).filter(|&pos| map[pos] == '@').count() < 4
        }

        let map = Map::from(&input);

        let mut part1 = 0;
        for (i, row) in map.rows().enumerate() {
            for (j, cell) in row.iter().copied().enumerate() {
                if cell == '@' && is_accessible((i, j), &map) {
                    part1 += 1;
                }
            }
        }

        let mut map = map;
        let mut part2 = 0;
        loop {
            let mut removed: HashSet<(usize, usize)> = HashSet::new();
            for (i, row) in map.rows().enumerate() {
                for (j, cell) in row.iter().copied().enumerate() {
                    if cell == '@' && is_accessible((i, j), &map) {
                        removed.insert((i, j));
                    }
                }
            }
            if removed.is_empty() {
                break;
            }
            part2 += removed.len();
            for pos in removed {
                map[pos] = '.';
            }
        }
        Self::solutions(part1, part2)
    }
}
