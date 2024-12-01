use aoc::parse::{self, Parser};
use aoc::solutions::{Answers, Solutions};
use aoc::util::*;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::iter;

pub struct Solutions2024;

impl Solutions for Solutions2024 {
    fn day1(input: Vec<String>) -> Answers {
        let parser = parse::uint::<u32>.pair(parse::whitespace.right(parse::uint::<u32>));
        let (mut left, mut right): (Vec<u32>, Vec<u32>) = input
            .iter()
            .map(|line| parser.parse_exact(line).unwrap())
            .unzip();
        left.sort_unstable();
        right.sort_unstable();

        let distances = left.iter().zip(right.iter()).map(|(a, b)| a.abs_diff(*b));
        let part1 = distances.sum::<u32>();

        let mut counts = HashMap::new();
        for num in right.iter() {
            *counts.entry(num).or_insert(0) += 1;
        }

        let similarities = left
            .iter()
            .zip(left.iter().map(|x| counts.get(x).copied().unwrap_or(0)))
            .map(|(x, count)| x * count);
        let part2 = similarities.sum::<u32>();

        Self::solutions(part1, part2)
    }
}
