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

    fn day2(input: Vec<String>) -> Answers {
        fn is_safe<T: Clone + Iterator<Item = i32>>(arr: T) -> bool {
            let mut diffs = arr.clone().zip(arr.skip(1)).map(|(a, b)| b - a);
            let first = diffs.next().unwrap();
            if first == 0 || first.abs() > 3 {
                return false;
            }
            let sign = first.signum();
            diffs.all(|d| d != 0 && d.abs() <= 3 && d.signum() == sign)
        }

        fn can_be_safe(arr: &[i32]) -> bool {
            let except = |i| {
                arr.iter()
                    .enumerate()
                    .filter(move |(j, _c)| i != *j)
                    .map(|(_j, c)| c)
                    .copied()
            };
            if is_safe(arr.iter().copied()) {
                return true;
            }
            (0..(arr.len())).map(except).any(is_safe)
        }

        let reports = input.iter().map(|line| {
            parse::uint::<i32>
                .interspersed(parse::whitespace)
                .parse_exact(line)
                .unwrap()
        });
        let part1 = reports
            .clone()
            .filter(|x| is_safe(x.iter().copied()))
            .count();
        let part2 = reports.filter(|x| can_be_safe(x)).count();
        Self::solutions(part1, part2)
    }

    fn day3(input: Vec<String>) -> Answers {
        #[derive(Clone, Copy)]
        enum Instr {
            Mul(u32, u32),
            Do,
            Dont,
        }

        let mul = "mul".right(parse::surround(
            '(',
            parse::uint::<u32>.pair(','.right(parse::uint::<u32>)),
            ')',
        ));

        let part1: u32 = {
            let mut sum: u32 = 0;
            for mut line in input.iter().map(AsRef::as_ref) {
                while let Some(((a, b), rest)) = mul.first_match(line) {
                    line = rest;
                    sum += a * b;
                }
            }
            sum
        };

        let parser = mul
            .map(|(a, b)| Instr::Mul(a, b))
            .or("do()".yielding(Instr::Do))
            .or("don't()".yielding(Instr::Dont));

        let part2 = {
            let mut sum: u32 = 0;
            let mut enabled = true;
            for mut line in input.iter().map(AsRef::as_ref) {
                while let Some((instr, rest)) = parser.first_match(line) {
                    line = rest;
                    match instr {
                        Instr::Mul(a, b) if enabled => {
                            sum += a * b;
                        }
                        Instr::Do => {
                            enabled = true;
                        }
                        Instr::Dont => {
                            enabled = false;
                        }
                        _ => {}
                    }
                }
            }
            sum
        };

        Self::solutions(part1, part2)
    }

    fn day4(input: Vec<String>) -> Answers {
        fn word_matches(
            map: &Map,
            pos: (usize, usize),
            direction: (isize, isize),
            word: &str,
        ) -> bool {
            word.chars()
                .zip(std::iter::successors(Some(pos), |p| {
                    map.step(*p, direction)
                }))
                .filter(|(c, pos)| map[*pos] == *c)
                .count()
                == word.len()
        }

        let map = Map::from(&input);
        let mut part1 = 0;
        for row in 0..map.height() {
            for col in 0..map.width() {
                for dx in [-1, 0, 1] {
                    for dy in [-1, 0, 1] {
                        if word_matches(&map, (row, col), (dy, dx), "XMAS") {
                            part1 += 1;
                        }
                    }
                }
            }
        }

        fn is_diagonal_center(
            map: &Map,
            pos: (usize, usize),
            (dx, dy): (isize, isize),
            word: &str,
        ) -> bool {
            assert!(word.len() % 2 == 1);
            let halflen = word.len() / 2;
            if map[pos] != word.chars().nth(halflen).unwrap() {
                return false;
            }
            let first = &word[..=halflen].chars().rev().collect::<String>();
            let first = &first;
            let second = &word[halflen..];

            (word_matches(map, pos, (-dx, -dy), first) && word_matches(map, pos, (dx, dy), second))
                || (word_matches(map, pos, (dx, dy), first)
                    && word_matches(map, pos, (-dx, -dy), second))
        }

        fn is_cross_center(map: &Map, pos: (usize, usize), word: &str) -> bool {
            // \ diagonal
            // / diagonal
            is_diagonal_center(map, pos, (1, 1), word)
                && is_diagonal_center(map, pos, (1, -1), word)
        }

        let mut part2 = 0;
        for row in 0..map.height() {
            for col in 0..map.width() {
                if is_cross_center(&map, (row, col), "MAS") {
                    part2 += 1;
                }
            }
        }

        Self::solutions(part1, part2)
    }

    fn day5(input: Vec<String>) -> Answers {
        fn meets_ordering(update: &[u32], constraints: &HashMap<u32, HashSet<u32>>) -> bool {
            let mut visited = HashSet::new();
            update.iter().copied().all(|x| {
                visited.insert(x);
                constraints[&x].is_disjoint(&visited)
            })
        }

        fn sort(update: &mut [u32], constraints: &HashMap<u32, HashSet<u32>>) {
            let len = update.len();
            for i in 0..len {
                let mut iter = update[..(len - i)].iter_mut();
                if let Some(mut left) = iter.next() {
                    for right in iter {
                        if constraints[right].contains(left) {
                            std::mem::swap(left, right)
                        }
                        left = right;
                    }
                }
            }
        }

        let mut constraints: HashMap<u32, HashSet<u32>> = HashMap::new();
        let order = parse::uint::<u32>.left('|').pair(parse::uint::<u32>);

        let mut lines = input.iter();
        for (a, b) in lines
            .by_ref()
            .take_while(|line| !line.is_empty())
            .map(|line| order.parse_exact(line).unwrap())
        {
            constraints.entry(a).or_default().insert(b);
            constraints.entry(b).or_default();
        }

        let update = parse::uint::<u32>.interspersed(',');
        let (ordered, unordered): (Vec<_>, Vec<_>) = lines
            .map(|line| update.parse_exact(line).unwrap())
            .partition(|update| meets_ordering(update, &constraints));

        fn mid_element(arr: &[u32]) -> u32 {
            arr[arr.len() / 2]
        }

        let part1 = ordered
            .iter()
            .map(AsRef::as_ref)
            .map(mid_element)
            .sum::<u32>();

        let part2 = unordered
            .into_iter()
            .map(|mut line| {
                sort(&mut line, &constraints);
                mid_element(&line)
            })
            .sum::<u32>();

        Self::solutions(part1, part2)
    }
}
