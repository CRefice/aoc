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

    fn day5(input: Vec<String>) -> Answers {
        let mut chunks = input.split(String::is_empty);
        let ranges = chunks.next().unwrap();
        let ingredients = chunks.next().unwrap();

        let ranges = ranges
            .iter()
            .map(|line| {
                parse::uint::<u64>
                    .pair('-'.right(parse::uint::<u64>))
                    .parse_exact(line)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let ingredients = ingredients
            .iter()
            .map(|line| parse::uint::<u64>.parse_exact(line).unwrap());

        let part1 = ingredients
            .filter(|x| ranges.iter().any(|(a, b)| a <= x && x <= b))
            .count();

        #[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
        struct Boundary {
            x: u64,
            is_end: bool,
        }

        let mut boundaries: Vec<Boundary> = ranges
            .into_iter()
            .flat_map(|(a, b)| {
                [
                    Boundary {
                        x: a,
                        is_end: false,
                    },
                    Boundary { x: b, is_end: true },
                ]
            })
            .collect();
        boundaries.sort_unstable();

        let mut part2 = 0;
        let mut active = 0;
        let mut last_start = 0;
        dbg!(&boundaries);
        for Boundary { x, is_end } in boundaries {
            if !is_end {
                if active == 0 {
                    last_start = x;
                }
                active += 1;
            } else {
                active -= 1;
                if active == 0 {
                    part2 += x - last_start + 1;
                }
            }
        }

        Self::solutions(part1, part2)
    }

    fn day10(input: Vec<String>) -> Answers {
        let indicators = parse::surround(
            '[',
            parse::zero_or_more('.'.or('#').map(|x| x == '#')).map(|positions| {
                positions
                    .into_iter()
                    .enumerate()
                    .map(|(i, enabled)| (enabled as u64) << (i as u64))
                    .reduce(|x, y| x | y)
                    .unwrap()
            }),
            ']',
        );

        let buttons = parse::surround(
            '(',
            parse::uint::<u64>.interspersed(',').map(|x| {
                x.into_iter()
                    .map(|x| 1u64 << x)
                    .reduce(|x, y| x | y)
                    .unwrap()
            }),
            ')',
        )
        .interspersed(parse::whitespace);

        let parser = indicators.pair(parse::whitespace.right(buttons));

        fn min_presses(
            state: u64,
            target: u64,
            buttons: &[u64],
            pressed: &mut HashSet<u64>,
            memo: &mut HashMap<u64, usize>,
        ) -> Option<usize> {
            if state == target {
                return Some(0);
            } else if buttons.is_empty() {
                return None;
            }
            if let Some(answer) = memo.get(&state) {
                return Some(*answer);
            }
            let answer = buttons
                .iter()
                .copied()
                .filter_map(|button| {
                    if !pressed.insert(button) {
                        return None;
                    }
                    let state = state ^ button;
                    let answer = min_presses(state, target, buttons, pressed, memo);
                    pressed.remove(&button);
                    answer
                })
                .min()?
                .saturating_add(1);
            memo.insert(state, answer);
            eprintln!("{state} -> [{pressed:?}] = {answer:?}");
            Some(answer)
        }

        let mut total = 0;
        for line in input.iter() {
            let mut memo = HashMap::new();
            let (indicators, buttons) = parser.pars(line).unwrap().0;

            dbg!(&buttons);
            total += dbg!(min_presses(
                0,
                indicators,
                &buttons,
                &mut HashSet::new(),
                &mut memo
            ))
            .unwrap();
        }

        Self::part1(total)
    }

    fn day11(input: Vec<String>) -> Answers {
        let parser = parse::word
            .left(": ")
            .pair(parse::word.interspersed(parse::whitespace));

        let paths: HashMap<&str, Vec<&str>> = input
            .iter()
            .map(|line| parser.parse_exact(line).unwrap())
            .collect();

        fn num_paths(paths: &HashMap<&str, Vec<&str>>, from: &str, to: &str) -> usize {
            fn num_paths_memo<'a>(
                paths: &HashMap<&str, Vec<&'a str>>,
                from: &'a str,
                to: &str,
                memo: &mut HashMap<&'a str, usize>,
            ) -> usize {
                if from == to {
                    return 1;
                }
                let Some(neighbors) = paths.get(from) else {
                    return 0;
                };
                if let Some(ans) = memo.get(from) {
                    return *ans;
                }
                let ans = neighbors
                    .iter()
                    .map(|node| num_paths_memo(paths, node, to, memo))
                    .sum();
                memo.insert(from, ans);
                ans
            }

            num_paths_memo(paths, from, to, &mut HashMap::new())
        }

        let part1 = num_paths(&paths, "you", "out");
        let part2 = (num_paths(&paths, "svr", "dac")
            * num_paths(&paths, "dac", "fft")
            * num_paths(&paths, "fft", "out"))
            + (num_paths(&paths, "svr", "fft")
                * num_paths(&paths, "fft", "dac")
                * num_paths(&paths, "dac", "out"));
        Self::solutions(part1, part2)
    }
}
