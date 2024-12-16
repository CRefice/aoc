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

    fn day6(input: Vec<String>) -> Answers {
        fn turn_right((y, x): (isize, isize)) -> (isize, isize) {
            (x, -y)
        }

        let mut map = Map::from(&input);
        let mut pos = map.find('^').unwrap();
        let mut visited = HashSet::new();
        map[pos] = '.';

        let mut direction: (isize, isize) = (-1, 0);
        loop {
            visited.insert(pos);
            match map.step(pos, direction) {
                None => break,
                Some(p) if map[p] == '.' => {
                    pos = p;
                }
                Some(p) => {
                    assert_eq!(map[p], '#');
                    direction = turn_right(direction);
                }
            }
        }
        Self::part1(visited.len())
    }

    fn day7(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        struct Eqn {
            result: u64,
            operands: Vec<u64>,
        }

        impl Eqn {
            pub fn is_solvable(&self) -> bool {
                self.solve_rec(1, self.operands[0], false)
            }

            pub fn is_solvable_with_concat(&self) -> bool {
                self.solve_rec(1, self.operands[0], true)
            }

            fn solve_rec(&self, mut idx: usize, accumulator: u64, use_concat: bool) -> bool {
                if idx == self.operands.len() {
                    return accumulator == self.result;
                }
                if accumulator > self.result {
                    return false;
                }
                let x = self.operands[idx];
                idx += 1;
                if self.solve_rec(idx, accumulator * x, use_concat) {
                    return true;
                }
                if self.solve_rec(idx, accumulator + x, use_concat) {
                    return true;
                }
                if use_concat {
                    if let Some(accumulator) = Self::try_concat(accumulator, x) {
                        return self.solve_rec(idx, accumulator, use_concat);
                    }
                }
                false
            }

            fn try_concat(a: u64, b: u64) -> Option<u64> {
                let digits = num_digits(b);
                let multiplier = 10_u64.checked_pow(digits)?;
                a.checked_mul(multiplier)?.checked_add(b)
            }
        }

        let eqn = parse::uint::<u64>
            .left(": ")
            .pair(parse::uint::<u64>.interspersed(' '))
            .map(|(result, operands)| Eqn { result, operands });

        let eqns: Vec<Eqn> = input
            .iter()
            .map(|line| eqn.parse_exact(line).unwrap())
            .collect();

        let part1 = eqns
            .iter()
            .filter(|e| e.is_solvable())
            .map(|e| e.result)
            .sum::<u64>();

        let part2 = eqns
            .iter()
            .filter(|e| e.is_solvable_with_concat())
            .map(|e| e.result)
            .sum::<u64>();

        Self::solutions(part1, part2)
    }

    fn day8(input: Vec<String>) -> Answers {
        let height = input.len() as isize;
        let width = input[0].len() as isize;

        let mut antennas: HashMap<char, Vec<(isize, isize)>> = HashMap::new();
        for (i, row) in input.into_iter().enumerate() {
            for (j, c) in row.chars().enumerate() {
                if c != '.' {
                    antennas
                        .entry(c)
                        .or_default()
                        .push((i as isize, j as isize));
                }
            }
        }

        let mut antinodes = HashSet::new();
        for antennas in antennas.values() {
            let pairs = antennas
                .iter()
                .copied()
                .flat_map(|a1| antennas.iter().copied().map(move |a2| (a1, a2)));

            for (a, b) in pairs.filter(|&(a, b)| a != b) {
                let y = 2 * b.0 - a.0;
                let x = 2 * b.1 - a.1;
                if y >= 0 && y < height && x >= 0 && x < width {
                    antinodes.insert((y, x));
                }
            }
        }

        Self::part1(antinodes.len())
    }

    fn day9(input: Vec<String>) -> Answers {
        let line = input.into_iter().next().unwrap();
        let disk: Vec<u32> = line.chars().map(|c| c.to_digit(10).unwrap()).collect();

        fn part1(mut disk: Vec<u32>) -> u64 {
            let mut l = 0;
            let mut r = disk.len() - 1;

            let mut part1: u64 = 0;
            let mut blocks = 0;
            while l <= r {
                if l % 2 == 0 {
                    let lid = (l / 2) as u32;
                    let file = disk[l];
                    for _ in 0..file {
                        part1 += (lid * blocks) as u64;
                        blocks += 1;
                    }
                    l += 1;
                    continue;
                }
                let rid = (r / 2) as u32;
                while disk[r] > 0 && disk[l] > 0 {
                    part1 += (rid * blocks) as u64;
                    blocks += 1;
                    disk[r] -= 1;
                    disk[l] -= 1;
                }
                if disk[l] == 0 {
                    l += 1;
                } else {
                    r -= 2;
                }
            }
            part1
        }

        fn part2(mut disk: Vec<u32>) -> u64 {
            unimplemented!();
        }

        Self::solutions(part1(disk.clone()), part2(disk))
    }

    fn day10(input: Vec<String>) -> Answers {
        fn trailhead_score(pos: (usize, usize), map: &mut Map) -> usize {
            let c = map[pos];
            let x = map[pos].to_digit(10).unwrap();
            if x == 9 {
                return 1;
            }
            let mut count = 0;
            map[pos] = '.';
            for dir in [Map::UP, Map::LEFT, Map::RIGHT, Map::DOWN] {
                if let Some(p) =
                    map.step_if(pos, dir, |y| y.to_digit(10).is_some_and(|y| y == x + 1))
                {
                    count += trailhead_score(p, map)
                }
            }
            map[pos] = c;
            count
        }

        let mut map = Map::from(&input);
        let mut count = 0;
        for i in 0..map.height() {
            for j in 0..map.width() {
                if map[(i, j)] == '0' {
                    count += trailhead_score((i, j), &mut map);
                }
            }
        }
        Self::part1(count)
    }

    fn day11(input: Vec<String>) -> Answers {
        let line = input.into_iter().next().unwrap();
        let mut stones = parse::uint::<u64>
            .interspersed(parse::whitespace)
            .parse_exact(&line)
            .unwrap();

        fn blink(stones: &mut Vec<u64>) {
            for x in std::mem::take(stones) {
                if x == 0 {
                    stones.push(1);
                    continue;
                }
                let digits = num_digits(x);
                if digits % 2 == 0 {
                    let mask = 10_u64.pow(digits / 2);
                    stones.push(x / mask);
                    stones.push(x % mask);
                } else {
                    stones.push(x * 2024);
                }
            }
        }

        for _ in 0..25 {
            blink(&mut stones);
        }

        let part1 = stones.len();

        for _i in 25..75 {
            blink(&mut stones);
            dbg!(_i, stones.len());
        }

        let part2 = stones.len();
        Self::solutions(part1, part2)
    }

    fn day12(input: Vec<String>) -> Answers {
        unimplemented!()
    }

    fn day13(input: Vec<String>) -> Answers {
        struct Machine {
            a: (u64, u64),
            b: (u64, u64),
            target: (u64, u64),
        }

        impl Machine {
            fn mul((x, y): (u64, u64), i: u64) -> (u64, u64) {
                (x * i, y * i)
            }

            fn sub((x0, y0): (u64, u64), (x1, y1): (u64, u64)) -> Option<(u64, u64)> {
                x0.checked_sub(x1)
                    .and_then(|x| y0.checked_sub(y1).map(move |y| (x, y)))
            }

            fn steps_to(
                (x, y): (u64, u64),
                (tx, ty): (u64, u64),
                (dx, dy): (u64, u64),
            ) -> Option<u64> {
                Self::sub((tx, ty), (x, y))?;
                if (tx - x) % dx != 0 {
                    return None;
                }
                if (ty - y) % dy != 0 {
                    return None;
                }
                let stepsx = (tx - x) / dx;
                let stepsy = (ty - y) / dy;
                Some(stepsx).filter(|&x| x == stepsy)
            }

            fn solve(&self) -> Option<u64> {
                let mut pos = (0, 0);
                let mut best = None;
                for i in 0.. {
                    if pos == self.target {
                        return Some(i);
                    }
                    if let Some(steps) = Self::steps_to(pos, self.target, self.a) {
                        best = Some(i + steps * 3);
                    }
                    pos.0 += self.b.0;
                    pos.1 += self.b.1;
                    if Self::sub(self.target, pos).is_none() {
                        return best;
                    }
                }
                None
            }
        }

        let mut machines = input
            .split(String::is_empty)
            .map(|lines| {
                let button = "Button "
                    .right(parse::any_char)
                    .right(": X+")
                    .right(parse::uint::<u64>)
                    .pair(", Y+".right(parse::uint::<u64>));
                let prize = "Prize: X="
                    .right(parse::uint::<u64>)
                    .pair(", Y=".right(parse::uint::<u64>));

                let a = button.parse_exact(&lines[0]).unwrap();
                let b = button.parse_exact(&lines[1]).unwrap();
                let target = prize.parse_exact(&lines[2]).unwrap();
                if gcd(a.0, b.0) == 1 || gcd(a.1, b.1) == 1 {
                } else if Machine::steps_to(
                    (0, 0),
                    (target.0 + 10000000000000, target.1 + 10000000000000),
                    a,
                )
                .is_none()
                {
                    dbg!(a, b);
                }
                Machine { a, b, target }
            })
            .collect::<Vec<_>>();

        let part1 = machines.iter().flat_map(Machine::solve).sum::<u64>();

        machines.iter_mut().for_each(|m| {
            m.target.0 += 10000000000000;
            m.target.1 += 10000000000000;
        });

        let part2 = machines.iter().flat_map(Machine::solve).sum::<u64>();
        Self::solutions(part1, part2)
    }
}
