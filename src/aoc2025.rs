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

    fn day6(input: Vec<String>) -> Answers {
        let (ops, input) = input.split_last().unwrap();
        let split_points: Vec<usize> = ops
            .match_indices(['+', '*'])
            .map(|m| m.0)
            .chain(std::iter::once(ops.len() + 1))
            .collect();

        let ranges = split_points
            .iter()
            .zip(&split_points[1..])
            .map(|(&start, &end)| start..(end - 1)); // -1 because whitespace separator

        fn apply(c: char) -> impl Fn(u64, u64) -> u64 {
            match c {
                '+' => |x, y| x + y,
                '*' => |x, y| x * y,
                _ => unreachable!(),
            }
        }

        let mut part1 = 0;
        for range in ranges.clone() {
            let op: char = ops.chars().nth(range.start).unwrap();
            let column = input
                .iter()
                .map(|line| line[range.clone()].trim().parse::<u64>().unwrap());
            part1 += column.reduce(apply(op)).unwrap();
        }

        let mut part2 = 0;
        for range in ranges {
            let op: char = ops.chars().nth(range.start).unwrap();

            let columns = range.map(|i| {
                input
                    .iter()
                    .map(|line| line.chars().nth(i).unwrap())
                    .filter(|c| !c.is_ascii_whitespace())
                    .collect::<String>()
                    .parse::<u64>()
                    .unwrap()
            });

            part2 += columns.reduce(apply(op)).unwrap();
        }
        Self::solutions(part1, part2)
    }

    fn day7(input: Vec<String>) -> Answers {
        let map = Map::from(&input);

        fn beam_splits(map: &Map) -> usize {
            let start = map.find('S').unwrap();

            let mut set = HashSet::from_iter([start]);

            let mut splits = 0;
            while !set.is_empty() {
                dbg!(set.len());
                let mut new_set = HashSet::new();
                for pos in set.into_iter() {
                    let Some(next) = map.step(pos, Map::DOWN) else {
                        continue;
                    };
                    if map[next] == '^' {
                        map.step(next, Map::LEFT).map(|x| new_set.insert(x));
                        map.step(next, Map::RIGHT).map(|x| new_set.insert(x));
                        splits += 1;
                    } else {
                        new_set.insert(next);
                    }
                }
                set = new_set;
            }
            splits
        }

        fn num_paths(map: &Map) -> usize {
            let mut paths = vec![0; map.width()];
            for row in map.rows() {
                dbg!(&paths);
                for (i, c) in row.iter().enumerate() {
                    match c {
                        'S' => paths[i] = 1,
                        '^' => {
                            paths[i - 1] += paths[i];
                            paths[i + 1] += paths[i];
                            paths[i] = 0;
                        }
                        _ => (),
                    }
                }
            }
            paths.into_iter().sum()
        }

        Self::solutions(beam_splits(&map), num_paths(&map))
    }

    fn day8(input: Vec<String>) -> Answers {
        struct UnionFind {
            ids: Vec<usize>,
            sizes: Vec<usize>,
        }
        impl UnionFind {
            fn new(size: usize) -> Self {
                Self {
                    ids: (0..size).collect(),
                    sizes: vec![1; size],
                }
            }

            fn union(&mut self, a: usize, b: usize) {
                let mut a = self.find_compress(a);
                let mut b = self.find_compress(b);

                if a == b {
                    return;
                }
                if self.sizes[a] < self.sizes[b] {
                    (b, a) = (a, b)
                }
                self.ids[b] = a;
                self.sizes[a] += self.sizes[b];
            }

            fn find(&self, a: usize) -> usize {
                let mut root = a;
                while self.ids[root] != root {
                    root = self.ids[root];
                }
                root
            }

            fn find_compress(&mut self, a: usize) -> usize {
                let mut root = a;
                while self.ids[root] != root {
                    root = self.ids[root];
                }
                let mut a = a;
                while self.ids[a] != a {
                    let next = self.ids[a];
                    self.ids[a] = root;
                    a = next;
                }
                root
            }

            fn group_sizes(&self) -> HashMap<usize, usize> {
                let mut sizes = HashMap::new();
                for x in 0..(self.ids.len()) {
                    *sizes.entry(self.find(x)).or_default() += 1;
                }
                sizes
            }
        }

        fn dot(a: (u64, u64, u64), b: (u64, u64, u64)) -> u64 {
            let d0 = a.0.abs_diff(b.0);
            let d1 = a.1.abs_diff(b.1);
            let d2 = a.2.abs_diff(b.2);
            d0 * d0 + d1 * d1 + d2 * d2
        }

        let boxes: Vec<(u64, u64, u64)> = input
            .iter()
            .map(|line| {
                let parts = parse::uint::<u64>
                    .interspersed(',')
                    .parse_exact(line)
                    .unwrap();
                (parts[0], parts[1], parts[2])
            })
            .collect();

        let mut pairs = (0..boxes.len())
            .flat_map(|i| ((i + 1)..boxes.len()).map(move |j| (i, j)))
            .collect::<Vec<_>>();

        pairs.sort_unstable_by_key(|&(i, j)| dot(boxes[i], boxes[j]));

        let mut uf = UnionFind::new(boxes.len());
        let mut last_pair = (0, 0);
        for &(i, j) in &pairs[..(1000.min(pairs.len()))] {
            uf.union(i, j);
            last_pair = (i, j);
        }

        let mut part1 = uf.group_sizes().values().copied().collect::<Vec<_>>();
        part1.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
        let part1 = part1[0] * part1[1] * part1[2];

        for &(i, j) in &pairs[(1000.min(pairs.len()))..] {
            uf.union(i, j);
            last_pair = (i, j);
            if uf.group_sizes().len() == 1 {
                break;
            }
        }

        let part2 = boxes[last_pair.0].0 * boxes[last_pair.1].0;
        Self::solutions(part1, part2)
    }

    fn day9(input: Vec<String>) -> Answers {
        let tiles: Vec<(u64, u64)> = input
            .iter()
            .map(|line| {
                parse::uint::<u64>
                    .pair(','.right(parse::uint::<u64>))
                    .parse_exact(line)
                    .unwrap()
            })
            .collect();

        let pairs = (0..tiles.len()).flat_map(|i| ((i + 1)..tiles.len()).map(move |j| (i, j)));
        let rectangles = pairs.clone().map(|(i, j)| {
            let left = tiles[i].0.min(tiles[j].0);
            let right = tiles[i].0.max(tiles[j].0);
            let top = tiles[i].1.min(tiles[j].1);
            let bottom = tiles[i].1.max(tiles[j].1);
            (left, right, top, bottom)
        });

        let area = |(left, right, top, bottom)| (right - left + 1) * (bottom - top + 1);

        let part1 = rectangles.clone().map(area).max().unwrap();

        let edges = tiles.iter().zip(tiles.iter().cycle().skip(1));
        let part2 = rectangles
            .filter(|(left, right, top, bottom)| {
                edges.clone().all(|((ax, ay), (bx, by))| {
                    if ax == bx {
                        // horiz edge
                        (ay >= bottom && by >= bottom)
                            || (ay <= top && by <= top)
                            || (ax <= left && bx <= left)
                            || (ax >= right && bx >= right)
                    } else {
                        // vertical edge
                        (ax >= right && bx >= right)
                            || (ax <= left && bx <= left)
                            || (ay <= top && by <= top)
                            || (ay >= bottom && by >= bottom)
                    }
                })
            })
            .map(area)
            .max()
            .unwrap();

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

        let joltages = parse::surround('{', parse::uint::<u64>.interspersed(','), '}');

        let parser = indicators
            .pair(parse::whitespace.right(buttons))
            .pair(parse::whitespace.right(joltages));

        fn precompute_combinations(buttons: &[u64]) -> HashMap<u64, Vec<Vec<u64>>> {
            fn rec(
                state: u64,
                buttons: &[u64],
                pressed: &mut Vec<u64>,
                out: &mut HashMap<u64, Vec<Vec<u64>>>,
            ) {
                let Some((button, buttons)) = buttons.split_first() else {
                    out.entry(state).or_default().push(pressed.clone());
                    return;
                };

                // First recurse without pressing button
                rec(state, buttons, pressed, out);

                // Then with pressing
                let state = state ^ button;
                pressed.push(*button);
                rec(state, buttons, pressed, out);
                pressed.pop();
            }

            let mut out = HashMap::new();
            rec(0, buttons, &mut Vec::new(), &mut out);
            out
        }

        struct BitIndexIter(u64);

        impl Iterator for BitIndexIter {
            type Item = usize;

            fn next(&mut self) -> Option<Self::Item> {
                if self.0 == 0 {
                    None
                } else {
                    let idx = self.0.trailing_zeros();
                    self.0 &= self.0.wrapping_sub(1);
                    Some(idx.try_into().unwrap())
                }
            }
        }

        fn min_presses_to_reach(
            joltages: Vec<u64>,
            combinations: &HashMap<u64, Vec<Vec<u64>>>,
        ) -> Option<usize> {
            if joltages.iter().all(|&x| x == 0) {
                return Some(0);
            }

            let diff = joltages
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let bit = (!x.is_multiple_of(2)) as u64;
                    bit << i
                })
                .fold(0, |x, y| x | y);

            combinations
                .get(&diff)?
                .iter()
                .filter_map(|combo| {
                    let mut joltages = joltages.clone();
                    for button in combo {
                        for idx in BitIndexIter(*button) {
                            joltages[idx] = joltages[idx].checked_sub(1)?;
                        }
                    }
                    for j in joltages.iter() {
                        debug_assert!(j.is_multiple_of(2))
                    }
                    joltages.iter_mut().for_each(|x| *x /= 2);
                    min_presses_to_reach(joltages, combinations).map(|x| 2 * x + combo.len())
                })
                .min()
        }

        let mut part1 = 0;
        let mut part2 = 0;
        for line in input.iter() {
            let ((indicators, buttons), joltages) = parser.parse_exact(line).unwrap();

            let combos = precompute_combinations(&buttons);
            part1 += combos[&indicators].iter().map(|x| x.len()).min().unwrap();
            part2 += min_presses_to_reach(joltages, &combos).unwrap();
        }

        Self::solutions(part1, part2)
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

    fn day12(input: Vec<String>) -> Answers {
        let mut parts = input.split(String::is_empty);
        let regions = parts.nth(6).unwrap();

        let area = parse::uint::<u64>.left('x').pair(parse::uint::<u64>);
        let presents = parse::uint::<u64>.interspersed(parse::whitespace);

        let region = area.left(": ").pair(presents);

        let regions = regions.iter().map(|line| region.parse_exact(line).unwrap());
        Self::part1(
            regions
                .filter(|((w, h), presents)| w * h >= 9 * presents.iter().sum::<u64>())
                .count(),
        )
    }
}
