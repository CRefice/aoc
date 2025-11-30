use aoc::parse::{self, Parser};
use aoc::solutions::Answers;
use aoc::util::*;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::iter;

pub struct Solutions;

impl aoc::solutions::Solutions for Solutions {
    fn day1(input: Vec<String>) -> Answers {
        let part1 = input.iter().map(|line| {
            let mut digits = line.chars().filter_map(|c| c.to_digit(10));
            let a = digits.next().unwrap();
            let b = digits.last().unwrap_or(a);
            a * 10 + b
        });
        let part2 = input.iter().map(|line| {
            let suffixes = (0..line.len()).map(|i| &line[i..]);
            let p = "one"
                .yielding(1)
                .or("two".yielding(2))
                .or("three".yielding(3))
                .or("four".yielding(4))
                .or("five".yielding(5))
                .or("six".yielding(6))
                .or("seven".yielding(7))
                .or("eight".yielding(8))
                .or("nine".yielding(9));

            let mut digits = suffixes.filter_map(|s| {
                let literal = s.chars().next().and_then(|c| c.to_digit(10));
                literal.or_else(|| p.parse_exact(s).ok())
            });
            let a = digits.next().unwrap();
            let b = digits.last().unwrap_or(a);
            a * 10 + b
        });
        Self::solutions(part1.sum::<u32>(), part2.sum::<u32>())
    }

    fn day2(input: Vec<String>) -> Answers {
        fn max_possible(color: &str) -> u32 {
            match color {
                "red" => 12,
                "green" => 13,
                "blue" => 14,
                _ => unreachable!(),
            }
        }

        let subset = parse::uint::<u32>
            .pair(parse::trimmed(parse::word))
            .interspersed(", ");

        let p = "Game "
            .right(parse::uint::<u32>)
            .left(": ")
            .pair(subset.interspersed("; "));

        let games = input
            .iter()
            .map(|line| p.parse_exact(line).unwrap())
            .collect::<Vec<_>>();

        let possible = games.iter().filter(|(_, sets)| {
            sets.iter().all(|subset| {
                subset
                    .iter()
                    .all(|(amount, color)| max_possible(color) >= *amount)
            })
        });

        let part1 = possible.map(|(id, _)| id).sum::<u32>();

        let powers = games.iter().map(|(_, sets)| {
            let (mut max_red, mut max_green, mut max_blue) = (0, 0, 0);
            for set in sets {
                for (num, color) in set {
                    match *color {
                        "red" => max_red = max_red.max(*num),
                        "green" => max_green = max_green.max(*num),
                        "blue" => max_blue = max_blue.max(*num),
                        _ => unreachable!(),
                    }
                }
            }
            max_red * max_green * max_blue
        });
        let part2 = powers.sum::<u32>();

        Self::solutions(part1, part2)
    }

    fn day3(input: Vec<String>) -> Answers {
        fn is_symbol(c: char) -> bool {
            !c.is_numeric() && c != '.'
        }

        fn symbol_neighbors(
            input: &[String],
            line: usize,
            start: usize,
            end: usize,
        ) -> Vec<(char, usize, usize)> {
            let mut vec = Vec::new();
            let line_start = line.saturating_sub(1);
            let line_end = line.saturating_add(1).min(input.len() - 1);
            let start = start.saturating_sub(1);
            let end = end.saturating_add(1).min(input[0].len() - 1);
            for (row, line) in input.iter().enumerate().skip(line_start).take(line_end + 1) {
                let neighbors = line[start..=end]
                    .chars()
                    .zip(start..=end)
                    .filter(|(c, _)| is_symbol(*c))
                    .map(|(c, col)| (c, row, col));
                vec.extend(neighbors);
            }
            vec
        }

        let mut neighbor_nums = HashMap::new();
        let mut part1 = 0;
        for i in 0..input.len() {
            let mut chars = input[i].chars().enumerate();
            loop {
                let mut nums = chars
                    .by_ref()
                    .skip_while(|(_, c)| !c.is_numeric())
                    .take_while(|(_, c)| c.is_numeric());
                let Some((start, _)) = nums.next() else {
                    break;
                };
                let (end, _) = nums.last().unwrap_or((start, ' '));
                let num = input[i][start..=end].parse::<u32>().unwrap();

                let neighbors = symbol_neighbors(&input[..], i, start, end);
                if !neighbors.is_empty() {
                    part1 += num;
                }
                for (_, i, j) in neighbors.into_iter().filter(|(c, _, _)| *c == '*') {
                    neighbor_nums.entry((i, j)).or_insert(Vec::new()).push(num);
                }
            }
        }

        let part2 = neighbor_nums
            .values()
            .filter(|v| v.len() == 2)
            .map(|v| v.iter().product::<u32>())
            .sum::<u32>();
        Self::solutions(part1, part2)
    }

    fn day4(input: Vec<String>) -> Answers {
        let cards = parse::trimmed("Card")
            .right(parse::uint::<u32>)
            .left(':')
            .right(
                parse::zero_or_more(parse::whitespace.right(parse::uint::<u32>))
                    .map(HashSet::from_iter),
            )
            .left(parse::whitespace.right('|'))
            .pair(
                parse::zero_or_more(parse::whitespace.right(parse::uint::<u32>))
                    .map(HashSet::from_iter),
            );

        let cards = input
            .iter()
            .map(|line| cards.parse_exact(line).unwrap())
            .collect::<Vec<_>>();

        let matching = cards
            .iter()
            .map(|(winning, mine): &(HashSet<u32>, HashSet<u32>)| {
                winning.intersection(mine).count()
            });

        let score = matching
            .clone()
            .map(|wins| if wins >= 1 { 1 << (wins - 1) } else { 0 })
            .sum::<u64>();

        let part2 = matching
            .scan(
                VecDeque::new(),
                |cards: &mut VecDeque<usize>, wins: usize| {
                    let instances: usize = 1 + cards.pop_front().unwrap_or(0);
                    let mut won = iter::repeat(instances).take(wins);
                    for (card, instances) in cards.iter_mut().zip(won.by_ref()) {
                        *card += instances;
                    }
                    for _ in won {
                        cards.push_back(instances);
                    }
                    Some(instances)
                },
            )
            .sum::<usize>();

        Self::solutions(score, part2)
    }

    fn day5(input: Vec<String>) -> Answers {
        fn find_in_map(key: u64, map: &[(u64, u64, u64)]) -> Option<u64> {
            let index = map.partition_point(|elem| elem.1 <= key);
            if index == 0 {
                return None;
            }
            let (to, from, len) = map[index - 1];
            let offset = key - from;
            if offset >= len {
                None
            } else {
                Some(to + offset)
            }
        }

        fn end_point((_to, from, len): (u64, u64, u64)) -> u64 {
            from + len
        }

        fn find_range_in_map(
            (mut start, mut len): (u64, u64),
            map: &[(u64, u64, u64)],
        ) -> Vec<(u64, u64)> {
            let a = map.partition_point(|elem| end_point(*elem) < start);
            // a: first intersecting range
            if a == map.len() {
                return vec![(start, len)];
            }
            let b = map.partition_point(|(_to, from, _chunk_len)| *from <= (start + len));
            // b: first non-intersecting range (from > end)

            let mut res = Vec::new();
            for (to, from, chunk_len) in map[a..b].iter().cloned() {
                if len == 0 {
                    break;
                }

                if start <= from {
                    let l = from - start;
                    res.push((start, l));
                    start = from;
                    len -= l;
                }

                let offset = start - from;
                if offset < chunk_len {
                    // [------------]
                    //            [...]
                    //
                    // [     ]     [       ]
                    // +
                    //     [         ]
                    //  =
                    //     [ ][   ][ ]
                    let chunk_len = len.min(chunk_len - offset);
                    start += chunk_len;
                    len -= chunk_len;
                    res.push((to + offset, chunk_len));
                }
            }
            if len > 0 {
                res.push((start, len));
            }
            res
        }

        let mut chunks = input.split(|s| s.is_empty());

        let seeds = "seeds: "
            .right(parse::uint::<u64>.interspersed(' '))
            .parse_exact(chunks.next().and_then(|c| c.first()).unwrap())
            .unwrap();

        let maps: Vec<_> = chunks
            .map(|c| {
                let mut map: Vec<(u64, u64, u64)> = c
                    .iter()
                    .skip(1)
                    .map(|line| {
                        parse::uint
                            .interspersed(' ')
                            .map(|v| (v[0], v[1], v[2]))
                            .parse_exact(line)
                            .unwrap()
                    })
                    .collect();
                map.sort_by_key(|elem| elem.1);
                map
            })
            .collect();

        let part1 = seeds
            .iter()
            .cloned()
            .map(|mut seed| {
                for map in maps.iter() {
                    seed = find_in_map(seed, map).unwrap_or(seed);
                }
                seed
            })
            .min()
            .unwrap();

        let mut part2: Vec<(u64, u64)> =
            seeds.chunks(2).map(|chunk| (chunk[0], chunk[1])).collect();

        for map in maps.iter() {
            part2 = part2
                .into_iter()
                .flat_map(|range| find_range_in_map(range, map))
                .collect();
        }

        let part2 = part2.into_iter().map(|chunk| chunk.0).min().unwrap();
        Self::solutions(part1, part2)
    }

    fn day6(input: Vec<String>) -> Answers {
        fn find_losing_time(
            mut l: u64,
            mut r: u64,
            time: u64,
            distance: u64,
            mut cmp: impl FnMut(u64, u64) -> bool,
        ) -> u64 {
            while l + 1 < r {
                let t = l + (r - l) / 2;
                if cmp(t * (time - t), distance) {
                    l = t;
                } else {
                    r = t;
                }
            }
            l
        }

        fn error_margin(time: u64, distance: u64) -> u64 {
            let best_time = time / 2;
            let min = find_losing_time(0, best_time, time, distance, |a, b| a <= b);
            let max = find_losing_time(best_time, time, time, distance, |a, b| a > b);
            max - min
        }

        fn combine_digits(nums: &[u64]) -> u64 {
            let mut num = 0u64;
            for n in nums {
                num *= 10_u64.pow(n.ilog10() + 1);
                num += *n;
            }
            num
        }

        let times: Vec<u64> = "Time:"
            .left(parse::whitespace)
            .right(parse::uint.interspersed(parse::whitespace))
            .parse_exact(&input[0])
            .unwrap();
        let distances: Vec<u64> = "Distance:"
            .left(parse::whitespace)
            .right(parse::uint.interspersed(parse::whitespace))
            .parse_exact(&input[1])
            .unwrap();

        let part1 = times
            .iter()
            .zip(distances.iter())
            .map(|(&time, &distance)| error_margin(time, distance))
            .product::<u64>();

        let time = combine_digits(&times);
        let distance = combine_digits(&distances);
        let part2 = error_margin(time, distance);

        Self::solutions(part1, part2)
    }

    fn day7(input: Vec<String>) -> Answers {
        #[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
        enum Type {
            Card,
            Pair,
            TwoPair,
            Three,
            Full,
            Four,
            Five,
        }

        impl Type {
            fn of(hand: &str) -> Self {
                let mut map = HashMap::new();
                for c in hand.chars() {
                    *map.entry(c).or_insert(0u32) += 1;
                }
                let mut v: Vec<u32> = map.values().copied().collect();
                v.sort_by(|a, b| b.cmp(a));
                let mut v = v.into_iter();
                match v.next().unwrap() {
                    5 => Self::Five,
                    4 => Self::Four,
                    3 => {
                        if v.next().unwrap() == 2 {
                            Self::Full
                        } else {
                            Self::Three
                        }
                    }
                    2 => {
                        if v.next().unwrap() == 2 {
                            Self::TwoPair
                        } else {
                            Self::Pair
                        }
                    }
                    _ => Self::Card,
                }
            }

            fn joker_rule(hand: &str) -> Self {
                let mut map = HashMap::new();
                for c in hand.chars() {
                    *map.entry(c).or_insert(0u32) += 1;
                }
                let jokers = map.remove(&'J').unwrap_or(0);

                let mut v: Vec<u32> = map.values().copied().collect();
                v.sort_by(|a, b| b.cmp(a));
                let mut v = v.into_iter();
                match v.next().unwrap_or(0) + jokers {
                    5 => Self::Five,
                    4 => Self::Four,
                    3 => {
                        if v.next().unwrap() == 2 {
                            Self::Full
                        } else {
                            Self::Three
                        }
                    }
                    2 => {
                        if v.next().unwrap() == 2 {
                            Self::TwoPair
                        } else {
                            Self::Pair
                        }
                    }
                    _ => Self::Card,
                }
            }
        }

        fn compare_hands(
            mine: &str,
            theirs: &str,
            mut rule: impl FnMut(&str) -> Type,
            order: &[char],
        ) -> Ordering {
            let card_order = |card: char| order.iter().rev().position(|&c| c == card).unwrap();

            let a = rule(mine);
            let b = rule(theirs);
            a.cmp(&b).then_with(|| {
                mine.chars()
                    .map(card_order)
                    .cmp(theirs.chars().map(card_order))
            })
        }

        let mut cards: Vec<(&str, u32)> = input
            .iter()
            .map(|line| {
                parse::predicate(|c| c.is_alphanumeric())
                    .left(' ')
                    .pair(parse::uint)
                    .parse_exact(line)
                    .unwrap()
            })
            .collect();

        let default_order = [
            'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
        ];
        let joker_order = [
            'A', 'K', 'Q', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'J',
        ];

        cards.sort_by(|(a, _), (b, _)| compare_hands(a, b, Type::of, &default_order));
        let part1 = cards
            .iter()
            .zip(1..)
            .map(|((_, bid), rank)| bid * rank)
            .sum::<u32>();

        cards.sort_by(|(a, _), (b, _)| compare_hands(a, b, Type::joker_rule, &joker_order));
        let part2 = cards
            .iter()
            .zip(1..)
            .map(|((_, bid), rank)| bid * rank)
            .sum::<u32>();
        Self::solutions(part1, part2)
    }

    fn day8(input: Vec<String>) -> Answers {
        fn search(
            start: &str,
            mut is_end: impl FnMut(&str) -> bool,
            mut instructions: impl Iterator<Item = (usize, char)>,
            map: &HashMap<&str, (&str, &str)>,
        ) -> usize {
            let mut node = start;
            loop {
                let (step, dir) = instructions.next().unwrap();
                if is_end(node) {
                    break step;
                }
                let (l, r) = map.get(node).unwrap();
                node = match dir {
                    'L' => l,
                    'R' => r,
                    _ => unreachable!(),
                };
            }
        }

        let instructions: &str = &input[0];
        let map: HashMap<&str, (&str, &str)> = input[2..]
            .iter()
            .map(|line| {
                parse::predicate(char::is_alphanumeric)
                    .left(" = ")
                    .pair(
                        '('.right(parse::predicate(char::is_alphanumeric))
                            .left(", ")
                            .pair(parse::predicate(char::is_alphanumeric))
                            .left(')'),
                    )
                    .parse_exact(line)
                    .unwrap()
            })
            .collect();

        let instructions = instructions.chars().cycle().enumerate();
        let part1 = search("AAA", |node| node == "ZZZ", instructions.clone(), &map);

        let part2 = {
            let nodes = map.keys().filter(|node| node.ends_with('A'));
            let steps = nodes.map(|start| {
                search(
                    start,
                    |node| node.ends_with('Z'),
                    instructions.clone(),
                    &map,
                )
            });
            steps.reduce(lcm).unwrap()
        };
        Self::solutions(part1, part2)
    }

    fn day9(input: Vec<String>) -> Answers {
        fn differences(seq: &[i64]) -> Vec<i64> {
            seq.iter().zip(&seq[1..]).map(|(a, b)| b - a).collect()
        }

        fn extrapolate(seq: &[i64]) -> (i64, i64) {
            if seq.iter().all(|&x| x == 0) {
                return (0, 0);
            }

            let (first, last) = extrapolate(&differences(seq));
            let first = seq.first().unwrap() - first;
            let last = seq.last().unwrap() + last;
            (first, last)
        }

        let sequences = input.iter().map(|line| {
            parse::int::<i64>
                .interspersed(parse::whitespace)
                .parse_exact(line)
                .unwrap()
        });

        let (part2, part1) = sequences
            .map(|seq| extrapolate(&seq))
            .reduce(|(a, b), (c, d)| (a + c, b + d))
            .unwrap();
        Self::solutions(part1, part2)
    }

    fn day10(input: Vec<String>) -> Answers {
        fn offset(
            map: &Map,
            (row, col): (usize, usize),
            (or, oc): (isize, isize),
        ) -> Option<(usize, usize)> {
            row.checked_add_signed(or)
                .and_then(|row| col.checked_add_signed(oc).map(|col| (row, col)))
                .filter(|&(row, col)| row < map.height() && col < map.width())
        }

        fn connectivity(map: &Map, (row, col): (usize, usize)) -> HashSet<(usize, usize)> {
            let offsets = match map[row][col] {
                '|' => &[(1, 0), (-1, 0)][..],
                '-' => &[(0, 1), (0, -1)],
                'L' => &[(-1, 0), (0, 1)],
                'J' => &[(-1, 0), (0, -1)],
                '7' => &[(1, 0), (0, -1)],
                'F' => &[(1, 0), (0, 1)],
                'S' => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
                _ => &[],
            };
            offsets
                .iter()
                .filter_map(|off| offset(map, (row, col), *off))
                .collect()
        }

        let map: Map = Map::from(&input);

        let start = map
            .rows()
            .enumerate()
            .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, c)| ((i, j), c)))
            .find(|((_i, _j), &c)| c == 'S')
            .map(|pair| pair.0)
            .unwrap();

        let pipes = {
            let mut pipes = vec![start];
            let mut prev = start;
            let mut next = connectivity(&map, start)
                .into_iter()
                .find(|&pipe| connectivity(&map, pipe).contains(&start))
                .unwrap();
            while map[next.0][next.1] != 'S' {
                pipes.push(next);

                let mut edges = connectivity(&map, next);
                edges.remove(&prev);

                prev = next;
                next = edges
                    .into_iter()
                    .find(|&pipe| connectivity(&map, pipe).contains(&next))
                    .unwrap();
            }
            pipes
        };

        fn left(direction: (isize, isize)) -> (isize, isize) {
            let (y, x) = direction;
            (-x, y)
        }

        fn right(direction: (isize, isize)) -> (isize, isize) {
            let (y, x) = direction;
            (x, -y)
        }

        #[derive(Debug)]
        enum Side {
            Left,
            Right,
        }

        fn checked_offset(
            (y, x): (usize, usize),
            direction: (isize, isize),
            (height, width): (usize, usize),
        ) -> Option<(usize, usize)> {
            let Some(y) = y.checked_add_signed(direction.0) else {
                return None;
            };
            let Some(x) = x.checked_add_signed(direction.1) else {
                return None;
            };
            if y >= height || x >= width {
                return None;
            }
            Some((y, x))
        }

        fn walk(
            start: (usize, usize),
            direction: (isize, isize),
            dimensions: (usize, usize),
        ) -> impl Iterator<Item = (usize, usize)> {
            iter::successors(checked_offset(start, direction, dimensions), move |&pos| {
                checked_offset(pos, direction, dimensions)
            })
        }

        fn out_side(path: &[(usize, usize)], dimensions: (usize, usize)) -> Side {
            let pipes: HashSet<(usize, usize)> = HashSet::from_iter(path.iter().copied());
            for (from, to) in path.iter().zip(&path[1..]) {
                let direction = (
                    to.0 as isize - from.0 as isize,
                    to.1 as isize - from.1 as isize,
                );
                let left = left(direction);
                let right = right(direction);
                for start in [from, to] {
                    if walk(*start, left, dimensions).all(|pos| !pipes.contains(&pos)) {
                        return Side::Left;
                    }
                    if walk(*start, right, dimensions).all(|pos| !pipes.contains(&pos)) {
                        return Side::Right;
                    }
                }
            }
            unreachable!()
        }

        fn flood_fill(
            start: (usize, usize),
            direction: (isize, isize),
            visited: &mut HashSet<(usize, usize)>,
            dimensions: (usize, usize),
        ) {
            let mut boundary: Vec<(usize, usize)> = checked_offset(start, direction, dimensions)
                .into_iter()
                .collect();

            while let Some(pos) = boundary.pop() {
                if !visited.insert(pos) {
                    continue;
                }
                for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(p) = checked_offset(pos, dir, dimensions) {
                        boundary.push(p);
                    }
                }
            }
        }

        let dimensions = (map.height(), map.width());
        let side = out_side(&pipes, dimensions);
        let direction = match side {
            Side::Left => left,
            Side::Right => right,
        };

        let mut visited: HashSet<(usize, usize)> = HashSet::from_iter(pipes.iter().copied());
        for (from, to) in pipes.iter().zip(&pipes[1..]) {
            let dir = direction((
                to.0 as isize - from.0 as isize,
                to.1 as isize - from.1 as isize,
            ));
            flood_fill(*from, dir, &mut visited, dimensions);
            flood_fill(*to, dir, &mut visited, dimensions);
        }
        let outside = dimensions.0 * dimensions.1 - visited.len();

        let pipes: HashSet<(usize, usize)> = HashSet::from_iter(pipes);

        Self::solutions(pipes.len() / 2, outside)
    }

    fn day11(input: Vec<String>) -> Answers {
        fn expand_indices(
            empties: impl Iterator<Item = bool>,
            expand_amount: usize,
        ) -> impl Iterator<Item = usize> {
            empties.scan(0, move |state, is_empty| {
                let x = *state;
                *state += if is_empty { expand_amount } else { 1 };
                Some(x)
            })
        }

        fn find_galaxies(map: &Map, expand_amount: usize) -> Vec<(usize, usize)> {
            let actual_rows: Vec<usize> = expand_indices(
                map.rows().map(|row| row.iter().all(|&c| c != '#')),
                expand_amount,
            )
            .collect();
            let (width, height) = (map.width(), map.height());
            let actual_cols: Vec<usize> = expand_indices(
                (0..width).map(|col| (0..height).all(|row| map[(row, col)] != '#')),
                expand_amount,
            )
            .collect();

            (0..height)
                .flat_map(|row| (0..width).map(move |col| (row, col)))
                .filter_map(|(row, col)| {
                    if map[(row, col)] == '#' {
                        Some((actual_rows[row], actual_cols[col]))
                    } else {
                        None
                    }
                })
                .collect()
        }

        fn distances(galaxies: &[(usize, usize)]) -> usize {
            let combinations = galaxies
                .iter()
                .copied()
                .enumerate()
                .skip(1)
                .flat_map(|(i, g1)| galaxies[..i].iter().copied().map(move |g2| (g1, g2)));

            fn cartesian_distance((x1, y1): (usize, usize), (x2, y2): (usize, usize)) -> usize {
                x1.abs_diff(x2) + y1.abs_diff(y2)
            }

            combinations
                .map(|(g1, g2)| cartesian_distance(g1, g2))
                .sum()
        }

        let map = Map::from(&input);
        let part1 = {
            let galaxies = find_galaxies(&map, 2);
            distances(&galaxies)
        };
        let part2 = {
            let galaxies = find_galaxies(&map, 1000000);
            distances(&galaxies)
        };

        Self::solutions(part1, part2)
    }

    fn day12(input: Vec<String>) -> Answers {
        fn possible_arrangements(row: &str, groups: &[u64]) -> u64 {
            let mut dp = vec![vec![0]; groups.len() + 1];
            dp[0][0] = 1;

            for c in row.chars().chain(iter::once('.')) {
                if c == '#' {
                    for bucket in dp.iter_mut() {
                        bucket.push(0);
                    }
                } else {
                    for i in (0..groups.len()).rev() {
                        if let Some(v) = dp[i].iter().rev().nth(groups[i] as usize).copied() {
                            *dp[i + 1].last_mut().unwrap() += v;
                        }

                        if c == '?' {
                            let num = dp[i].last().copied().unwrap_or(0);
                            dp[i].push(num);
                        } else {
                            let len = dp[i].len() - 1;
                            dp[i].drain(0..len);
                        }
                    }
                }
            }

            *dp.last().and_then(|v| v.last()).unwrap()
        }

        let rows: Vec<_> = input
            .iter()
            .map(|line| {
                let mut chunks = line.split_whitespace();
                let (row, groups) = (chunks.next().unwrap(), chunks.next().unwrap());
                let groups = parse::uint::<u64>
                    .interspersed(',')
                    .parse_exact(groups)
                    .unwrap();
                (row, groups)
            })
            .collect();

        let part1: u64 = rows
            .iter()
            .map(|(row, groups)| possible_arrangements(row, groups))
            .sum();

        let part2: u64 = rows
            .iter()
            .map(|(row, groups)| {
                let row = [*row; 5].join("?");
                let groups = groups.repeat(5);
                possible_arrangements(&row, &groups)
            })
            .sum();

        Self::solutions(part1, part2)
    }

    fn day13(input: Vec<String>) -> Answers {
        fn find_mirror_plane(map: &Map) -> Option<usize> {
            for (index, _) in map
                .rows()
                .zip(map.rows().skip(1))
                .enumerate()
                .filter(|(_, (a, b))| a == b)
            {
                if map
                    .rows()
                    .take(index + 1)
                    .rev()
                    .zip(map.rows().skip(index + 1))
                    .all(|(a, b)| a == b)
                {
                    return Some(index + 1);
                }
            }
            None
        }

        fn pairwise_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
            a.iter()
                .zip(b)
                .map(|(a, b)| if a == b { 0 } else { 1 })
                .sum()
        }

        fn find_smudged_mirror_plane(map: &Map) -> Option<usize> {
            for (index, _) in map
                .rows()
                .zip(map.rows().skip(1))
                .enumerate()
                .filter(|(_, (a, b))| pairwise_distance(a, b) <= 1)
            {
                let mut allowed_distance = 1;
                if map
                    .rows()
                    .take(index + 1)
                    .rev()
                    .zip(map.rows().skip(index + 1))
                    .all(|(a, b)| {
                        let dist = pairwise_distance(a, b);
                        if dist <= allowed_distance {
                            allowed_distance -= dist;
                            true
                        } else {
                            false
                        }
                    })
                    && allowed_distance == 0
                {
                    return Some(index + 1);
                }
            }
            None
        }

        let maps: Vec<Map> = input.split(|s| s.is_empty()).map(Map::from).collect();

        let part1 = maps
            .iter()
            .map(|map| {
                if let Some(horiz) = find_mirror_plane(map) {
                    100 * horiz
                } else {
                    find_mirror_plane(&map.transposed()).expect("Needs to have a mirror plane")
                }
            })
            .sum::<usize>();
        let part2 = maps
            .iter()
            .map(|map| {
                if let Some(horiz) = find_smudged_mirror_plane(map) {
                    100 * horiz
                } else {
                    find_smudged_mirror_plane(&map.transposed())
                        .expect("Needs to have a mirror plane")
                }
            })
            .sum::<usize>();

        Self::solutions(part1, part2)
    }

    fn day14(input: Vec<String>) -> Answers {
        fn slide_vertical(map: &mut Map, dir: isize, rows: impl Iterator<Item = usize> + Clone) {
            for col in 0..map.width() {
                let mut rows = rows.clone().peekable();
                let mut slide_to = *rows.peek().unwrap();
                for row in rows {
                    match map[(row, col)] {
                        '#' => {
                            slide_to = row.checked_add_signed(dir).unwrap_or(0);
                        }
                        'O' => {
                            map[(row, col)] = '.';
                            map[(slide_to, col)] = 'O';
                            slide_to = slide_to.checked_add_signed(dir).unwrap_or(0);
                        }
                        _ => (),
                    }
                }
            }
        }

        fn slide_horizontal(map: &mut Map, dir: isize, cols: impl Iterator<Item = usize> + Clone) {
            for row in map.rows_mut() {
                let mut cols = cols.clone().peekable();
                let mut slide_to = *cols.peek().unwrap();
                for col in cols {
                    match row[col] {
                        '#' => {
                            slide_to = col.checked_add_signed(dir).unwrap_or(0);
                        }
                        'O' => {
                            row[col] = '.';
                            row[slide_to] = 'O';
                            slide_to = slide_to.checked_add_signed(dir).unwrap_or(0);
                        }
                        _ => (),
                    }
                }
            }
        }

        fn spin_cycle(map: &Map) -> Option<Map> {
            let mut map = map.clone();
            let (height, width) = (map.height(), map.width());
            slide_vertical(&mut map, 1, 0..height);
            slide_horizontal(&mut map, 1, 0..width);
            slide_vertical(&mut map, -1, (0..height).rev());
            slide_horizontal(&mut map, -1, (0..width).rev());
            Some(map)
        }

        let map: Map = Map::from(&input);
        let part1 = {
            let mut map = map.clone();
            let width = map.width();
            slide_vertical(&mut map, 1, 0..width);

            map.rows()
                .zip((1..=map.height()).rev())
                .map(|(row, index)| index * row.iter().filter(|&&c| c == 'O').count())
                .sum::<usize>()
        };

        // Returns (start, length) of cycle
        fn find_cycle(map: Map) -> (Map, usize, usize) {
            let mut seen = HashMap::new();
            for (i, map) in iter::successors(Some(map), spin_cycle).enumerate() {
                if let Some(oldi) = seen.insert(map.clone(), i) {
                    return (map, oldi, i - oldi);
                }
            }
            unreachable!()
        }

        let part2 = {
            const TARGET: usize = 1000000000;

            let (mut map, start, length) = find_cycle(map);
            let additional = (TARGET - start) % length;
            for _ in 0..additional {
                map = spin_cycle(&map).unwrap();
            }

            map.rows()
                .zip((1..=map.height()).rev())
                .map(|(row, index)| index * row.iter().filter(|&&c| c == 'O').count())
                .sum::<usize>()
        };

        Self::solutions(part1, part2)
    }

    fn day15(input: Vec<String>) -> Answers {
        fn hash(s: &str) -> usize {
            s.chars().fold(0, |curr, elem| {
                let curr = curr + (elem as usize);
                (curr * 17) % 256
            })
        }

        let seq = input[0].split(',');
        let part1 = seq.clone().map(hash).sum::<usize>();

        let mut boxes = vec![OrderedMap::new(); 256];
        for instr in seq {
            if let Some((label, length)) = instr.split_once('=') {
                let length = length.parse::<usize>().unwrap();
                let idx = hash(label);
                if let Some(l) = boxes[idx].get_mut(label) {
                    *l = length;
                } else {
                    boxes[idx].insert(label, length);
                }
            } else {
                let label = instr.trim_end_matches('-');
                let idx = hash(label);
                boxes[idx].remove(label);
            }
        }

        let part2 = boxes
            .into_iter()
            .zip(1..)
            .map(|(bx, i)| {
                bx.into_iter()
                    .zip(1..)
                    .map(|(length, j)| i * j * length)
                    .sum::<usize>()
            })
            .sum::<usize>();

        Self::solutions(part1, part2)
    }

    fn day16(input: Vec<String>) -> Answers {
        #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
        struct Beam {
            pos: (usize, usize),
            dir: (isize, isize),
        }

        fn reflect_right((x, y): (isize, isize)) -> (isize, isize) {
            (-y, -x)
        }

        fn reflect_left((x, y): (isize, isize)) -> (isize, isize) {
            (y, x)
        }

        fn maybe_split_horiz_wall((_y, x): (isize, isize)) -> Option<[(isize, isize); 2]> {
            if x != 0 {
                None
            } else {
                Some([(0, -1), (0, 1)])
            }
        }

        fn maybe_split_vert_wall((y, _x): (isize, isize)) -> Option<[(isize, isize); 2]> {
            if y != 0 {
                None
            } else {
                Some([(1, 0), (-1, 0)])
            }
        }

        fn traverse_beam(
            start: Beam,
            parents: &mut Vec<Beam>,
            map: &Map,
            memo: &mut HashMap<Beam, HashSet<(usize, usize)>>,
        ) -> usize {
            if let Some(cycle) = memo.get(&start).cloned() {
                // Found cycle: at this point memo[start] will have the longest path, so
                // propagate it to all the parents
                for parent in parents.iter() {
                    memo.get_mut(parent).unwrap().extend(&cycle);
                }
                return cycle.len();
            }

            memo.insert(start, HashSet::new());
            parents.push(start);

            let Beam { mut pos, mut dir } = start;
            loop {
                for parent in parents.iter() {
                    memo.get_mut(parent).unwrap().insert(pos);
                }

                match map[pos] {
                    '/' => dir = reflect_right(dir),
                    '\\' => dir = reflect_left(dir),
                    '|' => {
                        if let Some([a, b]) = maybe_split_vert_wall(dir) {
                            traverse_beam(Beam { pos, dir: a }, parents, map, memo);
                            traverse_beam(Beam { pos, dir: b }, parents, map, memo);
                            break;
                        }
                    }
                    '-' => {
                        if let Some([a, b]) = maybe_split_horiz_wall(dir) {
                            traverse_beam(Beam { pos, dir: a }, parents, map, memo);
                            traverse_beam(Beam { pos, dir: b }, parents, map, memo);
                            break;
                        }
                    }
                    _ => (),
                }
                let Some(next) = map.step(pos, dir) else {
                    break;
                };
                pos = next;
            }

            assert_eq!(Some(start), parents.pop());
            memo[&start].len()
        }

        let map = Map::from(&input);

        let beam = Beam {
            pos: (0, 0),
            dir: (0, 1),
        };
        let mut memo = HashMap::new();
        let mut parents = Vec::new();
        let part1 = traverse_beam(beam, &mut parents, &map, &mut memo);

        let part2 = {
            let beams = (0..map.width())
                .map(|col| ((0, col), (1, 0))) // Top
                .chain((0..map.height()).map(|row| ((row, 0), (0, 1)))) // Left
                .chain((0..map.width()).map(|col| ((map.height() - 1, col), (-1, 0)))) // Bottom
                .chain((0..map.height()).map(|row| ((row, map.width() - 1), (0, -1)))) // Right
                .map(|(pos, dir)| Beam { pos, dir });

            beams
                .map(|beam| traverse_beam(beam, &mut parents, &map, &mut memo))
                .max()
                .unwrap()
        };

        Self::solutions(part1, part2)
    }

    fn day17(input: Vec<String>) -> Answers {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        struct Node {
            pos: (usize, usize),
            dir: (isize, isize),
            consecutive: u32,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Item(u32, Node);

        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.cmp(&other.0).reverse()
            }
        }

        fn left(direction: (isize, isize)) -> (isize, isize) {
            let (y, x) = direction;
            (-x, y)
        }

        fn right(direction: (isize, isize)) -> (isize, isize) {
            let (y, x) = direction;
            (x, -y)
        }

        fn step_normal(Item(cost, node): &Item, dir: (isize, isize), map: &Map) -> Option<Item> {
            let pos = map.step(node.pos, dir)?;
            let cost = cost + map[pos].to_digit(10).unwrap();
            let consecutive = 1 + if node.dir == dir { node.consecutive } else { 0 };
            if consecutive > 3 {
                None
            } else {
                Some(Item(
                    cost,
                    Node {
                        pos,
                        dir,
                        consecutive,
                    },
                ))
            }
        }

        fn step_ultra(Item(cost, node): &Item, dir: (isize, isize), map: &Map) -> Option<Item> {
            let pos = map.step(node.pos, dir)?;
            let cost = cost + map[pos].to_digit(10).unwrap();
            let consecutive = 1 + if node.dir == dir { node.consecutive } else { 0 };
            if consecutive > 10 || (node.dir != dir && node.consecutive < 4) {
                None
            } else {
                Some(Item(
                    cost,
                    Node {
                        pos,
                        dir,
                        consecutive,
                    },
                ))
            }
        }

        fn neighbors<F>(item: &Item, map: &Map, step_fn: &F) -> impl Iterator<Item = Item>
        where
            F: Fn(&Item, (isize, isize), &Map) -> Option<Item>,
        {
            let dir = item.1.dir;
            step_fn(item, dir, map)
                .into_iter()
                .chain(step_fn(item, left(dir), map))
                .chain(step_fn(item, right(dir), map))
        }

        fn min_costs<F>(map: &Map, start: Node, step_fn: &F) -> HashMap<Node, u32>
        where
            F: Fn(&Item, (isize, isize), &Map) -> Option<Item>,
        {
            let mut costs = vec![vec![HashMap::new(); map.width()]; map.height()];
            costs[0][0].insert(start, 0);

            let mut pq = BinaryHeap::from([Item(0, start)]);
            while let Some(item) = pq.pop() {
                for neighbor @ Item(cost, node) in neighbors(&item, map, step_fn) {
                    let min = costs[node.pos.0][node.pos.1]
                        .entry(node)
                        .or_insert(u32::MAX);
                    if &cost < min {
                        *min = cost;
                        pq.push(neighbor);
                    }
                }
            }

            let end = (map.height() - 1, map.width() - 1);
            costs[end.0].pop().unwrap()
        }

        let map = Map::from(&input);

        let start = Node {
            pos: (0, 0),
            dir: (0, 1),
            consecutive: 0,
        };
        let part1 = min_costs(&map, start, &step_normal)
            .into_values()
            .min()
            .unwrap();
        let part2 = min_costs(&map, start, &step_ultra)
            .into_iter()
            .filter(|(node, _)| node.consecutive >= 4)
            .map(|entry| entry.1)
            .min()
            .unwrap();

        Self::solutions(part1, part2)
    }

    fn day18(input: Vec<String>) -> Answers {
        fn area(steps: &[((isize, isize), isize)]) -> usize {
            fn cross((y0, x0): (isize, isize), (y1, x1): (isize, isize)) -> isize {
                x0 * y1 - x1 * y0
            }
            fn mul((y0, x0): (isize, isize), amt: isize) -> (isize, isize) {
                (y0 * amt, x0 * amt)
            }

            let directions = steps.iter().map(|(dir, _distance)| dir).copied();

            let windings = directions
                .clone()
                .zip(directions.cycle().skip(1))
                .map(|(a, b)| cross(a, b));

            let steps =
                steps
                    .iter()
                    .copied()
                    .zip(windings)
                    .scan(1, |side, ((dir, distance), winding)| {
                        let extra = if *side == winding { winding } else { 0 };
                        *side = winding;
                        Some(mul(dir, distance + extra))
                    });

            let segments = steps.scan((0, 0), |pos, dir| {
                let start = *pos;
                let end = (start.0 + dir.0, start.1 + dir.1);
                *pos = end;
                Some((start, end))
            });

            segments
                .map(|(a, b)| cross(a, b))
                .sum::<isize>()
                .unsigned_abs()
                / 2
        }

        let p = parse::any_char
            .pair(parse::surround(
                parse::whitespace,
                parse::uint::<isize>,
                parse::whitespace,
            ))
            .pair(parse::surround(
                '(',
                '#'.right(parse::predicate(|c| c.is_ascii_hexdigit())),
                ')',
            ));

        let steps = input.iter().map(|line| p.parse_exact(line).unwrap());

        let part1 = {
            let steps = steps
                .clone()
                .map(|((dir, distance), _)| {
                    let dir = match dir {
                        'U' => (-1, 0),
                        'D' => (1, 0),
                        'L' => (0, -1),
                        'R' => (0, 1),
                        _ => unreachable!(),
                    };
                    (dir, distance)
                })
                .collect::<Vec<_>>();
            area(&steps)
        };

        let part2 = {
            let steps = steps
                .map(|(_, hex)| {
                    let distance = isize::from_str_radix(&hex[..5], 16).unwrap();
                    let dir = match hex.chars().last().unwrap() {
                        '0' => (0, 1),
                        '1' => (1, 0),
                        '2' => (0, -1),
                        '3' => (-1, 0),
                        _ => unreachable!(),
                    };
                    (dir, distance)
                })
                .collect::<Vec<_>>();
            area(&steps)
        };

        Self::solutions(part1, part2)
    }

    fn day19(input: Vec<String>) -> Answers {
        #[derive(Debug, Clone, Copy)]
        enum Rule<'a> {
            Condition {
                part: char,
                ord: Ordering,
                num: u32,
                dest: &'a str,
            },
            Fallback {
                dest: &'a str,
            },
        }

        impl<'a> Rule<'a> {
            pub fn apply(&self, item: &Ratings) -> Option<&'a str> {
                match self {
                    Rule::Fallback { dest } => Some(dest),
                    Rule::Condition {
                        part,
                        ord,
                        num,
                        dest,
                    } => {
                        let i = match part {
                            'x' => 0,
                            'm' => 1,
                            'a' => 2,
                            's' => 3,
                            _ => unreachable!(),
                        };
                        if item[i].cmp(num) == *ord {
                            Some(dest)
                        } else {
                            None
                        }
                    }
                }
            }
        }

        type Ratings = [u32; 4];

        fn is_accepted(item: &Ratings, workflows: &HashMap<&str, Vec<Rule>>) -> bool {
            let mut name = "in";
            while name != "A" && name != "R" {
                name = workflows[name]
                    .iter()
                    .filter_map(|rule| rule.apply(item))
                    .next()
                    .unwrap();
            }
            name == "A"
        }

        fn find_accepted<'a>(
            workflows: &HashMap<&'a str, Vec<Rule<'a>>>,
        ) -> Vec<Vec<(Rule<'a>, bool)>> {
            let mut ret = Vec::new();
            let mut search = vec![("in", 0, Vec::new())];
            while let Some((name, i, path)) = search.pop() {
                if name == "A" {
                    ret.push(path);
                    continue;
                } else if name == "R" {
                    continue;
                }

                let rule = workflows[name][i];
                match rule {
                    Rule::Condition { dest, .. } => {
                        let mut succ_path = path.clone();
                        succ_path.push((rule, true));
                        search.push((dest, 0, succ_path));

                        let mut fail_path = path;
                        fail_path.push((rule, false));
                        search.push((name, i + 1, fail_path));
                    }
                    Rule::Fallback { dest } => {
                        search.push((dest, 0, path));
                    }
                }
            }
            ret
        }

        type RatingsRange = [(u32, u32); 4];

        fn trim_range(range: &mut RatingsRange, path: &[(Rule<'_>, bool)]) {
            for (rule, cond) in path {
                let Rule::Condition { part, ord, num, .. } = rule else {
                    unreachable!("path contains non-condition rule");
                };
                let i = match part {
                    'x' => 0,
                    'm' => 1,
                    'a' => 2,
                    's' => 3,
                    _ => unreachable!(),
                };
                let range = &mut range[i];
                match (ord, cond) {
                    (Ordering::Less, true) => {
                        range.1 = range.1.min(num - 1);
                    }
                    (Ordering::Less, false) => {
                        range.0 = range.0.max(*num);
                    }
                    (Ordering::Greater, true) => {
                        range.0 = range.0.max(num + 1);
                    }
                    (Ordering::Greater, false) => {
                        range.1 = range.1.min(*num);
                    }
                    _ => unreachable!(),
                }
            }
        }

        let (workflow, ratings) = {
            let condition = parse::any_char
                .pair(parse::predicate(|c| c == '<' || c == '>').map(|c| match c {
                    "<" => Ordering::Less,
                    ">" => Ordering::Greater,
                    x => unreachable!("unexpected: {}", x),
                }))
                .pair(parse::uint::<u32>)
                .left(':')
                .pair(parse::word);
            let rule = condition
                .map(|(((part, ord), num), dest)| Rule::Condition {
                    part,
                    ord,
                    num,
                    dest,
                })
                .or(parse::word.map(|dest| Rule::Fallback { dest }));
            let workflow = parse::word.pair(parse::surround('{', rule.interspersed(','), '}'));

            let rating = parse::any_char.right('=').right(parse::uint::<u32>);
            let ratings = parse::surround(
                '{',
                rating.interspersed(',').map(|v| v.try_into().unwrap()),
                '}',
            );
            (workflow, ratings)
        };

        let mut chunks = input.split(String::is_empty);
        let workflows: HashMap<&str, Vec<Rule>> = chunks
            .next()
            .unwrap()
            .iter()
            .map(|line| workflow.parse_exact(line).unwrap())
            .collect();

        let ratings: Vec<Ratings> = chunks
            .next()
            .unwrap()
            .iter()
            .map(|line| ratings.parse_exact(line).unwrap())
            .collect();

        let part1: u32 = ratings
            .iter()
            .filter(|item| is_accepted(item, &workflows))
            .map(|ratings| ratings.iter().sum::<u32>())
            .sum();

        let part2 = {
            let paths = find_accepted(&workflows);
            paths
                .into_iter()
                .map(|path| {
                    let mut ratings = [(1, 4000); 4];
                    trim_range(&mut ratings, &path);
                    ratings
                        .iter()
                        .map(|(a, b)| (b - a + 1) as usize)
                        .product::<usize>()
                })
                .sum::<usize>()
        };

        Self::solutions(part1, part2)
    }

    fn day20(input: Vec<String>) -> Answers {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
        enum Pulse {
            #[default]
            Low,
            High,
        }

        #[derive(Debug, Clone)]
        enum ModuleState<'a> {
            Broadcast,
            FlipFlop { on: bool },
            Conj { inputs: HashMap<&'a str, Pulse> },
        }

        impl<'a> ModuleState<'a> {
            fn receive(&mut self, from: &'a str, pulse: Pulse) -> Option<Pulse> {
                match self {
                    ModuleState::Broadcast => Some(pulse),
                    ModuleState::FlipFlop { on } if matches!(pulse, Pulse::Low) => {
                        *on = !*on;
                        if *on {
                            Some(Pulse::High)
                        } else {
                            Some(Pulse::Low)
                        }
                    }
                    ModuleState::Conj { inputs } => {
                        *inputs.get_mut(&from).unwrap() = pulse;
                        if inputs.values().all(|&p| p == Pulse::High) {
                            Some(Pulse::Low)
                        } else {
                            Some(Pulse::High)
                        }
                    }
                    _ => None,
                }
            }
        }

        #[derive(Debug, Clone)]
        struct Module<'a> {
            state: ModuleState<'a>,
            outputs: Vec<&'a str>,
        }

        let mut modules: HashMap<&str, Module> = {
            let module_state =
                "broadcaster"
                    .map(|name| (name, ModuleState::Broadcast))
                    .or(parse::any_char.pair(parse::word).map(|(c, name)| {
                        let state = if c == '%' {
                            ModuleState::FlipFlop { on: false }
                        } else {
                            ModuleState::Conj {
                                inputs: HashMap::new(),
                            }
                        };
                        (name, state)
                    }));
            let outputs = parse::word.interspersed(", ");
            let module = module_state.left(" -> ").pair(outputs);

            let mut modules: HashMap<&str, Module> = input
                .iter()
                .map(|line| {
                    let ((name, state), outputs) = module.parse_exact(line).unwrap();
                    (name, Module { state, outputs })
                })
                .collect();

            let inputs = modules
                .iter()
                .flat_map(|(name, module)| {
                    module.outputs.iter().map(move |output| (*output, *name))
                })
                .collect::<Vec<_>>();

            for (to, from) in inputs {
                if let Some(module) = modules.get_mut(to) {
                    if let ModuleState::Conj { inputs } = &mut module.state {
                        inputs.insert(from, Default::default());
                    }
                }
            }

            modules
        };

        fn process<'a>(
            from: &'a str,
            to: &'a str,
            pulse: Pulse,
            modules: &mut HashMap<&str, Module<'a>>,
            pulses: &mut VecDeque<(&'a str, &'a str, Pulse)>,
        ) {
            let Some(module) = modules.get_mut(to) else {
                return;
            };
            let pulse = module.state.receive(from, pulse);
            if let Some(pulse) = pulse {
                let from = to;
                for to in module.outputs.iter() {
                    pulses.push_back((from, to, pulse));
                }
            }
        }

        let mut pulses: VecDeque<(&str, &str, Pulse)> = VecDeque::new();

        let part1 = {
            let (mut num_low, mut num_high) = (0, 0);
            for _ in 0..1000 {
                pulses.push_back(("button", "broadcaster", Pulse::Low));
                while let Some((from, to, pulse)) = pulses.pop_front() {
                    match pulse {
                        Pulse::Low => num_low += 1,
                        Pulse::High => num_high += 1,
                    }
                    process(from, to, pulse, &mut modules, &mut pulses);
                }
            }
            num_low * num_high
        };

        fn chain_period(chain: &[&str], modules: &HashMap<&str, Module>) -> usize {
            chain.iter().enumerate().fold(0, |mut acc, (i, module)| {
                let is_output = modules[module]
                    .outputs
                    .iter()
                    .any(|m| matches!(modules[m].state, ModuleState::Conj { .. }));
                if is_output {
                    acc |= 1 << i;
                }
                acc
            })
        }

        // Looking at the input, it forms four distinct structures
        // where each represents a pulse counter from 1 to a certain number
        // Each counter is made of a chain of flip-flops and a single combinator they
        // are connected to.
        let chains: Vec<Vec<&str>> = modules["broadcaster"]
            .outputs
            .iter()
            .copied()
            .map(|module| {
                std::iter::successors(Some(module), |m| {
                    modules[m]
                        .outputs
                        .iter()
                        .copied()
                        .find(|m| matches!(modules[m].state, ModuleState::FlipFlop { .. }))
                })
                .collect()
            })
            .collect();

        let part2 = chains
            .iter()
            .map(|c| chain_period(c, &modules))
            .product::<usize>();

        Self::solutions(part1, part2)
    }

    fn day21(input: Vec<String>) -> Answers {
        fn distances(start: (usize, usize), map: &Map) -> HashMap<(usize, usize), usize> {
            let mut ret = HashMap::new();
            // Num positions 1 step ago and 2 steps ago
            let mut boundary = HashSet::from([start]);
            let mut step = 0;
            while !boundary.is_empty() {
                for (y, x) in boundary.iter().copied() {
                    ret.insert((y, x), step);
                }
                boundary = boundary
                    .into_iter()
                    .flat_map(|s| {
                        let dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)];
                        let m = &map;
                        let r = &ret;
                        dirs.into_iter().filter_map(move |dir| {
                            m.step(s, dir)
                                .filter(|&p| m[p] != '#' && !r.contains_key(&p))
                        })
                    })
                    .collect();
                step += 1;
            }
            ret
        }

        let map = Map::from(&input);
        let start = map.find('S').unwrap();

        let distances = distances(start, &map);
        let part1 = {
            distances
                .values()
                .copied()
                .filter(|&dist| dist <= 64 && dist % 2 == 0)
                .count()
        };

        let part2 = {
            let n = map.width();
            let even = distances.values().filter(|&&dist| dist % 2 == 0).count();
            let odd = distances.values().filter(|&&dist| dist % 2 == 1).count();

            let even_corners = distances
                .values()
                .filter(|&&dist| dist > n / 2 && dist % 2 == 0)
                .count();
            let odd_corners = distances
                .values()
                .filter(|&&dist| dist > n / 2 && dist % 2 == 1)
                .count();

            let m = 26501365 / n;
            assert_eq!(m, 202300);
            let e = m * m;
            let o = (m + 1) * (m + 1);

            o * odd + e * even - ((m + 1) * odd_corners) + (m * even_corners)
        };
        Self::solutions(part1, part2)
    }

    fn day22(input: Vec<String>) -> Answers {
        type Coord = (u32, u32, u32);
        type Brick = (Coord, Coord);
        type BrickId = usize;
        let bricks: Vec<Brick> = {
            fn coord(s: &str) -> parse::ParseResult<Coord> {
                parse::uint::<u32>
                    .left(',')
                    .pair(parse::uint::<u32>)
                    .left(',')
                    .pair(parse::uint::<u32>)
                    .map(|((x, y), z)| (x, y, z))
                    .pars(s)
            }

            let brick = coord.left('~').pair(coord);
            let mut bricks: Vec<(Coord, Coord)> = input
                .iter()
                .map(|line| brick.parse_exact(line).unwrap())
                .collect();
            bricks.sort_by_key(|((_, _, z0), _)| *z0);
            bricks
        };

        #[derive(Debug, Clone, Copy)]
        struct Cell {
            topmost: BrickId,
            height: u32,
        }

        /// Returns (height, supports)
        fn find_supporting_bricks(
            ((x0, y0, _), (x1, y1, _)): Brick,
            cells: &HashMap<(u32, u32), Cell>,
        ) -> (u32, HashSet<BrickId>) {
            let mut base = 0;
            let mut supporting = HashSet::new();
            for x in x0..=x1 {
                for y in y0..=y1 {
                    if let Some(Cell { topmost, height }) = cells.get(&(x, y)).copied() {
                        if height > base {
                            supporting.clear();
                            base = height;
                        }
                        if height == base {
                            supporting.insert(topmost);
                        }
                    }
                }
            }
            (base, supporting)
        }

        let mut cells: HashMap<(u32, u32), Cell> = HashMap::new();
        let mut supporting: Vec<HashSet<BrickId>> = vec![HashSet::new(); bricks.len()];
        let mut num_supports: Vec<usize> = Vec::new();
        let mut heights: Vec<u32> = Vec::new();
        for (i, brick @ ((x0, y0, z0), (x1, y1, z1))) in bricks.iter().copied().enumerate() {
            let height = z1 - z0 + 1;
            let (base, supp) = find_supporting_bricks(brick, &cells);
            for x in x0..=x1 {
                for y in y0..=y1 {
                    cells.insert(
                        (x, y),
                        Cell {
                            topmost: i,
                            height: base + height,
                        },
                    );
                }
            }
            for support in supp.iter().copied() {
                supporting[support].insert(i);
            }
            num_supports.push(supp.len());
            heights.push(base);
        }

        let part1 = supporting
            .iter()
            .enumerate()
            .filter(|(_brick, supported)| supported.iter().all(|&b| num_supports[b] > 1))
            .count();

        fn count_falling_if_removed(
            id: BrickId,
            heights: &[u32],
            supporting: &[HashSet<BrickId>],
            num_supports: &[usize],
        ) -> usize {
            #[derive(PartialEq, Eq)]
            struct Brick {
                id: BrickId,
                height: u32,
            }

            impl Ord for Brick {
                fn cmp(&self, other: &Self) -> Ordering {
                    // Reverse order for min heap
                    other.height.cmp(&self.height)
                }
            }
            impl PartialOrd for Brick {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut num_supports = Vec::from(num_supports);
            let mut falling: HashSet<usize> = HashSet::from([id]);
            let mut worklist = BinaryHeap::new();
            worklist.push(Brick {
                id,
                height: heights[id],
            });

            while let Some(Brick { id, .. }) = worklist.pop() {
                for supported in supporting[id].iter().copied() {
                    if falling.contains(&supported) {
                        continue;
                    }
                    num_supports[supported] -= 1;
                    if num_supports[supported] == 0 {
                        falling.insert(supported);
                        worklist.push(Brick {
                            id: supported,
                            height: heights[supported],
                        });
                    }
                }
            }
            falling.len() - 1
        }

        let part2 = (0..heights.len())
            .map(|brick| count_falling_if_removed(brick, &heights, &supporting, &num_supports))
            .sum::<usize>();

        Self::solutions(part1, part2)
    }

    fn day23(input: Vec<String>) -> Answers {
        type NodeId = usize;
        #[derive(Default, Debug)]
        struct Graph {
            nodes: HashMap<(usize, usize), NodeId>,
            adj: Vec<HashMap<NodeId, usize>>,
        }

        fn neighbors(map: &Map, pos: (usize, usize)) -> impl '_ + Iterator<Item = (usize, usize)> {
            let dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)];
            dirs.into_iter()
                .filter_map(move |dir| map.step_if(pos, dir, |c| c != '#'))
        }

        fn connect(map: &Map, start: (usize, usize), graph: &mut Graph) {
            let mut worklist = Vec::from([(start, 0)]);
            let start = graph.nodes.get(&start).copied().unwrap();
            let mut visited = HashSet::new();
            while let Some((pos, len)) = worklist.pop() {
                if !visited.insert(pos) {
                    continue;
                }

                if let Some(other) = graph.nodes.get(&pos).copied() {
                    if other != start {
                        graph.adj[start].insert(other, len);
                        continue;
                    }
                }

                for next in neighbors(map, pos) {
                    worklist.push((next, len + 1));
                }
            }
        }

        fn longest_path(
            start: NodeId,
            end: NodeId,
            graph: &Graph,
            visited: &mut HashSet<NodeId>,
        ) -> Option<usize> {
            if start == end {
                return Some(0);
            }
            if !visited.insert(start) {
                return None;
            }
            let max = graph.adj[start]
                .iter()
                .filter_map(|(&neighbor, len)| {
                    longest_path(neighbor, end, graph, visited).map(|x| x + len)
                })
                .max();
            visited.remove(&start);
            max
        }

        let map = Map::from(&input);
        let start = (0, map[0].iter().position(|&c| c == '.').unwrap());
        let end = (
            map.height() - 1,
            map[map.height() - 1]
                .iter()
                .position(|&c| c == '.')
                .unwrap(),
        );

        let cells = (0..map.height()).flat_map(|row| (0..map.width()).map(move |col| (row, col)));
        let nodes = cells
            .filter(|&pos| map[pos] != '#' && neighbors(&map, pos).count() != 2)
            .zip(0usize..)
            .collect::<HashMap<_, _>>();
        let mut graph = Graph {
            nodes: nodes.clone(),
            adj: vec![HashMap::new(); nodes.len()],
        };

        let start = nodes[&start];
        let end = nodes[&end];

        for node in nodes.into_keys() {
            connect(&map, node, &mut graph);
        }

        let mut visited = HashSet::new();
        let part2 = longest_path(start, end, &graph, &mut visited).unwrap();
        Self::solutions(0, part2)
    }

    fn day24(input: Vec<String>) -> Answers {
        type Coord = (isize, isize, isize);
        type Coordf = (f64, f64, f64);
        type Trajectory = (Coord, Coord);
        fn intersection(a: Trajectory, b: Trajectory) -> Option<Coordf> {
            let (x1, y1, _) = a.0;
            let (x2, y2) = (x1 + a.1 .0, y1 + a.1 .1);
            let (x3, y3, _) = b.0;
            let (x4, y4) = (x3 + b.1 .0, y3 + b.1 .1);
            let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if denom == 0 {
                return None;
            }

            let t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
            let t = t as f64 / denom as f64;
            let u = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);
            let u = u as f64 / denom as f64;
            if t < 0.0 || u < 0.0 {
                return None;
            }
            let px = x1 as f64 + (t * (x2 - x1) as f64);
            let py = y1 as f64 + (t * (y2 - y1) as f64);
            Some(dbg!((px, py, 0.0)))
        }

        let stones: Vec<Trajectory> = {
            fn coord(s: &str) -> parse::ParseResult<Coord> {
                parse::int::<isize>
                    .left(','.left(parse::whitespace))
                    .pair(parse::int::<isize>)
                    .left(','.left(parse::whitespace))
                    .pair(parse::int::<isize>)
                    .map(|((a, b), c)| (a, b, c))
                    .pars(s)
            }
            input
                .iter()
                .map(|line| {
                    coord
                        .left(parse::trimmed('@'))
                        .pair(coord)
                        .parse_exact(line)
                        .unwrap()
                })
                .collect()
        };

        let combinations = (0..stones.len()).flat_map(|i| {
            let s = &stones;
            (i + 1..stones.len()).map(move |j| (s[i], s[j]))
        });
        let min = 200000000000000;
        let max = 400000000000000;
        let range = min..=max;
        let part1 = combinations
            .filter_map(|(a, b)| intersection(a, b))
            .map(|(x, y, _)| (x as isize, y as isize))
            .filter(|(x, y)| range.contains(x) && range.contains(y))
            .count();

        Self::part1(part1)
    }

    fn day25(input: Vec<String>) -> Answers {
        type Graph<'a> = HashMap<&'a str, HashSet<&'a str>>;

        fn count_connected(graph: &Graph, v: &str) -> usize {
            let mut visited = HashSet::new();
            let mut worklist = VecDeque::from([v]);
            while let Some(node) = worklist.pop_front() {
                visited.insert(node);
                worklist.extend(graph[node].difference(&visited));
            }
            visited.len()
        }

        let mut graph: Graph = HashMap::new();
        for line in input.iter() {
            let (from, tos) = parse::word
                .left(": ")
                .pair(parse::word.interspersed(parse::whitespace))
                .parse_exact(&line)
                .unwrap();
            for to in tos {
                graph.entry(from).or_default().insert(to);
                graph.entry(to).or_default().insert(from);
            }
        }

        let mut remove_edge = |a, b| {
            graph.get_mut(a).unwrap().remove(b);
            graph.get_mut(b).unwrap().remove(a);
        };

        // Determined from graphviz
        remove_edge("lmg", "krx");
        remove_edge("vzb", "tnr");
        remove_edge("tvf", "tqn");
        let a = count_connected(&graph, "lmg");
        let b = count_connected(&graph, "krx");
        Self::part1(a * b)
    }
}
