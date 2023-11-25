use crate::solutions::{Answers, Solutions};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::iter;
use std::ops::RangeInclusive;
use std::str::FromStr;

pub struct Solutions2021;

impl Solutions for Solutions2021 {
    fn day11(input: Vec<String>) -> Answers {
        fn step(input: &mut Vec<Vec<u32>>) -> usize {
            fn flash(input: &mut Vec<Vec<u32>>, row: usize, col: usize) -> usize {
                dbg!((row, col));
                let min_row = row.saturating_sub(1);
                let max_row = (row + 2).min(input.len());
                let min_col = col.saturating_sub(1);
                let max_col = (col + 2).min(input[row].len());
                let mut flashes = 1;
                for row in min_row..max_row {
                    for col in min_col..max_col {
                        let val = &mut input[row][col];
                        if *val < 10 {
                            *val += 1;
                            if *val == 10 {
                                flashes += flash(input, row, col);
                            }
                        }
                    }
                }
                flashes
            }

            for line in input.iter_mut() {
                for x in line.iter_mut() {
                    *x += 1;
                }
            }

            let mut flashes = 0;
            for row in 0..input.len() {
                for col in 0..input[row].len() {
                    if input[row][col] == 10 {
                        flashes += flash(input, row, col);
                    }
                }
            }

            for line in input.iter_mut() {
                for x in line.iter_mut() {
                    if *x == 10 {
                        *x = 0;
                    }
                }
            }
            flashes
        }

        fn fmt_print(input: &Vec<Vec<u32>>) -> String {
            let mut result = String::with_capacity(input.len() * (input[0].len() + 1));
            for line in input {
                for x in line {
                    result.push(char::from_digit(*x, 10).unwrap());
                }
                result.push('\n');
            }
            result
        }

        let mut input: Vec<Vec<u32>> = input
            .iter()
            .map(|line| line.chars().map(|c| c.to_digit(10).unwrap()).collect())
            .collect();

        let num_flashes: usize = (0..100)
            .map(|s| {
                println!("After {} steps:\n{}", s, fmt_print(&input));
                dbg!(step(&mut input))
            })
            .sum();
        Self::part1(num_flashes)
    }

    fn day12(input: Vec<String>) -> Answers {
        type EdgeMap<'a> = HashMap<&'a str, Vec<&'a str>>;

        fn visit<'a>(current: &'a str, edges: &EdgeMap, mut visited: HashSet<&'a str>) -> usize {
            if current == "end" {
                return 1;
            }
            if visited.contains(current) {
                return 0;
            }
            if current.chars().all(char::is_lowercase) {
                visited.insert(current);
            }
            edges
                .get(current)
                .unwrap()
                .iter()
                .map(|neighbor| visit(neighbor, edges, visited.clone()))
                .sum()
        }

        fn visit2<'a>(
            current: &'a str,
            edges: &EdgeMap,
            mut visited: HashSet<&'a str>,
            mut visited_twice: bool,
        ) -> usize {
            if current == "end" {
                return 1;
            }
            if visited.contains(current) {
                if visited_twice || current == "start" {
                    return 0;
                }
                visited_twice = true;
            }
            if current.chars().all(char::is_lowercase) {
                visited.insert(current);
            }
            edges
                .get(current)
                .unwrap()
                .iter()
                .map(|neighbor| visit2(neighbor, edges, visited.clone(), visited_twice))
                .sum()
        }

        let mut edges: EdgeMap = HashMap::new();
        for line in input.iter() {
            let mut parts = line.split('-');
            let from = parts.next().unwrap();
            let to = parts.next().unwrap();
            edges.entry(from).or_default().push(to);
            edges.entry(to).or_default().push(from);
        }

        let part1 = visit("start", &edges, Default::default()).to_string();
        let part2 = visit2("start", &edges, Default::default(), false).to_string();
        (part1, Some(part2))
    }

    fn day13(input: Vec<String>) -> Answers {
        fn fold_x(points: HashSet<(u32, u32)>, position: u32) -> HashSet<(u32, u32)> {
            points
                .into_iter()
                .map(|(x, y)| {
                    let x = if x > position {
                        position - (x - position)
                    } else {
                        x
                    };
                    (x, y)
                })
                .collect()
        }
        fn fold_y(points: HashSet<(u32, u32)>, position: u32) -> HashSet<(u32, u32)> {
            points
                .into_iter()
                .map(|(x, y)| {
                    let y = if y > position {
                        position - (y - position)
                    } else {
                        y
                    };
                    (x, y)
                })
                .collect()
        }

        fn fold(instruction: &str, points: HashSet<(u32, u32)>) -> HashSet<(u32, u32)> {
            let mut parts = instruction.split('=');
            let instruction = parts.next().unwrap();
            let position = parts.next().unwrap().parse::<u32>().unwrap();
            match instruction.chars().last().unwrap() {
                'x' => fold_x(points, position),
                'y' => fold_y(points, position),
                _ => unreachable!(),
            }
        }

        fn draw(points: &HashSet<(u32, u32)>) -> String {
            let mut lines = Vec::new();
            for (x, y) in points.iter() {
                let (x, y) = (*x as usize, *y as usize);
                while y >= lines.len() {
                    lines.push(Vec::new());
                }
                let line = &mut lines[y];
                while x >= line.len() {
                    line.push(' ');
                }
                line[x] = '#';
            }

            iter::once('\n')
                .chain(
                    lines
                        .into_iter()
                        .flat_map(|line| line.into_iter().chain(iter::once('\n'))),
                )
                .collect()
        }

        let num_points = input.iter().position(|s| s.is_empty()).unwrap();
        let mut points = input[..num_points]
            .iter()
            .map(|s| {
                let mut parts = s.split(',');
                (
                    parts.next().unwrap().parse::<u32>().unwrap(),
                    parts.next().unwrap().parse::<u32>().unwrap(),
                )
            })
            .collect::<HashSet<_>>();

        points = fold(&input[num_points + 1], points);
        let answer1 = points.len();

        for instruction in &input[num_points + 2..] {
            points = fold(instruction, points);
        }
        let answer2 = draw(&points);

        Self::solutions(answer1, answer2)
    }

    fn day14(input: Vec<String>) -> Answers {
        type PatternMap = HashMap<(char, char), char>;
        type PairCounter = HashMap<(char, char), usize>;

        fn step(pairs: PairCounter, patterns: &PatternMap) -> PairCounter {
            let mut new_pairs = HashMap::new();
            for (&(a, b), &count) in pairs.iter() {
                if let Some(&insertion) = patterns.get(&(a, b)) {
                    *new_pairs.entry((a, insertion)).or_insert(0) += count;
                    *new_pairs.entry((insertion, b)).or_insert(0) += count;
                } else {
                    new_pairs.insert((a, b), count);
                }
            }
            new_pairs
        }

        fn count_pairs(s: &str) -> PairCounter {
            let mut counter = HashMap::new();
            for pair in s.chars().zip(s.chars().skip(1).chain(iter::once('\n'))) {
                *counter.entry(pair).or_insert(0) += 1;
            }
            counter
        }

        fn compute_score(pairs: &PairCounter) -> usize {
            let mut char_occurrences = HashMap::new();
            for ((a, _), count) in pairs.iter() {
                *char_occurrences.entry(a).or_insert(0) += count;
            }
            char_occurrences.values().max().unwrap() - char_occurrences.values().min().unwrap()
        }

        let mut pairs = count_pairs(&input[0]);

        let patterns = input[2..]
            .iter()
            .map(|line| {
                let mut parts = line.split(" -> ");
                let pattern = parts.next().unwrap();
                let insertion = parts.next().unwrap().chars().next().unwrap();
                let mut pattern = pattern.chars();
                let pattern = (pattern.next().unwrap(), pattern.next().unwrap());
                (pattern, insertion)
            })
            .collect::<PatternMap>();

        for _ in 0..10 {
            pairs = step(pairs, &patterns);
        }
        let answer = compute_score(&pairs);

        for _ in 10..40 {
            pairs = step(pairs, &patterns);
        }
        let answer2 = compute_score(&pairs);

        (answer.to_string(), Some(answer2.to_string()))
    }

    fn day15(input: Vec<String>) -> Answers {
        fn answer(input: &[String], reps: u32) -> u32 {
            let mut risk_lines = iter::repeat(input.iter())
                .zip(0u32..)
                .take(reps as usize)
                .flat_map(|(lines, offset)| {
                    lines.map(move |line| {
                        iter::repeat(line.chars().flat_map(|c| c.to_digit(10)))
                            .zip(0u32..)
                            .take(reps as usize)
                            .flat_map(move |(iterator, tile)| {
                                iterator.map(move |x| ((x + offset + tile - 1) % 9) + 1)
                            })
                    })
                });

            for c in risk_lines.clone().next().unwrap() {
                print!("{}", c);
            }
            println!("");

            let first_line = risk_lines.next().unwrap();
            let first_line = iter::once(0).chain(first_line.skip(1));
            // accumulate risks
            let mut risks: Vec<u32> = first_line
                .scan(0, |state, x| {
                    debug_assert!(x < 10);
                    *state += x;
                    Some(*state)
                })
                .collect();

            for mut new_risks in risk_lines {
                risks[0] += new_risks.next().unwrap();
                for i in 1..risks.len() {
                    risks[i] = risks[i].min(risks[i - 1]) + new_risks.next().unwrap();
                }
            }

            *risks.iter().last().unwrap()
        }

        Self::solutions(answer(&input, 1), answer(&input, 5))
    }

    fn day16(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        enum Packet {
            Literal {
                version: usize,
                value: u64,
            },
            Operator {
                version: usize,
                typeid: u8,
                subpackets: Vec<Packet>,
            },
        }

        fn parse_literal(mut line: &str, version: usize) -> (Packet, &str) {
            let mut value = 0u64;
            let mut last = false;
            while !last {
                last = line.chars().next().unwrap() == '0';
                let number = u64::from_str_radix(&line[1..5], 2).unwrap();
                line = &line[5..];
                value = value << 4 | number;
            }
            (Packet::Literal { version, value }, line)
        }

        fn parse_operator(mut line: &str, version: usize, typeid: u8) -> (Packet, &str) {
            let length_is_literal = line.chars().next().unwrap() == '0';
            line = &line[1..];
            let mut subpackets = Vec::new();
            let rest = if length_is_literal {
                let length_in_bits = usize::from_str_radix(&line[..15], 2).unwrap();
                let line = &line[15..];

                let rest = &line[length_in_bits..];
                let mut to_parse = &line[..length_in_bits];
                while !to_parse.is_empty() {
                    let (packet, rest_to_parse) = parse(to_parse);
                    subpackets.push(packet);
                    to_parse = rest_to_parse;
                }
                rest
            } else {
                let length_in_packets = usize::from_str_radix(&line[..11], 2).unwrap();
                let mut to_parse = &line[11..];
                for _ in 0..length_in_packets {
                    let (packet, rest_to_parse) = parse(to_parse);
                    subpackets.push(packet);
                    to_parse = rest_to_parse;
                }
                to_parse
            };
            (
                Packet::Operator {
                    version,
                    typeid,
                    subpackets,
                },
                rest,
            )
        }

        fn parse(line: &str) -> (Packet, &str) {
            let version = usize::from_str_radix(&line[0..3], 2).unwrap();
            let typeid = u8::from_str_radix(&line[3..6], 2).unwrap();
            let to_parse = &line[6..];
            match typeid {
                4 => parse_literal(to_parse, version),
                _ => parse_operator(to_parse, version, typeid),
            }
        }

        fn add_version_numbers(packet: &Packet) -> usize {
            match packet {
                Packet::Literal { version, .. } => *version,
                Packet::Operator {
                    version,
                    subpackets,
                    ..
                } => *version + subpackets.iter().map(add_version_numbers).sum::<usize>(),
            }
        }

        fn eval(packet: &Packet) -> u64 {
            match packet {
                Packet::Literal { value, .. } => *value,
                Packet::Operator {
                    typeid, subpackets, ..
                } => match typeid {
                    0 => subpackets.iter().map(eval).sum(),
                    1 => subpackets.iter().map(eval).product(),
                    2 => subpackets.iter().map(eval).min().unwrap(),
                    3 => subpackets.iter().map(eval).max().unwrap(),
                    other => {
                        debug_assert!(subpackets.len() == 2);
                        let a = eval(&subpackets[0]);
                        let b = eval(&subpackets[1]);
                        let condition = match other {
                            5 => a > b,
                            6 => a < b,
                            7 => a == b,
                            _ => unreachable!(),
                        };
                        if condition {
                            1
                        } else {
                            0
                        }
                    }
                },
            }
        }

        // Convert line to binary string for easier bitwise scanning
        let line = &input[0];
        let mut packet = String::with_capacity(line.len() * 4);
        for c in line.chars() {
            let digit = c.to_digit(16).unwrap();
            write!(packet, "{:04b}", digit).unwrap();
        }
        let (packet, _) = parse(&packet);
        Self::solutions(add_version_numbers(&packet), eval(&packet))
    }

    fn day17(input: Vec<String>) -> Answers {
        fn is_target_reachable(
            pos: (isize, isize),
            velocity: (isize, isize),
            x_bounds: &RangeInclusive<isize>,
            y_bounds: &RangeInclusive<isize>,
        ) -> bool {
            !((  velocity.0 == 0 && !x_bounds.contains(&pos.0)) // over/under-shot
             || (velocity.0 > 0 && pos.0 > *x_bounds.end()) // overshot 
             || (velocity.1 <= 0 && pos.1 < *y_bounds.start())) // miss
        }

        fn shoot(
            mut velocity: (isize, isize),
            x_bounds: &RangeInclusive<isize>,
            y_bounds: &RangeInclusive<isize>,
        ) -> Option<isize> {
            let mut pos = (0, 0);
            let mut max_y = pos.1;
            while is_target_reachable(pos, velocity, x_bounds, y_bounds) {
                pos.0 += velocity.0;
                pos.1 += velocity.1;

                max_y = max_y.max(pos.1);
                if x_bounds.contains(&pos.0) && y_bounds.contains(&pos.1) {
                    // target hit!
                    return Some(max_y);
                }

                velocity.0 -= velocity.0.signum(); // drag
                velocity.1 -= 1; // gravity
            }
            None
        }

        let input = input.into_iter().next().unwrap();
        let mut parts = input.split(": ");
        let area = parts.nth(1).unwrap();
        let mut parts = area.split(", ").map(|dim| {
            let dim = &dim[2..]; // skip x=
            let mut bounds = dim.split("..").map(|s| s.parse::<isize>().unwrap());
            bounds.next().unwrap()..=bounds.next().unwrap()
        });

        let x_bounds = parts.next().unwrap();
        let y_bounds = parts.next().unwrap();

        let velocities = (1..1000).flat_map(|x| (-1000..1000).map(move |y| (x, y)));
        let hitting_shots = velocities
            .flat_map(|velocity| shoot(velocity, &x_bounds, &y_bounds))
            .collect::<Vec<_>>();

        let max_y = hitting_shots.iter().max().unwrap();
        let num_hits = hitting_shots.len();

        Self::solutions(max_y, num_hits)
    }

    fn day19(input: Vec<String>) -> Answers {
        fn diff(
            (x1, y1, z1): (isize, isize, isize),
            (x2, y2, z2): (isize, isize, isize),
        ) -> (isize, isize, isize) {
            (x1 - x2, y1 - y2, z1 - z2)
        }

        let mut scanner_reports = input.split(|s| s.is_empty()).map(|lines| {
            let lines = &lines[1..];
            lines
                .iter()
                .map(|line| {
                    let mut parts = line.split(',').map(|s| s.parse::<isize>().unwrap());
                    (
                        parts.next().unwrap(),
                        parts.next().unwrap(),
                        parts.next().unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        });

        let reference_report = scanner_reports.next().unwrap();
        let reference_distances = reference_report.iter().map(|p1| {
            reference_report
                .iter()
                .map(move |p2| diff(*p1, *p2))
                .collect::<HashSet<_>>()
        });

        for dist in reference_distances {
            dbg!(dist);
        }
        Self::part1("")
    }

    fn day20(input: Vec<String>) -> Answers {
        fn _enhance(
            image: HashSet<(isize, isize)>,
            _algorithm: &[bool],
        ) -> HashSet<(isize, isize)> {
            let mut alg_indices = HashMap::new();
            for (x, y) in image.into_iter() {
                *alg_indices.entry((x - 1, y - 1)).or_insert(0) |= 1 << 0;
                *alg_indices.entry((x - 0, y - 1)).or_insert(0) |= 1 << 1;
                *alg_indices.entry((x + 0, y - 1)).or_insert(0) |= 1 << 2;
            }
            HashSet::new()
        }

        let mut input = input.into_iter();
        let algorithm = input
            .next()
            .unwrap()
            .chars()
            .map(|c| c == '#')
            .collect::<Vec<_>>();
        input.next(); // skip empty line

        let _image = input
            .zip(0isize..)
            .flat_map(|(line, y)| {
                line.chars()
                    .zip(0isize..)
                    .filter_map(move |(c, x)| if c == '#' { Some((x, y)) } else { None })
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>();

        dbg!(algorithm);
        Self::part1("")
    }

    fn day21(input: Vec<String>) -> Answers {
        #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
        struct Player {
            position: usize,
            score: usize,
        }

        impl Player {
            fn move_by(&self, num_spaces: usize) -> Player {
                let position = ((self.position - 1 + num_spaces) % 10) + 1;
                let score = self.score + position;
                Player { position, score }
            }
        }

        fn roll(dice: &mut usize) -> usize {
            let result = *dice;
            *dice = result % 100 + 1;
            result
        }

        let mut players = input.into_iter().map(|line| {
            let position = line.split(' ').last().unwrap().parse::<usize>().unwrap();
            Player { position, score: 0 }
        });

        let players = (players.next().unwrap(), players.next().unwrap());

        let answer1 = {
            let mut players = players;
            let mut dice = 1;
            let mut num_rolls = 0;
            loop {
                let num_spaces = roll(&mut dice) + roll(&mut dice) + roll(&mut dice);
                num_rolls += 3;
                players.0 = players.0.move_by(num_spaces);
                if players.0.score >= 1000 {
                    break players.1.score * num_rolls;
                }
                players = (players.1, players.0);
            }
        };

        fn possible_dice_rolls() -> impl Iterator<Item = usize> {
            (1..=3).flat_map(|roll1| {
                (1..=3).flat_map(move |roll2| (1..=3).map(move |roll3| roll1 + roll2 + roll3))
            })
        }

        fn winning_universes(
            player1: Player,
            player2: Player,
            dp: &mut HashMap<(Player, Player), (usize, usize)>,
        ) -> (usize, usize) {
            // player1: player whose next turn it is
            if let Some(result) = dp.get(&(player1, player2)).copied() {
                return result;
            }

            let result = possible_dice_rolls()
                .map(|dice_roll| {
                    let player1 = player1.move_by(dice_roll);
                    if player1.score >= 21 {
                        (1, 0)
                    } else {
                        let (r2, r1) = winning_universes(player2, player1, dp);
                        (r1, r2)
                    }
                })
                .reduce(|a, b| (a.0 + b.0, a.1 + b.1))
                .unwrap();

            dp.insert((player1, player2), result);
            result
        }

        let mut dp = HashMap::new();
        let universes = winning_universes(players.0, players.1, &mut dp);
        let answer2 = universes.0.max(universes.1);
        Self::solutions(answer1, answer2)
    }

    fn day22(input: Vec<String>) -> Answers {
        #[derive(Debug, Default)]
        struct Cuboid {
            x1: isize,
            x2: isize,
            y1: isize,
            y2: isize,
            z1: isize,
            z2: isize,
            on: bool,
            removed: Vec<Cuboid>,
        }

        impl FromStr for Cuboid {
            type Err = ();
            fn from_str(s: &str) -> Result<Cuboid, Self::Err> {
                let mut parts = s.split(' ');
                let on = parts.next().unwrap() == "on";
                let mut parts = parts.next().unwrap().split(',').map(|input| {
                    let input = &input[2..]; //skip x=
                    let mut bounds = input.split("..").map(|s| s.parse::<isize>().unwrap());
                    (bounds.next().unwrap(), bounds.next().unwrap())
                });
                let (x1, x2) = parts.next().unwrap();
                let (y1, y2) = parts.next().unwrap();
                let (z1, z2) = parts.next().unwrap();
                Ok(Cuboid {
                    x1,
                    x2,
                    y1,
                    y2,
                    z1,
                    z2,
                    on,
                    ..Default::default()
                })
            }
        }

        impl Cuboid {
            fn intersection(&self, other: &Cuboid) -> Option<Cuboid> {
                if self.x1 > other.x2
                    || self.x2 < other.x1
                    || self.y1 > other.y2
                    || self.y2 < other.y1
                    || self.z1 > other.z2
                    || self.z2 < other.z1
                {
                    None
                } else {
                    Some(Cuboid {
                        x1: self.x1.max(other.x1),
                        x2: self.x2.min(other.x2),
                        y1: self.y1.max(other.y1),
                        y2: self.y2.min(other.y2),
                        z1: self.z1.max(other.z1),
                        z2: self.z2.min(other.z2),
                        on: self.on,
                        ..Default::default()
                    })
                }
            }

            fn remove(&mut self, other: &Cuboid) {
                if let Some(isect) = self.intersection(other) {
                    for removed in &mut self.removed {
                        removed.remove(&isect);
                    }
                    self.removed.push(isect)
                }
            }

            fn size(&self) -> usize {
                let len_x = (self.x2 - self.x1) as usize + 1;
                let len_y = (self.y2 - self.y1) as usize + 1;
                let len_z = (self.z2 - self.z1) as usize + 1;
                len_x * len_y * len_z - self.removed.iter().map(Self::size).sum::<usize>()
            }
        }

        fn compute_answer<T: Iterator<Item = Cuboid>>(it: T) -> usize {
            let mut on_cuboids: Vec<Cuboid> = Vec::new();
            for cuboid in it {
                for on in on_cuboids.iter_mut() {
                    on.remove(&cuboid);
                }
                if cuboid.on {
                    on_cuboids.push(cuboid);
                }
            }
            on_cuboids.iter().map(Cuboid::size).sum()
        }

        let cuboids = input
            .into_iter()
            .map(|line| line.parse::<Cuboid>().unwrap());

        let considered_area = Cuboid {
            x1: -50,
            x2: 50,
            y1: -50,
            y2: 50,
            z1: -50,
            z2: 50,
            ..Default::default()
        };

        let answer1 = compute_answer(
            cuboids
                .clone()
                .filter_map(|cuboid| cuboid.intersection(&considered_area)),
        );

        let answer2 = compute_answer(cuboids);
        Self::solutions(answer1, answer2)
    }

    fn day24(input: Vec<String>) -> Answers {
        #[derive(Debug, Default)]
        struct Registers {
            w: i64,
            x: i64,
            y: i64,
            z: i64,
        }

        impl Registers {
            fn get_mut(&mut self, name: char) -> &mut i64 {
                match name {
                    'w' => &mut self.w,
                    'x' => &mut self.x,
                    'y' => &mut self.y,
                    'z' => &mut self.z,
                    _ => unreachable!(),
                }
            }
        }

        fn run_instr(instr: &str, regs: &mut Registers, inputs: &mut impl Iterator<Item = i64>) {
            let mut parts = instr.split(' ');
            let op = parts.next().unwrap();
            let reg_name = parts.next().unwrap().chars().next().unwrap();
            let operand = parts
                .map(|s| {
                    if s.starts_with(char::is_alphabetic) {
                        *regs.get_mut(s.chars().next().unwrap())
                    } else {
                        s.parse::<i64>().unwrap()
                    }
                })
                .next();

            let reg = regs.get_mut(reg_name);

            match op {
                "inp" => {
                    *reg = inputs.next().unwrap();
                }
                "add" => {
                    *reg += operand.unwrap();
                }
                "mul" => {
                    *reg *= operand.unwrap();
                }
                "div" => {
                    *reg /= operand.unwrap();
                }
                "mod" => {
                    *reg %= operand.unwrap();
                }
                "eql" => {
                    *reg = if *reg == operand.unwrap() { 1 } else { 0 };
                }
                _ => {}
            }
        }

        fn digits(mut num: i64) -> impl Iterator<Item = i64> {
            let mut divisor = 1;
            while num >= divisor * 10 {
                divisor *= 10;
            }

            std::iter::from_fn(move || {
                if divisor == 0 {
                    None
                } else {
                    let v = num / divisor;
                    num %= divisor;
                    divisor /= 10;
                    Some(v)
                }
            })
        }

        fn run(program: &[String], input: i64) -> Registers {
            let mut regs = Registers::default();
            let mut digits = digits(dbg!(input));
            for line in program {
                run_instr(line, &mut regs, &mut digits);
            }
            regs
        }

        let mut correct_inputs = (11111111111111..=99999999999999).rev().filter_map(|num| {
            let regs = run(&input, num);
            if regs.z == 0 {
                Some(num)
            } else {
                None
            }
        });

        Self::part1(correct_inputs.next().unwrap())
    }

    fn day25(input: Vec<String>) -> Answers {
        fn step_east(region: &mut Vec<Vec<char>>) -> bool {
            let mut moved = false;
            for line in region.iter_mut() {
                let first_spot = line[0];
                let mut col = 0;
                while col < line.len() {
                    let next_col = (col + 1) % line.len();
                    let spots = if next_col == 0 {
                        (line[col], first_spot)
                    } else {
                        (line[col], line[next_col])
                    };
                    if let ('>', '.') = spots {
                        moved = true;
                        line.swap(col, next_col);
                        col += 1;
                    }
                    col += 1;
                }
            }
            moved
        }

        fn step_south(region: &mut Vec<Vec<char>>) -> bool {
            let num_cols = region[0].len();
            let mut moved = false;
            for col in 0..num_cols {
                let first_spot = region[0][col];
                let mut row = 0;
                while row < region.len() {
                    let spots = if row + 1 == region.len() {
                        (region[row][col], first_spot)
                    } else {
                        (region[row][col], region[row + 1][col])
                    };
                    if let ('v', '.') = spots {
                        let (line, next_line) = if row + 1 == region.len() {
                            let (last, range) = region.split_last_mut().unwrap();
                            (last, range.first_mut().unwrap())
                        } else {
                            let (up, down) = region.split_at_mut(row + 1);
                            (up.last_mut().unwrap(), down.first_mut().unwrap())
                        };
                        std::mem::swap(&mut line[col], &mut next_line[col]);
                        moved = true;
                        row += 1;
                    }
                    row += 1;
                }
            }
            moved
        }

        fn step(region: &mut Vec<Vec<char>>) -> bool {
            step_east(region) | step_south(region)
        }

        fn _fmt_region(region: &Vec<Vec<char>>) -> String {
            let mut result = String::with_capacity(region.len() * (region[0].len() + 1));
            for line in region {
                for &c in line {
                    result.push(c);
                }
                result.push('\n');
            }
            result
        }

        let mut region = input
            .iter()
            .map(|s| s.chars().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let step = (1..).skip_while(|_| step(&mut region)).next().unwrap();

        Self::part1(step)
    }
}
