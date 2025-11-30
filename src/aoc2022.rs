use aoc::parse::{self, Parser};
use aoc::solutions::Answers;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

pub struct Solutions;

impl aoc::solutions::Solutions for Solutions {
    fn day1(input: Vec<String>) -> Answers {
        let groups = input.split(String::is_empty);
        let total_calories =
            groups.map(|lines| lines.iter().map(|line| line.parse::<i32>().unwrap()).sum());

        let mut sorted = total_calories.collect::<Vec<_>>();
        sorted.sort_unstable();
        sorted.reverse(); // Descending order

        let max: i32 = sorted[0];
        let top_three_sum: i32 = sorted[0..3].iter().sum();
        Self::solutions(max, top_three_sum)
    }

    fn day2(input: Vec<String>) -> Answers {
        #[derive(PartialEq, Eq, Copy, Clone)]
        struct Rps(u32);
        // 0 == rock, 1 == paper, 2 == scissors

        impl Rps {
            fn score(&self) -> u32 {
                self.0 + 1
            }

            fn score_against(&self, other: &Rps) -> u32 {
                if &Self::winning_against(other) == self {
                    6
                } else if other == self {
                    3
                } else {
                    0
                }
            }

            fn parse(line: &str) -> Rps {
                match line {
                    "A" | "X" => Rps(0),
                    "B" | "Y" => Rps(1),
                    "C" | "Z" => Rps(2),
                    _ => unreachable!(),
                }
            }

            fn winning_against(other: &Rps) -> Rps {
                Rps((other.0 + 1) % 3)
            }

            fn losing_against(other: &Rps) -> Rps {
                Rps((other.0 + 3 - 1) % 3)
            }
        }

        let scores = input.iter().map(|line| {
            let mut tokens = line.split_whitespace();
            let opponent = Rps::parse(tokens.next().unwrap());
            let me = Rps::parse(tokens.next().unwrap());
            me.score() + me.score_against(&opponent)
        });

        let scores2 = input.iter().map(|line| {
            let mut tokens = line.split_whitespace();
            let opponent = Rps::parse(tokens.next().unwrap());
            let me: Rps = match tokens.next().unwrap() {
                "X" => Rps::losing_against(&opponent),
                "Y" => opponent,
                "Z" => Rps::winning_against(&opponent),
                _ => unreachable!(),
            };
            me.score() + me.score_against(&opponent)
        });

        Self::solutions(scores.sum::<u32>(), scores2.sum::<u32>())
    }

    fn day3(input: Vec<String>) -> Answers {
        fn priority(item: char) -> u32 {
            if item.is_lowercase() {
                item as u32 - 'a' as u32 + 1
            } else {
                item as u32 - 'A' as u32 + 27
            }
        }

        let rucksacks = input.iter().map(|line| {
            let size = line.len() / 2;
            let comp1 = line[..size].chars().collect::<HashSet<_>>();
            let comp2 = line[size..].chars().collect::<HashSet<_>>();
            let common = comp1.intersection(&comp2).next().unwrap();
            priority(*common)
        });

        let groups = input.chunks(3);
        let badges = groups.map(|group| {
            let mut sacks = group
                .iter()
                .map(|line| line.chars().collect::<HashSet<_>>());
            let sack1 = sacks.next().unwrap();
            let sack2 = sacks.next().unwrap();
            let sack3 = sacks.next().unwrap();
            let common = sack1
                .intersection(&sack2)
                .copied()
                .collect::<HashSet<char>>()
                .intersection(&sack3)
                .copied()
                .next()
                .unwrap();
            priority(common)
        });

        Self::solutions(rucksacks.sum::<u32>(), badges.sum::<u32>())
    }

    fn day4(input: Vec<String>) -> Answers {
        fn parse_interval(line: &str) -> (u32, u32) {
            let (start, end) = line.split_once('-').unwrap();
            (start.parse().unwrap(), end.parse().unwrap())
        }

        fn contains((a1, a2): &(u32, u32), (b1, b2): &(u32, u32)) -> bool {
            b1 >= a1 && b2 <= a2
        }

        fn overlaps((a1, a2): &(u32, u32), (b1, b2): &(u32, u32)) -> bool {
            a1 <= b2 && b1 <= a2
        }

        let intervals = input.iter().map(|line| {
            let (first, second) = line.split_once(',').unwrap();
            let first = parse_interval(first);
            let second = parse_interval(second);
            (first, second)
        });

        let num_containing = intervals
            .clone()
            .filter(|(first, second)| contains(first, second) || contains(second, first))
            .count();

        let num_overlapping = intervals
            .filter(|(first, second)| overlaps(first, second))
            .count();

        Self::solutions(num_containing, num_overlapping)
    }

    fn day5(input: Vec<String>) -> Answers {
        fn div_ceil(a: usize, b: usize) -> usize {
            (a + b - 1) / b
        }

        fn parse_move_instr(line: &str) -> (usize, usize, usize) {
            let mut words = line.split_whitespace();
            let amount = words.by_ref().nth(1).unwrap().parse::<usize>().unwrap();
            let src_stack = words.by_ref().nth(1).unwrap().parse::<usize>().unwrap() - 1;
            let dst_stack = words.by_ref().nth(1).unwrap().parse::<usize>().unwrap() - 1;
            (amount, src_stack, dst_stack)
        }

        fn move_fifo(amount: usize, src_stack: usize, dst_stack: usize, stacks: &mut [Vec<char>]) {
            for _ in 0..amount {
                let item: char = stacks[src_stack].pop().unwrap();
                stacks[dst_stack].push(item);
            }
        }

        fn move_batch(amount: usize, src_stack: usize, dst_stack: usize, stacks: &mut [Vec<char>]) {
            let start_index = stacks[src_stack].len() - amount;
            let rest = stacks[src_stack].split_off(start_index);
            stacks[dst_stack].extend(rest);
        }

        fn top_string(stacks: &[Vec<char>]) -> String {
            stacks.iter().map(|stack| stack.last().unwrap()).collect()
        }

        let mut stacks = vec![Vec::new(); 9];
        let mut lines = input.iter();

        for line in lines.by_ref().take_while(|line| !line.starts_with(" 1")) {
            let row_len = div_ceil(line.len(), 4);
            stacks.resize_with(row_len, Vec::new);
            for (crate_id, stack) in line.chars().skip(1).step_by(4).zip(stacks.iter_mut()) {
                if crate_id != ' ' {
                    stack.push(crate_id);
                }
            }
        }
        // Stacks are inserted top-down, but we want the top to be the last item
        for stack in stacks.iter_mut() {
            stack.reverse();
        }

        lines.next(); // Skip empty line
        let instructions = lines.map(|s| parse_move_instr(s)).collect::<Vec<_>>();

        let stacks_fifo = {
            let mut stacks = stacks.clone();
            for (amount, src_stack, dst_stack) in instructions.iter().cloned() {
                move_fifo(amount, src_stack, dst_stack, &mut stacks);
            }
            stacks
        };
        let stacks_batch = {
            for (amount, src_stack, dst_stack) in instructions {
                move_batch(amount, src_stack, dst_stack, &mut stacks);
            }
            stacks
        };

        Self::solutions(top_string(&stacks_fifo), top_string(&stacks_batch))
    }

    fn day6(input: Vec<String>) -> Answers {
        fn find_packet_of_len(message: &[char], len: usize) -> usize {
            message
                .windows(len)
                .zip(len..)
                .find(|(chars, _i)| {
                    let unique = chars.iter().collect::<HashSet<_>>();
                    unique.len() == len
                })
                .unwrap()
                .1
        }

        let input = input[0].chars().collect::<Vec<_>>();
        let start_of_packet = find_packet_of_len(&input, 4);
        let start_of_message = find_packet_of_len(&input, 14);
        Self::solutions(start_of_packet, start_of_message)
    }

    fn day7(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        enum FsEntry {
            Directory { children: Vec<PathBuf> },
            File { size: usize },
        }

        #[derive(Debug)]
        enum Command<'a> {
            Cd(&'a str),
            Ls,
        }

        fn parse_command(line: &str) -> Command {
            let mut tokens = line.split_whitespace();
            tokens.next(); // skip $
            match tokens.next().expect("Not enough tokens in string") {
                "cd" => Command::Cd(tokens.next().unwrap()),
                "ls" => Command::Ls,
                cmd => panic!("unexpected command {}", cmd),
            }
        }

        fn parse_fs_entry(line: &str) -> (&Path, FsEntry) {
            if line.starts_with("dir") {
                let dir_name = line.strip_prefix("dir ").unwrap();
                (
                    Path::new(dir_name),
                    FsEntry::Directory {
                        children: Vec::new(),
                    },
                )
            } else {
                let mut parts = line.split_whitespace();
                let size = parts.next().unwrap().parse::<usize>().unwrap();
                let name = parts.next().unwrap();
                (Path::new(name), FsEntry::File { size })
            }
        }

        type FileSystem = HashMap<PathBuf, FsEntry>;
        fn populate_dir_tree(lines: &[String]) -> FileSystem {
            let mut fs = FileSystem::new();
            let mut cur_path = PathBuf::from("/");
            fs.insert(
                cur_path.clone(),
                FsEntry::Directory {
                    children: Vec::new(),
                },
            );

            let mut input = lines.iter().peekable();
            loop {
                let Some(line) = input.next() else { break; };
                match parse_command(line) {
                    Command::Ls => {
                        while input.peek().map(|s| !s.starts_with('$')).unwrap_or(false) {
                            let (name, entry) = parse_fs_entry(input.next().unwrap());
                            let child_path = cur_path.join(name);
                            fs.insert(child_path.clone(), entry);
                            let Some(FsEntry::Directory{children}) = fs.get_mut(&cur_path) else {
                                panic!("Cur fsentry ({:?}) not a dir", cur_path);
                            };
                            children.push(child_path);
                        }
                    }
                    Command::Cd(dir) => match dir {
                        ".." => {
                            cur_path.pop();
                        }
                        "/" => {
                            cur_path = PathBuf::from("/");
                        }
                        dir => {
                            cur_path.push(dir);
                        }
                    },
                }
            }
            fs
        }

        fn total_size(filename: &Path, fs: &FileSystem) -> usize {
            match fs.get(filename).unwrap() {
                FsEntry::Directory { children } => children
                    .iter()
                    .map(|filename| total_size(filename, fs))
                    .sum(),
                FsEntry::File { size } => *size,
            }
        }

        let fs = populate_dir_tree(&input);
        let file_sizes = fs.keys().filter_map(|filename| {
            if let Some(FsEntry::Directory { .. }) = fs.get(filename) {
                Some(total_size(filename, &fs))
            } else {
                None
            }
        });
        let small_dir_size: usize = file_sizes.clone().filter(|x| x < &100000).sum();

        const TOTAL_DISK_SPACE: usize = 70000000;
        const UNUSED_SPACE_NEEDED: usize = 30000000;
        let min_size_to_free =
            UNUSED_SPACE_NEEDED - (TOTAL_DISK_SPACE - total_size(Path::new("/"), &fs));

        let min_freeable_dir_size = file_sizes.filter(|x| x > &min_size_to_free).min().unwrap();
        Self::solutions(small_dir_size, min_freeable_dir_size)
    }

    fn day8(input: Vec<String>) -> Answers {
        type Grid = Vec<Vec<bool>>;

        fn determine_visible(
            mut position: (isize, isize),
            direction: (isize, isize),
            heights: &[Vec<i32>],
            visible: &mut Grid,
        ) {
            let size = visible.len() as isize;
            let mut max_height = i32::min_value();
            while (0..size).contains(&position.0) && (0..size).contains(&position.1) {
                let height = heights[position.0 as usize][position.1 as usize];
                if height > max_height {
                    max_height = height;
                    visible[position.0 as usize][position.1 as usize] = true;
                }
                position.0 += direction.0;
                position.1 += direction.1;
            }
        }

        let heights: Vec<Vec<i32>> = input
            .iter()
            .map(|line| {
                line.chars()
                    .map(|c| c.to_digit(10).unwrap() as i32)
                    .collect()
            })
            .collect();

        let size = heights.len() as isize;
        let num_visible: usize = {
            let mut visible = vec![vec![false; size as usize]; size as usize];
            for i in 0..size {
                determine_visible((i, 0), (0, 1), &heights, &mut visible); // left side looks right
                determine_visible((0, i), (1, 0), &heights, &mut visible); // top side looks down
                determine_visible((i, size - 1), (0, -1), &heights, &mut visible); // right side looks left
                determine_visible((size - 1, i), (-1, 0), &heights, &mut visible);
                // bottom side looks up
            }

            visible
                .iter()
                .map(|line| line.iter().filter(|x| **x).count())
                .sum()
        };

        fn viewing_distance(
            mut position: (isize, isize),
            direction: (isize, isize),
            heights: &Vec<Vec<i32>>,
        ) -> usize {
            let size = heights.len() as isize;
            let max_height = heights[position.0 as usize][position.1 as usize];
            let mut distance = 0;
            while (0..size).contains(&(position.0 + direction.0))
                && (0..size).contains(&(position.1 + direction.1))
            {
                position.0 += direction.0;
                position.1 += direction.1;
                distance += 1;
                let height = heights[position.0 as usize][position.1 as usize];
                if height >= max_height {
                    break;
                }
            }
            distance
        }

        let max_score = {
            let mut max_score = 0;
            for y in 0..size {
                for x in 0..size {
                    let pos = (y, x);
                    let score = viewing_distance(pos, (1, 0), &heights)
                        * viewing_distance(pos, (0, 1), &heights)
                        * viewing_distance(pos, (-1, 0), &heights)
                        * viewing_distance(pos, (0, -1), &heights);
                    max_score = score.max(max_score);
                }
            }
            max_score
        };

        Self::solutions(num_visible, max_score)
    }

    fn day9(input: Vec<String>) -> Answers {
        type Rope = Vec<(i32, i32)>;

        fn distance((y0, x0): (i32, i32), (y1, x1): (i32, i32)) -> u32 {
            u32::max(x1.abs_diff(x0), y1.abs_diff(y0))
        }

        fn move_towards(mut tail: (i32, i32), head: (i32, i32)) -> (i32, i32) {
            match head.0.cmp(&tail.0) {
                Ordering::Greater => {
                    tail.0 += 1;
                }
                Ordering::Less => {
                    tail.0 -= 1;
                }
                _ => (),
            }
            match head.1.cmp(&tail.1) {
                Ordering::Greater => {
                    tail.1 += 1;
                }
                Ordering::Less => {
                    tail.1 -= 1;
                }
                _ => (),
            }

            tail
        }

        fn step_rope(rope: &mut Rope, direction: (i32, i32)) {
            rope[0].0 += direction.0;
            rope[0].1 += direction.1;
            for i in 0..(rope.len() - 1) {
                let head = rope[i];
                let tail = &mut rope[i + 1];
                if distance(head, *tail) > 1 {
                    *tail = move_towards(*tail, head);
                }
            }
        }

        fn spots_visited(mut rope: Rope, input: &[String]) -> usize {
            let mut visited = HashSet::new();
            for line in input {
                let mut tokens = line.split_whitespace();
                let direction = tokens.next().unwrap().chars().next().unwrap();
                let amount: i32 = tokens.next().unwrap().parse().unwrap();

                for _ in 0..amount {
                    step_rope(
                        &mut rope,
                        match direction {
                            'R' => (0, 1),
                            'L' => (0, -1),
                            'U' => (-1, 0),
                            'D' => (1, 0),
                            _ => unreachable!(),
                        },
                    );

                    visited.insert(*rope.last().unwrap());
                }
            }
            visited.len()
        }

        let short_rope: Rope = vec![(0, 0); 2];
        let long_rope: Rope = vec![(0, 0); 10];

        Self::solutions(
            spots_visited(short_rope, &input),
            spots_visited(long_rope, &input),
        )
    }

    fn day10(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        enum State {
            Wait,
            Change(isize),
        }

        #[derive(Default)]
        struct Crt {
            stack: Vec<State>,
            x: isize,
            cycle: isize,
        }

        const LINE_WIDTH: isize = 40;
        impl Crt {
            fn new() -> Self {
                Self {
                    stack: Vec::new(),
                    x: 1,
                    cycle: 1,
                }
            }

            fn dispatch(&mut self, instr: &str) {
                let mut tokens = instr.split_whitespace();
                let instr = tokens.next();
                match instr {
                    Some("noop") => self.stack.push(State::Wait),
                    Some("addx") => {
                        let operand: isize = tokens.next().unwrap().parse().unwrap();
                        self.stack.push(State::Change(operand));
                        self.stack.push(State::Wait);
                    }
                    _ => {}
                }
            }

            fn step_cycle(&mut self, mut instrs: impl Iterator<Item = String>) {
                if self.stack.is_empty() {
                    if let Some(line) = instrs.next() {
                        self.dispatch(&line);
                    } else {
                        self.stack.push(State::Wait);
                    }
                }
                if let State::Change(operand) = self.stack.pop().unwrap() {
                    self.x += operand;
                }
                self.cycle += 1;
            }

            fn signal_strength(&self) -> isize {
                self.x * self.cycle
            }

            fn output(&self) -> char {
                let pos = (self.cycle - 1) % LINE_WIDTH;
                if self.x.abs_diff(pos) <= 1 {
                    '#'
                } else {
                    '.'
                }
            }

            fn is_at_end(&self) -> bool {
                (self.cycle - 1) % LINE_WIDTH == 0
            }
        }

        let mut instrs = input.into_iter();
        let mut crt = Crt::new();

        let mut total_stength = 0;
        let mut output = String::new();
        for target_cycle in (20..=220_isize).step_by(40).chain(std::iter::once(240)) {
            while crt.cycle != target_cycle {
                if crt.is_at_end() {
                    output.push('\n');
                }
                crt.step_cycle(&mut instrs);
                output.push(crt.output());
            }
            total_stength += crt.signal_strength();
        }

        Self::solutions(total_stength, output)
    }

    fn day11(input: Vec<String>) -> Answers {
        #[derive(Debug, Clone)]
        struct Monkey {
            items: Vec<usize>,
            operation: Vec<String>,
            divisor: usize,
            targets: (usize, usize),
            items_inspected: usize,
        }

        fn parse(lines: &[String]) -> Result<Monkey, &str> {
            let mut lines = lines.iter();
            lines.next(); // Skip "Monkey X"
            let items = parse::trimmed("Starting items:")
                .right(parse::uint.interspersed(", "))
                .parse_exact(lines.next().unwrap())?;
            let operation = parse::trimmed("Operation: new =")
                .pars(lines.next().unwrap())?
                .1
                .split_whitespace()
                .map(str::to_owned)
                .collect();
            let divisor = parse::trimmed("Test: divisible by")
                .right(parse::uint)
                .parse_exact(lines.next().unwrap())?;
            let target_true = parse::trimmed("If true: throw to monkey")
                .right(parse::uint)
                .parse_exact(lines.next().unwrap())?;
            let target_false = parse::trimmed("If false: throw to monkey")
                .right(parse::uint)
                .parse_exact(lines.next().unwrap())?;

            Ok(Monkey {
                items,
                operation,
                divisor,
                targets: (target_true, target_false),
                items_inspected: 0,
            })
        }

        fn compute_operation(value: usize, op: &[String]) -> usize {
            fn parse(old: usize, value: &str) -> usize {
                if value == "old" {
                    old
                } else {
                    value.parse().unwrap()
                }
            }

            let l = parse(value, &op[0]);
            let r = parse(value, &op[2]);
            match op[1].as_str() {
                "*" => l * r,
                "+" => l + r,
                _ => unreachable!(),
            }
        }

        fn perform_round(monkeys: &mut [Monkey], divisor: usize, modulo: usize) {
            for i in 0..monkeys.len() {
                let items = std::mem::take(&mut monkeys[i].items);
                for item in items {
                    let worry = compute_operation(item, &monkeys[i].operation) / divisor;
                    let target = if worry % monkeys[i].divisor == 0 {
                        monkeys[i].targets.0
                    } else {
                        monkeys[i].targets.1
                    };
                    monkeys[i].items_inspected += 1;
                    monkeys[target].items.push(worry % modulo);
                }
            }
        }

        fn monkey_business_level(
            mut monkeys: Vec<Monkey>,
            num_rounds: usize,
            divisor: usize,
        ) -> usize {
            let factor = monkeys.iter().map(|m| m.divisor).product();
            for _round in 0..num_rounds {
                perform_round(&mut monkeys, divisor, factor);
            }

            monkeys.sort_by_key(|monkey| monkey.items_inspected);
            monkeys.reverse();

            monkeys[0].items_inspected * monkeys[1].items_inspected
        }

        let monkeys = input
            .split(String::is_empty)
            .map(parse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        Self::solutions(
            monkey_business_level(monkeys.clone(), 20, 3),
            monkey_business_level(monkeys, 10000, 1),
        )
    }

    fn day12(input: Vec<String>) -> Answers {
        fn find_in_map(map: &[Vec<char>], target: char) -> (usize, usize) {
            map.iter()
                .enumerate()
                .filter_map(move |(row, line)| {
                    line.iter()
                        .position(move |c| *c == target)
                        .map(move |col| (row, col))
                })
                .next()
                .unwrap()
        }

        fn check_move(
            (a, b): (usize, usize),
            direction: (isize, isize),
            map: &Vec<Vec<char>>,
        ) -> Option<(usize, usize)> {
            let height = map[a][b];
            if let Some((a1, b1)) = a
                .checked_add_signed(direction.0)
                .filter(|a| *a < map.len())
                .zip(
                    b.checked_add_signed(direction.1)
                        .filter(|b| *b < map[a].len()),
                )
            {
                // We search from end to start, so we want to descend in height
                // dest <= 1 + cur
                // dest >= cur - 1
                // neigh_height <= 1 + cur_height
                if map[a1][b1] as usize >= height as usize - 1 {
                    return Some((a1, b1));
                }
            }
            None
        }

        #[derive(Copy, Clone, Eq, PartialEq)]
        struct State {
            distance: u32,
            position: (usize, usize),
        }

        // The priority queue depends on `Ord`.
        // Explicitly implement the trait so the queue becomes a min-heap
        // instead of a max-heap.
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                // Notice that the we flip the ordering on costs.
                // In case of a tie we compare positions - this step is necessary
                // to make implementations of `PartialEq` and `Ord` consistent.
                other
                    .distance
                    .cmp(&self.distance)
                    .then_with(|| self.position.cmp(&other.position))
            }
        }

        // `PartialOrd` needs to be implemented as well.
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        fn shortest_distances(map: &Vec<Vec<char>>, start: (usize, usize)) -> Vec<Vec<u32>> {
            let (n, m) = (map.len(), map[0].len());
            let mut distances = vec![vec![u32::max_value(); m]; n];
            let mut heap = BinaryHeap::new();

            distances[start.0][start.1] = 0;
            heap.push(State {
                distance: 0,
                position: start,
            });

            while let Some(State { distance, position }) = heap.pop() {
                let cur_distance = distances[position.0][position.1];
                if distance > cur_distance {
                    continue;
                }

                for (a, b) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    .into_iter()
                    .filter_map(|dir| check_move(position, dir, map))
                {
                    let next = State {
                        distance: distance + 1,
                        position: (a, b),
                    };
                    if next.distance < distances[a][b] {
                        heap.push(next);
                        distances[a][b] = next.distance;
                    }
                }
            }
            distances
        }

        let mut map = input
            .iter()
            .map(|line| line.chars().collect())
            .collect::<Vec<_>>();
        let start = find_in_map(&map, 'S');
        let end = find_in_map(&map, 'E');
        map[start.0][start.1] = 'a';
        map[end.0][end.1] = 'z';

        let distances_from_end = shortest_distances(&map, end);
        let part1 = distances_from_end[start.0][start.1];

        let start_points = map.iter().enumerate().flat_map(|(row, line)| {
            line.iter()
                .copied()
                .enumerate()
                .filter(|(_, c)| *c == 'a')
                .map(move |(col, _)| (row, col))
        });
        let part2 = start_points
            .map(|(a, b)| distances_from_end[a][b])
            .min()
            .unwrap();

        Self::solutions(part1, part2)
    }

    fn day13(input: Vec<String>) -> Answers {
        #[derive(Debug, PartialEq, Eq, Clone)]
        enum List {
            Int(u32),
            List(Vec<List>),
        }

        fn parse(line: &str) -> (List, &str) {
            if let Some(mut rest) = line.strip_prefix('[') {
                let mut elems = Vec::new();
                while !rest.starts_with(']') {
                    let (elem, cont) = parse(rest);
                    rest = cont.trim_start_matches(',');
                    elems.push(elem);
                }
                (List::List(elems), &rest[1..])
            } else {
                let num_length = line.chars().take_while(|c| c.is_numeric()).count();
                let num: u32 = line[..num_length].parse::<u32>().unwrap();
                (List::Int(num), &line[num_length..])
            }
        }

        fn parse_pair(lines: &[String]) -> (List, List) {
            assert_eq!(lines.len(), 2);
            let (line_a, rest_a) = parse(&lines[0]);
            let (line_b, rest_b) = parse(&lines[1]);
            assert!(rest_a.is_empty());
            assert!(rest_b.is_empty());
            (line_a, line_b)
        }

        impl PartialOrd for List {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                match (self, other) {
                    (List::Int(a), List::Int(b)) => a.partial_cmp(b),
                    (a @ List::Int(_), b) => List::List(vec![a.clone()]).partial_cmp(b),
                    (a, b @ List::Int(_)) => a.partial_cmp(&List::List(vec![b.clone()])),
                    (List::List(a), List::List(b)) => a.partial_cmp(b),
                }
            }
        }

        impl Ord for List {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        let line_pairs = input.split(String::is_empty).map(parse_pair);

        let lt_index_sum: usize = line_pairs
            .clone()
            .enumerate()
            .filter(|(_, (a, b))| a < b)
            .map(|pair| pair.0 + 1)
            .sum::<usize>();

        let mut sorted = line_pairs.flat_map(|(a, b)| [a, b]).collect::<Vec<_>>();
        let divider_a = List::List(vec![List::List(vec![List::Int(2)])]);
        let divider_b = List::List(vec![List::List(vec![List::Int(6)])]);
        sorted.push(divider_a.clone());
        sorted.push(divider_b.clone());
        sorted.sort_unstable();

        let packet_index_a = sorted.iter().position(|list| list == &divider_a).unwrap() + 1;
        let packet_index_b = sorted.iter().position(|list| list == &divider_b).unwrap() + 1;
        let decoder_key = packet_index_a * packet_index_b;

        Self::solutions(lt_index_sum, decoder_key)
    }

    fn day14(input: Vec<String>) -> Answers {
        /// like a..b but swaps if a > b
        fn range(a: usize, b: usize) -> std::ops::RangeInclusive<usize> {
            if a > b {
                b..=a
            } else {
                a..=b
            }
        }

        fn simulate_grain(
            rocks: &HashSet<(usize, usize)>,
            grains: &HashSet<(usize, usize)>,
            bottom: usize,
            floor: Option<usize>,
        ) -> Option<(usize, usize)> {
            let mut grain = (500, 0);
            while grain.1 <= bottom {
                let next_pos = [
                    (grain.0, grain.1 + 1),
                    (grain.0 - 1, grain.1 + 1),
                    (grain.0 + 1, grain.1 + 1),
                ]
                .into_iter()
                .find(|pos| !rocks.contains(pos) && !grains.contains(pos));

                if let Some(pos) = next_pos {
                    match floor {
                        Some(floor) if pos.1 >= floor => {
                            return Some(grain);
                        }
                        _ => (),
                    }
                    grain = pos;
                } else {
                    return Some(grain);
                }
            }
            None
        }

        let rock_lines = input.iter().map(|s| {
            parse::uint::<usize>
                .left(',')
                .pair(parse::uint::<usize>)
                .interspersed(" -> ")
                .parse_exact(s)
                .expect("Parse error")
        });

        let mut rocks = HashSet::new();
        let mut bottom = 0;
        for line in rock_lines {
            for nodes in line.windows(2) {
                let [(x0, y0), (x1, y1)] = nodes else {
                    unreachable!()
                };

                for x in range(*x0, *x1) {
                    rocks.insert((x, *y0));
                }
                for y in range(*y0, *y1) {
                    rocks.insert((*x0, y));
                    bottom = bottom.max(y);
                }
            }
        }

        let mut grains = HashSet::new();
        while let Some(resting_place) = simulate_grain(&rocks, &grains, bottom, None) {
            grains.insert(resting_place);
        }
        let part1 = grains.len();

        let floor = bottom + 2;
        while !grains.contains(&(500, 0)) {
            let Some(resting_place) = simulate_grain(&rocks, &grains, floor + 1, Some(floor)) else {
                    unreachable!();
                };
            grains.insert(resting_place);
        }

        Self::solutions(part1, grains.len())
    }

    fn day15(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        struct Sensor {
            pos: (isize, isize),
            beacon: (isize, isize),
        }

        impl Sensor {
            fn range(&self) -> usize {
                self.pos.0.abs_diff(self.beacon.0) + self.pos.1.abs_diff(self.beacon.1)
            }
        }

        fn merge_ranges(ranges: Vec<RangeInclusive<isize>>) -> Vec<RangeInclusive<isize>> {
            let mut i = 0;
            let mut ret = Vec::new();
            while i < ranges.len() {
                let (start, mut end) = ranges[i].clone().into_inner();
                i += 1;
                while i < ranges.len() && ranges[i].start() <= &end {
                    end = end.max(*ranges[i].end());
                    i += 1;
                }
                ret.push(start..=end);
            }
            ret
        }

        fn merged_coverage_ranges_at(
            sensors: &[Sensor],
            ypos: isize,
        ) -> Vec<RangeInclusive<isize>> {
            let mut ranges = sensors
                .iter()
                .filter_map(move |s| {
                    let distance = ypos.abs_diff(s.pos.1);
                    if s.range() < distance {
                        return None;
                    }
                    let width = (s.range() - distance) as isize;
                    let (left, right) = (s.pos.0 - width, s.pos.0 + width);
                    Some(left..=right)
                })
                .collect::<Vec<_>>();

            ranges.sort_unstable_by_key(|range| range.clone().into_inner());
            merge_ranges(ranges)
        }

        let sensors = input
            .iter()
            .map(|s| {
                let sensor = "Sensor at x="
                    .right(parse::int::<isize>)
                    .left(", y=")
                    .pair(parse::int::<isize>);
                let beacon = ": closest beacon is at x="
                    .right(parse::int::<isize>)
                    .left(", y=")
                    .pair(parse::int::<isize>);
                let (pos, beacon) = sensor.pair(beacon).parse_exact(s).unwrap();
                Sensor { pos, beacon }
            })
            .collect::<Vec<_>>();

        const Y_POS: isize = 2000000;

        let ranges = merged_coverage_ranges_at(&sensors, Y_POS);
        let beacons_in_row = sensors
            .iter()
            .map(|s| s.beacon)
            .filter(|b| b.1 == Y_POS)
            .collect::<Vec<_>>();

        let part1 = ranges
            .into_iter()
            .map(|r| {
                r.end() - r.start() + isize::from(!beacons_in_row.iter().any(|b| r.contains(&b.1)))
            })
            .sum::<isize>();

        const MAP_SIZE: isize = 4000000;

        let (x, y) = (0..MAP_SIZE)
            .find_map(|y| {
                let ranges = merged_coverage_ranges_at(&sensors, y);
                if ranges.len() > 1 {
                    Some((ranges[0].end() + 1, y))
                } else {
                    None
                }
            })
            .unwrap();

        let frequency = x * MAP_SIZE + y;

        Self::solutions(part1, frequency)
    }

    fn day16(input: Vec<String>) -> Answers {
        type Valve<'a> = (usize, Vec<&'a str>);

        fn visit<'a>(
            valve: &'a str,
            valves: &'a HashMap<&'a str, Valve>,
            open: &mut HashSet<&'a str>,
            time_remaining: usize,
        ) -> usize {
            if time_remaining == 0 {
                return 0;
            }

            let (flow, neighbors) = &valves[valve];
            let pressure = open.iter().map(|v| valves[v].0).sum::<usize>();

            let mut max_pressure = pressure;
            if *flow > 0 && !open.contains(valve) {
                let time_remaining = time_remaining - 1;
                let pressure = pressure * 2;
                open.insert(valve);
                for neighbor in neighbors {
                    max_pressure = max_pressure
                        .max(pressure + visit(neighbor, valves, open, time_remaining - 1));
                }
                open.remove(valve);
            }
            for neighbor in neighbors {
                max_pressure =
                    max_pressure.max(pressure + visit(neighbor, valves, open, time_remaining - 1));
            }

            max_pressure
        }

        let valves = input
            .iter()
            .map(|line| {
                "Valve "
                    .right(parse::word)
                    .left(" has flow rate=")
                    .pair(
                        parse::uint::<usize>.pair(
                            "; tunnels lead to valves "
                                .or("; tunnel leads to valve ")
                                .right(parse::word.interspersed(", ")),
                        ),
                    )
                    .parse_exact(line)
                    .unwrap()
            })
            .collect();

        let mut open = HashSet::new();

        Self::part1(visit("AA", &valves, &mut open, 30))
    }

    fn day17(input: Vec<String>) -> Answers {
        const MINUS: [u8; 1] = [0b0011110];
        const PLUS: [u8; 3] = [0b0001000, 0b0011100, 0b0001000];
        const LANGLE: [u8; 3] = [0b0000100, 0b0000100, 0b0011100];
        const POLE: [u8; 4] = [0b0010000; 4];
        const SQUARE: [u8; 2] = [0b0011000; 2];

        const SHAPES: [&[u8]; 5] = [&MINUS, &PLUS, &LANGLE, &POLE, &SQUARE];

        fn bit_at(x: u8, index: u8) -> bool {
            x & (1 << index) != 0
        }

        fn move_left(shape: &mut [u8], height: usize, resting: &[u8]) {
            if shape.iter().any(|&x| bit_at(x, 6)) {
                return;
            }
            if resting.len() > height
                && resting[height..]
                    .iter()
                    .zip(shape.iter())
                    .any(|(a, b)| a & (b << 1) != 0)
            {
                return;
            }
            for x in shape.iter_mut() {
                *x <<= 1;
            }
        }

        fn move_right(shape: &mut [u8], height: usize, resting: &[u8]) {
            if shape.iter().any(|&x| bit_at(x, 0)) {
                return;
            }
            if resting.len() > height
                && resting[height..]
                    .iter()
                    .zip(shape.iter())
                    .any(|(a, b)| a & (b >> 1) != 0)
            {
                return;
            }
            for x in shape.iter_mut() {
                *x >>= 1;
            }
        }

        fn can_move_down(shape: &[u8], height: usize, resting: &[u8]) -> bool {
            if height >= resting.len() {
                return true;
            }
            resting[height..].iter().zip(shape).all(|(a, b)| a & b == 0)
        }

        fn place_resting(shape: &[u8], height: usize, resting: &mut Vec<u8>) {
            resting.resize(resting.len().max(height + shape.len()), 0);
            for (a, b) in resting[height..].iter_mut().zip(shape) {
                *a |= b;
            }
        }

        fn place_piece(
            mut shape: Vec<u8>,
            jets: &mut impl Iterator<Item = (usize, char)>,
            resting: &mut Vec<u8>,
        ) -> usize {
            // Store shapes coherently with the resting vector, bottom-to-top.
            shape.reverse();
            let mut height = resting.len() + 3;
            loop {
                let (jet_num, c) = jets.next().unwrap();
                match c {
                    '<' => move_left(&mut shape, height, resting),
                    '>' => move_right(&mut shape, height, resting),
                    _ => (),
                }

                if height == 0 || !can_move_down(&shape, height - 1, resting) {
                    place_resting(&shape, height, resting);
                    break jet_num;
                } else {
                    height -= 1;
                }
            }
        }

        fn last_n(slice: &[u8], n: usize) -> &[u8] {
            if n <= slice.len() {
                &slice[slice.len() - n..]
            } else {
                slice
            }
        }

        fn _print_stack(resting: &[u8]) {
            for x in resting.iter().rev() {
                for i in (0..7).rev() {
                    print!("{}", if bit_at(*x, i) { '#' } else { '.' });
                }
                println!();
            }
            println!();
        }

        let shapes = SHAPES.into_iter().map(|x| x.to_vec());
        let jets = input.join("");

        let part1 = {
            let mut resting: Vec<u8> = Vec::new();
            let mut jets = jets.chars().enumerate().cycle();

            for shape in shapes.clone().cycle().take(2022) {
                place_piece(shape, &mut jets, &mut resting);
            }
            resting.len()
        };

        let part2 = {
            const NUM_SHAPES_TOTAL: usize = 1000000000000;
            let mut resting: Vec<u8> = Vec::new();
            let mut jets = jets.chars().enumerate().cycle();
            let mut states = HashMap::new();

            let mut num_shapes = 0;
            loop {
                let last_jet = shapes
                    .clone()
                    .map(|shape| place_piece(shape, &mut jets, &mut resting))
                    .last()
                    .unwrap();
                num_shapes += shapes.len();

                let state = (last_jet, last_n(&resting, 50).to_vec());

                if let Some((old_shapes, old_height)) =
                    states.insert(state, (num_shapes, resting.len()))
                {
                    let shapes_since = num_shapes - old_shapes;
                    let mut total_height = resting.len();
                    let len = total_height - old_height;
                    while num_shapes + shapes_since <= NUM_SHAPES_TOTAL {
                        num_shapes += shapes_since;
                        total_height += len;
                    }
                    let starting_height = resting.len();
                    for shape in shapes.cycle().take(NUM_SHAPES_TOTAL - num_shapes) {
                        num_shapes += 1;
                        place_piece(shape, &mut jets, &mut resting);
                    }
                    total_height += resting.len() - starting_height;
                    break total_height;
                }
            }
        };

        Self::solutions(part1, part2)
    }

    fn day18(input: Vec<String>) -> Answers {
        let cubes: HashSet<(isize, isize, isize)> = input
            .iter()
            .map(|line| {
                let coords = parse::uint::<isize>
                    .interspersed(',')
                    .parse_exact(line)
                    .unwrap();
                let [x, y, z] = coords[..] else {
                    unreachable!();
                };
                (x, y, z)
            })
            .collect();

        let mut part1 = 0;
        for (x, y, z) in cubes.iter().copied() {
            part1 += usize::from(!cubes.contains(&(x - 1, y, z)));
            part1 += usize::from(!cubes.contains(&(x + 1, y, z)));
            part1 += usize::from(!cubes.contains(&(x, y - 1, z)));
            part1 += usize::from(!cubes.contains(&(x, y + 1, z)));
            part1 += usize::from(!cubes.contains(&(x, y, z - 1)));
            part1 += usize::from(!cubes.contains(&(x, y, z + 1)));
        }

        let (xrange, yrange, zrange) = {
            let xmin = cubes.iter().map(|p| p.0).min().unwrap();
            let xmax = cubes.iter().map(|p| p.0).max().unwrap();
            let ymin = cubes.iter().map(|p| p.1).min().unwrap();
            let ymax = cubes.iter().map(|p| p.1).max().unwrap();
            let zmin = cubes.iter().map(|p| p.2).min().unwrap();
            let zmax = cubes.iter().map(|p| p.2).max().unwrap();
            (
                (xmin - 1)..=(xmax + 1),
                (ymin - 1)..=(ymax + 1),
                (zmin - 1)..=(zmax + 1),
            )
        };

        let is_in_bbox = |&(x, y, z): &(isize, isize, isize)| -> bool {
            xrange.contains(&x) && yrange.contains(&y) & zrange.contains(&z)
        };

        let part2 = {
            let mut visited = HashSet::new();
            visited.insert((*xrange.start(), *yrange.start(), *zrange.start()));
            let mut boundary = vec![(*xrange.start(), *yrange.start(), *zrange.start())];
            let mut surface_area = 0;
            while !boundary.is_empty() {
                let (x, y, z) = boundary.pop().unwrap();
                for neighbor in [
                    (x - 1, y, z),
                    (x + 1, y, z),
                    (x, y - 1, z),
                    (x, y + 1, z),
                    (x, y, z - 1),
                    (x, y, z + 1),
                ]
                .iter()
                .copied()
                {
                    if cubes.contains(&neighbor) {
                        surface_area += 1;
                    } else if is_in_bbox(&neighbor) && visited.insert(neighbor) {
                        boundary.push(neighbor);
                    }
                }
            }
            surface_area
        };

        Self::solutions(part1, part2)
    }

    fn day19(input: Vec<String>) -> Answers {
        #[derive(Debug)]
        struct BlueprintCosts {
            ore: usize,
            clay: usize,
            obsidian: (usize, usize),
            geode: (usize, usize),
        }

        fn parse(input: &String) -> BlueprintCosts {
            "Blueprint "
                .right(parse::uint::<usize>)
                .right(": Each ore robot costs ")
                .right(parse::uint)
                .left(" ore. ")
                .pair("Each clay robot costs ".right(parse::uint).left(" ore. "))
                .pair(
                    "Each obsidian robot costs "
                        .right(parse::uint)
                        .left(" ore and ")
                        .pair(parse::uint)
                        .left(" clay. "),
                )
                .pair(
                    "Each geode robot costs "
                        .right(parse::uint)
                        .left(" ore and ")
                        .pair(parse::uint)
                        .left(" obsidian."),
                )
                .map(|(((ore, clay), obsidian), geode)| BlueprintCosts {
                    ore,
                    clay,
                    obsidian,
                    geode,
                })
                .parse_exact(input)
                .unwrap()
        }

        let blueprints = input.iter().map(parse).collect::<Vec<_>>();

        Self::part1("")
    }

    fn day20(input: Vec<String>) -> Answers {
        #[derive(Debug, Clone, Copy)]
        struct ListNode {
            num: isize,
            next: usize,
            prev: usize,
        }

        fn remove(nodes: &mut [ListNode], i: usize) {
            let node = nodes[i];
            nodes[node.next].prev = node.prev;
            nodes[node.prev].next = node.next;
        }

        fn insert_before(nodes: &mut [ListNode], dst: usize, src: usize) {
            let target = nodes[dst];
            nodes[dst].prev = src;
            nodes[target.prev].next = src;
            nodes[src].prev = target.prev;
            nodes[src].next = dst;
        }

        fn nth_after(nodes: &[ListNode], n: usize, mut i: usize) -> usize {
            for _ in 0..n {
                i = nodes[i].next;
            }
            i
        }

        fn decrypt(mut nodes: Vec<ListNode>, key: isize, rounds: usize) -> isize {
            for node in nodes.iter_mut() {
                node.num *= key;
            }

            let n = nodes.len();
            for _ in 0..rounds {
                for i in 0..n {
                    if nodes[i].num != 0 {
                        let num = nodes[i].num;
                        let delta = (num.rem_euclid(n as isize - 1) + 1) as usize;
                        let dst = nth_after(&nodes, delta, i);
                        remove(&mut nodes, i);
                        insert_before(&mut nodes, dst, i);
                    }
                }
            }

            let i = nodes.iter().position(|node| node.num == 0).unwrap();
            let a = nth_after(&nodes, 1000 % n, i);
            let b = nth_after(&nodes, 1000 % n, a);
            let c = nth_after(&nodes, 1000 % n, b);

            nodes[a].num + nodes[b].num + nodes[c].num
        }

        fn _print_inorder(nodes: &[ListNode]) {
            let mut i = 0;
            for _ in 0..nodes.len() {
                print!("{} ", nodes[i].num);
                i = nodes[i].next;
            }
            println!();
        }

        let n = input.len();
        let nodes: Vec<ListNode> = input
            .iter()
            .enumerate()
            .map(|(i, s)| ListNode {
                num: s.parse::<isize>().unwrap(),
                next: (i + 1) % n,
                prev: (i + n - 1) % n,
            })
            .collect();

        let part1 = decrypt(nodes.clone(), 1, 1);
        let part2 = decrypt(nodes, 811589153, 10);
        Self::solutions(part1, part2)
    }

    fn day21(input: Vec<String>) -> Answers {
        enum Monkey<'a> {
            Lit(usize),
            Operation([&'a str; 3]),
        }

        #[derive(Debug)]
        enum Expr {
            Var,
            Lit(usize),
            Operation {
                lhs: Box<Expr>,
                op: char,
                rhs: Box<Expr>,
            },
        }

        fn eval<'a>(monkeys: &HashMap<&'a str, Monkey<'a>>, target: &'a str) -> usize {
            match monkeys[target] {
                Monkey::Lit(x) => x,
                Monkey::Operation([a, "+", b]) => eval(monkeys, a) + eval(monkeys, b),
                Monkey::Operation([a, "-", b]) => eval(monkeys, a) - eval(monkeys, b),
                Monkey::Operation([a, "*", b]) => eval(monkeys, a) * eval(monkeys, b),
                Monkey::Operation([a, "/", b]) => eval(monkeys, a) / eval(monkeys, b),
                _ => unreachable!(),
            }
        }

        fn to_equation<'a>(monkeys: &HashMap<&'a str, Monkey<'a>>, target: &'a str) -> Expr {
            if target == "humn" {
                return Expr::Var;
            }
            match monkeys[target] {
                Monkey::Lit(x) => Expr::Lit(x),
                Monkey::Operation([a, op, b]) => Expr::Operation {
                    lhs: Box::new(to_equation(monkeys, a)),
                    op: op.chars().next().unwrap(),
                    rhs: Box::new(to_equation(monkeys, b)),
                },
            }
        }

        fn reduce(eqn: Expr) -> Expr {
            match eqn {
                Expr::Operation { lhs, op, rhs } => match (reduce(*lhs), reduce(*rhs)) {
                    (Expr::Lit(a), Expr::Lit(b)) => Expr::Lit(match op {
                        '+' => a + b,
                        '-' => a - b,
                        '*' => a * b,
                        '/' => a / b,
                        _ => unreachable!(),
                    }),
                    (a, b) => Expr::Operation {
                        lhs: Box::new(a),
                        op,
                        rhs: Box::new(b),
                    },
                },
                x => x,
            }
        }

        // Moves all operations to the right hand side,
        // leaving an eqn of the form x = <ret>
        fn rebalance_rhs(mut lhs: Expr, mut rhs: Expr) -> Expr {
            while let Expr::Operation { lhs: a, op, rhs: b } = lhs {
                match (*a, op, *b) {
                    (other, '+', lit @ Expr::Lit(_)) | (lit @ Expr::Lit(_), '+', other) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(rhs),
                            op: '-',
                            rhs: Box::new(lit),
                        }
                    }
                    (other, '-', lit @ Expr::Lit(_)) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(rhs),
                            op: '+',
                            rhs: Box::new(lit),
                        }
                    }
                    // a - b = c : b = a - c
                    (lit @ Expr::Lit(_), '-', other) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(lit),
                            op: '-',
                            rhs: Box::new(rhs),
                        }
                    }
                    (other, '*', lit @ Expr::Lit(_)) | (lit @ Expr::Lit(_), '*', other) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(rhs),
                            op: '/',
                            rhs: Box::new(lit),
                        }
                    }
                    (other, '/', lit @ Expr::Lit(_)) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(rhs),
                            op: '*',
                            rhs: Box::new(lit),
                        }
                    }
                    // a / b = c : c * b = a : b = a / c
                    (lit @ Expr::Lit(_), '/', other) => {
                        lhs = other;
                        rhs = Expr::Operation {
                            lhs: Box::new(lit),
                            op: '/',
                            rhs: Box::new(rhs),
                        }
                    }
                    (a, op, b) => unreachable!("Non-reducible equation: {:#?} {} {:#?}", a, op, b),
                }
            }
            rhs
        }

        let monkeys = input
            .iter()
            .map(|line| {
                let (name, rest) = parse::predicate(|c| c != ':')
                    .left(": ")
                    .pars(line)
                    .unwrap();
                let monkey = if rest.chars().all(char::is_numeric) {
                    Monkey::Lit(rest.parse().unwrap())
                } else {
                    let mut words = rest.split_whitespace();
                    Monkey::Operation([
                        words.next().unwrap(),
                        words.next().unwrap(),
                        words.next().unwrap(),
                    ])
                };
                (name, monkey)
            })
            .collect::<HashMap<_, _>>();

        let part1 = eval(&monkeys, "root");

        let Expr::Operation{ lhs, rhs, .. } = to_equation(&monkeys, "root") else {
            panic!("Root operation not binary");
        };
        let lhs = reduce(*lhs);
        let rhs = rebalance_rhs(lhs, *rhs);
        let Expr::Lit(part2) = reduce(rhs) else {
            panic!("Right hand side could not be reduced");
        };
        Self::solutions(part1, part2)
    }

    fn day22(input: Vec<String>) -> Answers {
        fn wrapping_add(x: usize, y: isize, n: usize) -> usize {
            (x as isize + y).rem_euclid(n as isize) as usize
        }

        fn wrapping_next_space(
            map: &[Vec<char>],
            (mut y, mut x): (usize, usize),
            facing: usize,
        ) -> (usize, usize) {
            match facing {
                // Horizontal movement
                0 | 2 => {
                    let dir = if facing == 0 { 1 } else { -1 };
                    loop {
                        x = wrapping_add(x, dir, map[y].len());
                        if map[y][x] != ' ' {
                            break;
                        }
                    }
                }
                // Vertical movement
                1 | 3 => {
                    let dir = if facing == 1 { 1 } else { -1 };
                    loop {
                        y = wrapping_add(y, dir, map.len());
                        match map[y].get(x) {
                            None | Some(' ') => (),
                            _ => break,
                        }
                    }
                }
                _ => unreachable!(),
            }
            (y, x)
        }

        struct Instruction {
            forward: usize,
            turn: Option<char>,
        }

        let mut parts = input.split(String::is_empty);
        let map: Vec<Vec<char>> = parts
            .next()
            .unwrap()
            .iter()
            .map(|line| line.chars().collect())
            .collect();
        let instrs = {
            let instr_line = &parts.next().unwrap()[0];
            parse::zero_or_more(
                parse::uint
                    .pair(parse::opt(parse::any_char))
                    .map(|(forward, turn)| Instruction { forward, turn }),
            )
            .parse_exact(instr_line)
            .unwrap()
        };

        let mut pos = (0, map[0].iter().position(|&c| c == '.').unwrap());
        let mut facing = 0;

        for Instruction { forward, turn } in instrs {
            for _ in 0..forward {
                let next = wrapping_next_space(&map, pos, facing);
                if map[next.0][next.1] == '#' {
                    break;
                }
                pos = next;
            }
            facing = match turn {
                Some('L') => wrapping_add(facing, -1, 4),
                Some('R') => wrapping_add(facing, 1, 4),
                _ => facing,
            }
        }

        let (row, column) = (pos.0 + 1, pos.1 + 1);
        let password = row * 1000 + column * 4 + facing;

        Self::part1(password)
    }

    fn day23(input: Vec<String>) -> Answers {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Direction {
            North,
            South,
            West,
            East,
            None,
        }

        fn neighbors((x, y): (isize, isize)) -> impl Iterator<Item = (isize, isize)> {
            let xrange = x - 1..=x + 1;
            xrange
                .flat_map(move |x| (y - 1..=y + 1).map(move |y| (x, y)))
                .filter(move |pos| *pos != (x, y))
        }

        fn neighbors_in((x, y): (isize, isize), dir: &Direction) -> [(isize, isize); 3] {
            match dir {
                Direction::North => [x - 1, x, x + 1].map(|x| (x, y - 1)),
                Direction::South => [x - 1, x, x + 1].map(|x| (x, y + 1)),
                Direction::West => [y - 1, y, y + 1].map(|y| (x - 1, y)),
                Direction::East => [y - 1, y, y + 1].map(|y| (x + 1, y)),
                _ => panic!(),
            }
        }

        fn target_position((x, y): (isize, isize), dir: &Direction) -> (isize, isize) {
            match dir {
                Direction::North => (x, y - 1),
                Direction::South => (x, y + 1),
                Direction::East => (x + 1, y),
                Direction::West => (x - 1, y),
                Direction::None => (x, y),
            }
        }

        fn count_occurrences(it: &[(isize, isize)]) -> HashMap<(isize, isize), usize> {
            let mut map = HashMap::new();
            for pos in it.iter().copied() {
                *map.entry(pos).or_insert(0) += 1;
            }
            map
        }

        fn simulate_round(elves: &mut [(isize, isize)], directions: &mut [Direction]) -> bool {
            let moves = elves.iter().map(|elf| {
                if neighbors(*elf).all(|neighbor| !elves.contains(&neighbor)) {
                    Direction::None
                } else {
                    directions
                        .iter()
                        .copied()
                        .find(|dir| {
                            neighbors_in(*elf, dir)
                                .iter()
                                .all(|neighbor| !elves.contains(&neighbor))
                        })
                        .unwrap_or(Direction::None)
                }
            });

            if moves.clone().all(|dir| dir == Direction::None) {
                return false;
            }

            let target_positions = elves
                .iter()
                .zip(moves)
                .map(|(elf, dir)| target_position(*elf, &dir))
                .collect::<Vec<_>>();

            let occurrences = count_occurrences(&target_positions);
            for (elf, target) in elves.iter_mut().zip(target_positions) {
                if occurrences[&target] == 1 {
                    *elf = target;
                }
            }

            directions.rotate_left(1);
            true
        }

        let mut elves = input
            .iter()
            .zip(0isize..)
            .flat_map(|(line, y)| line.chars().zip(0isize..).map(move |(c, x)| (x, y, c)))
            .filter(|(_, _, c)| *c == '#')
            .map(|(x, y, _)| (x, y))
            .collect::<Vec<_>>();

        let mut directions = vec![
            Direction::North,
            Direction::South,
            Direction::West,
            Direction::East,
        ];

        for _round in 0..10 {
            simulate_round(&mut elves, &mut directions);
        }

        let (xmin, xmax) = (
            elves.iter().map(|pos| pos.0).min().unwrap(),
            elves.iter().map(|pos| pos.0).max().unwrap() + 1,
        );
        let (ymin, ymax) = (
            elves.iter().map(|pos| pos.1).min().unwrap(),
            elves.iter().map(|pos| pos.1).max().unwrap() + 1,
        );
        let rect_area = ((ymax - ymin) * (xmax - xmin)) as usize;
        let part1 = rect_area - elves.len();

        let mut round = 11;
        while simulate_round(&mut elves, &mut directions) {
            round += 1;
        }

        Self::solutions(part1, round)
    }
}
