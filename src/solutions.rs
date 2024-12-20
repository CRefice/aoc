pub type Answers = (String, Option<String>);

pub trait Solutions {
    fn run(day: usize, input: Vec<String>) -> Answers {
        let days = [
            Self::day1,
            Self::day2,
            Self::day3,
            Self::day4,
            Self::day5,
            Self::day6,
            Self::day7,
            Self::day8,
            Self::day9,
            Self::day10,
            Self::day11,
            Self::day12,
            Self::day13,
            Self::day14,
            Self::day15,
            Self::day16,
            Self::day17,
            Self::day18,
            Self::day19,
            Self::day20,
            Self::day21,
            Self::day22,
            Self::day23,
            Self::day24,
            Self::day25,
        ];
        days.get(day - 1).expect("No problem available for day")(input)
    }

    fn part1<T: ToString>(solution: T) -> Answers {
        (solution.to_string(), None)
    }

    fn solutions<T: ToString, U: ToString>(part1: T, part2: U) -> Answers {
        (part1.to_string(), Some(part2.to_string()))
    }

    fn day1(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day2(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day3(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day4(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day5(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day6(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day7(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day8(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day9(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day10(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day11(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day12(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day13(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day14(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day15(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day16(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day17(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day18(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day19(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day20(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day21(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day22(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day23(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day24(_input: Vec<String>) -> Answers {
        todo!()
    }
    fn day25(_input: Vec<String>) -> Answers {
        todo!()
    }
}
