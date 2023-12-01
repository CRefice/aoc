use crate::parse::{self, Parser};
use crate::solutions::{Answers, Solutions};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

pub struct SolutionsYearHere;

impl Solutions for SolutionsYearHere {
    fn day1(input: Vec<String>) -> Answers {
        // For only answering part 1:
        // Self::part1("answer")
        Self::solutions("part1", "part2")
    }
}
