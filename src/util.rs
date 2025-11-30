use std::fmt::{Debug, Display};

#[derive(PartialEq, Eq, Clone, Hash)]
pub struct Map(Vec<Vec<char>>);

impl Map {
    pub fn from<T: AsRef<str>>(lines: &[T]) -> Self {
        Map(lines.iter().map(|s| s.as_ref().chars().collect()).collect())
    }

    pub fn rows(
        &self,
    ) -> impl std::iter::DoubleEndedIterator<Item = &[char]> + std::iter::ExactSizeIterator {
        self.0.iter().map(|row| row.as_slice())
    }

    pub fn rows_mut(
        &mut self,
    ) -> impl std::iter::DoubleEndedIterator<Item = &mut [char]> + std::iter::ExactSizeIterator
    {
        self.0.iter_mut().map(|row| row.as_mut_slice())
    }

    pub fn columns(
        &self,
    ) -> impl std::iter::DoubleEndedIterator<Item = impl Iterator<Item = &char>>
           + std::iter::ExactSizeIterator {
        let width = self.width();
        let height = self.height();
        (0..width).map(move |col| (0..height).map(move |row| &self.0[row][col]))
    }

    pub fn width(&self) -> usize {
        self.0.first().map(|row| row.len()).unwrap_or(0)
    }

    pub fn height(&self) -> usize {
        self.0.len()
    }

    pub fn transposed(&self) -> Self {
        Map((0..self.width())
            .map(|row| (0..self.height()).map(|col| self[(col, row)]).collect())
            .collect())
    }

    pub const UP: (isize, isize) = (-1, 0);
    pub const DOWN: (isize, isize) = (1, 0);
    pub const LEFT: (isize, isize) = (0, -1);
    pub const RIGHT: (isize, isize) = (0, 1);

    pub fn step(
        &self,
        (row, col): (usize, usize),
        (y, x): (isize, isize),
    ) -> Option<(usize, usize)> {
        let row = row.checked_add_signed(y).filter(|&row| row < self.height());
        let col = col.checked_add_signed(x).filter(|&col| col < self.width());
        row.and_then(|row| col.map(move |col| (row, col)))
    }

    pub fn step_if(
        &self,
        pos: (usize, usize),
        dir: (isize, isize),
        pred: impl FnOnce(char) -> bool,
    ) -> Option<(usize, usize)> {
        self.step(pos, dir).filter(|&p| pred(self[p]))
    }

    pub fn step_wrapping(
        &self,
        (row, col): (usize, usize),
        (y, x): (isize, isize),
    ) -> (usize, usize) {
        let row = (row + self.height()).checked_add_signed(y).unwrap() % self.height();
        let col = (col + self.width()).checked_add_signed(x).unwrap() % self.width();
        (row, col)
    }

    pub fn find(&self, target: char) -> Option<(usize, usize)> {
        self.rows()
            .enumerate()
            .find_map(|(i, row)| row.iter().position(|&c| c == target).map(|j| (i, j)))
    }
}

impl Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sep = "";
        for row in self.rows() {
            write!(f, "{}", std::mem::replace(&mut sep, "\n"))?;
            for c in row {
                write!(f, "{}", c)?;
            }
        }
        Ok(())
    }
}

impl Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Map")
            .field(&format_args!("{}", &self))
            .finish()
    }
}

impl std::ops::Index<(usize, usize)> for Map {
    type Output = char;

    fn index(&self, (row, col): (usize, usize)) -> &char {
        &self.0[row][col]
    }
}

impl std::ops::Index<usize> for Map {
    type Output = [char];

    fn index(&self, row: usize) -> &[char] {
        &self.0[row]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Map {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut char {
        &mut self.0[row][col]
    }
}

impl std::ops::IndexMut<usize> for Map {
    fn index_mut(&mut self, row: usize) -> &mut [char] {
        &mut self.0[row]
    }
}

pub fn gcd<T>(mut a: T, mut b: T) -> T
where
    T: PartialOrd + From<u8> + Copy + std::ops::Rem<Output = T>,
{
    while b > T::from(0u8) {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub fn lcm<T>(a: T, b: T) -> T
where
    T: PartialOrd
        + From<u8>
        + Copy
        + std::ops::Rem<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    a * b / gcd(a, b)
}

pub fn num_digits<T>(mut a: T) -> u32
where
    T: PartialOrd + From<u8> + Copy + std::ops::Div<Output = T>,
{
    let mut result = 1;
    while a > 9.into() {
        result += 1;
        a = a / 10.into();
    }
    result
}

use std::collections::HashMap;
use std::hash::Hash;
use std::mem::MaybeUninit;

type Idx = usize;
struct Node<V> {
    value: MaybeUninit<V>,
    prev: Option<Idx>,
    next: Option<Idx>,
}

impl<V: Copy> Clone for Node<V> {
    fn clone(&self) -> Self {
        Node {
            value: self.value,
            prev: self.prev,
            next: self.next,
        }
    }
}

/// A HashMap that preserves order of insertion.
///
/// # Examples
///
/// ```
/// use aoc::util::OrderedMap;
///
/// let mut map = OrderedMap::new();
/// map.insert("one", 1);
/// map.insert("two", 2);
/// map.insert("three", 3);
/// map.insert("four", 4);
///
/// assert_eq!(map.get("two"), Some(&2));
/// assert_eq!(map.remove("three"), Some(3));
/// assert_eq!(map.remove("five"), None);
///
/// let mut it = map.into_iter();
/// assert_eq!(it.next(), Some(1));
/// assert_eq!(it.next(), Some(2));
/// assert_eq!(it.next(), Some(4));
/// assert_eq!(it.next(), None);
/// ```
pub struct OrderedMap<K, V> {
    map: HashMap<K, usize>,
    nodes: Vec<Node<V>>,
    free: Vec<usize>,
    first: Option<usize>,
    last: Option<usize>,
}

impl<K, V> Default for OrderedMap<K, V>
where
    K: Hash + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> OrderedMap<K, V>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            first: None,
            last: None,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(idx) = self.map.get(&key).copied() {
            unsafe {
                return Some(std::mem::replace(
                    self.nodes[idx].value.assume_init_mut(),
                    value,
                ));
            }
        }

        let prev = self.last;
        let next = None;
        let idx = if let Some(idx) = self.free.pop() {
            self.nodes[idx].prev = prev;
            self.nodes[idx].next = next;
            idx
        } else {
            let idx = self.nodes.len();
            let node = Node {
                value: MaybeUninit::uninit(),
                prev,
                next,
            };
            self.nodes.push(node);
            idx
        };
        self.nodes[idx].value.write(value);
        self.map.insert(key, idx);

        if let Some(last) = self.last {
            self.nodes[last].next = Some(idx);
        }
        self.first.get_or_insert(idx);
        self.last = Some(idx);
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .get(key)
            .copied()
            .map(|idx| unsafe { self.nodes[idx].value.assume_init_ref() })
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .get(key)
            .copied()
            .map(|idx| unsafe { self.nodes[idx].value.assume_init_mut() })
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.map.remove(key)?;
        let Node { prev, next, .. } = self.nodes[idx];
        if let Some(prev) = prev {
            self.nodes[prev].next = next;
        }
        if let Some(next) = next {
            self.nodes[next].prev = prev;
        }

        if self.first == Some(idx) {
            self.first = next;
        }
        if self.last == Some(idx) {
            self.last = prev;
        }

        self.free.push(idx);
        Some(unsafe { self.nodes[idx].value.assume_init_read() })
    }
}

pub struct OrderedMapIter<V> {
    cur: Option<usize>,
    nodes: Vec<Node<V>>,
}

impl<V> Iterator for OrderedMapIter<V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.cur {
            let val = unsafe { self.nodes[idx].value.assume_init_read() };
            self.cur = self.nodes[idx].next;
            Some(val)
        } else {
            None
        }
    }
}

impl<K: Hash + Eq, V> IntoIterator for OrderedMap<K, V> {
    type Item = V;

    type IntoIter = OrderedMapIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        OrderedMapIter {
            cur: self.first,
            nodes: self.nodes,
        }
    }
}

impl<K, V> Clone for OrderedMap<K, V>
where
    K: Clone,
    V: Copy,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            nodes: self.nodes.clone(),
            free: self.free.clone(),
            first: self.first,
            last: self.last,
        }
    }
}
