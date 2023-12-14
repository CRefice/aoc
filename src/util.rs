#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Map(Vec<Vec<char>>);

impl Map {
    pub fn from<T: AsRef<str>>(lines: &[T]) -> Self {
        Map(lines.iter().map(|s| s.as_ref().chars().collect()).collect())
    }

    pub fn rows(
        &self,
    ) -> impl Iterator<Item = &[char]> + std::iter::DoubleEndedIterator + std::iter::ExactSizeIterator
    {
        self.0.iter().map(|row| row.as_slice())
    }

    pub fn rows_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut [char]> + std::iter::DoubleEndedIterator + std::iter::ExactSizeIterator
    {
        self.0.iter_mut().map(|row| row.as_mut_slice())
    }

    pub fn columns(
        &self,
    ) -> impl Iterator<Item = impl Iterator<Item = &char>>
           + std::iter::DoubleEndedIterator
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
