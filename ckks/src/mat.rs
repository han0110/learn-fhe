#[derive(Clone, Debug)]
pub struct Matrix<T> {
    height: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(width: usize, height: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            height,
            data: vec![T::default(); width * height],
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.data.len() / self.height
    }

    pub fn cols_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.height)
    }

    pub fn cols(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.height)
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.height()).map(|idx| self.data[idx..].iter().step_by(self.height()))
    }
}
