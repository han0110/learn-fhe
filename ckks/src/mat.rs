#[derive(Clone, Debug)]
pub struct Matrix<T> {
    width: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(width: usize, height: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            width,
            data: vec![T::default(); width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.data.len() / self.width
    }
}
