/// Performs an uninitialized resize of the buffer to the specified length.
///
///
/// # Safety
///
/// This should only be called when the buffer will be immediately
/// rewritten and filled with new data.
#[allow(clippy::uninit_vec)]
pub(crate) fn resize<T>(buf: &mut Vec<T>, size: usize) {
    if buf.len() < size {
        buf.reserve(size - buf.len());
        unsafe {
            buf.set_len(size); // uninitialized resize
        }
    }
}
