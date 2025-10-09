/*! Property checks used for determining inference time augmentations and grid sizes. See assertions.rs
 * for more details.
 */

use crate::dataset::Task;

/// If:
/// - For every train example, the input size equals the output size.
///
/// Returns:
/// - True when the premise holds for the given grids, otherwise None.
pub fn train_preserves_grid_size(task: &Task) -> bool {
    task.train.iter().all(|s| s.input.shape() == s.output.shape())
}
