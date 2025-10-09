use std::collections::HashMap;

use crate::{
    constants::{HID_CHS, INP_DIM, OUT_CHS, VIS_CHS},
    dataset::Dataset,
    grid::Grid,
    metrics::TaskReport,
    nca::NCA,
    substrate::Substrate,
};
use itertools::Itertools;
use macroquad::prelude::*;

pub fn display_visible_grid(grid: &Grid, x: f32, y: f32, w: f32, h: f32) {
    let height = grid.height();
    let width = grid.width();
    let sq_size = (h / height as f32).min(w / width as f32);
    let w = sq_size * width as f32;
    let h = sq_size * height as f32;
    draw_rectangle_lines(x, y, w, h, 1.0, WHITE.with_alpha(0.5));

    for yi in 0..height {
        for xi in 0..width {
            let cell_x = x + xi as f32 * sq_size;
            let cell_y = y + yi as f32 * sq_size;

            let value = grid[(yi, xi)];
            let color = COLOR_MAP[value as usize];

            // Fill cell
            draw_rectangle(cell_x, cell_y, sq_size, sq_size, color);

            // Cell border
            draw_rectangle_lines(cell_x, cell_y, sq_size, sq_size, 0.5, WHITE.with_alpha(0.25));
        }
    }
}

impl Substrate {
    pub fn display_channel_grid(&self, x: f32, y: f32, w: f32, h: f32, channel: usize) {
        let sq_size = (h / self.height as f32).min(w / self.width as f32);
        let w = sq_size * self.width as f32;
        let h = sq_size * self.height as f32;
        draw_rectangle_lines(x, y, w, h, 1.0, WHITE.with_alpha(0.5));

        for yi in 0..self.height {
            for xi in 0..self.width {
                let cell_x = x + xi as f32 * sq_size;
                let cell_y = y + yi as f32 * sq_size;

                let val = self.data[(yi, xi, channel)].clamp(0.0, 1.0);
                let color = GRAY.with_alpha(val.abs());

                // Fill cell
                draw_rectangle(cell_x, cell_y, sq_size, sq_size, color);

                // Cell border
                draw_rectangle_lines(cell_x, cell_y, sq_size, sq_size, 0.5, WHITE.with_alpha(0.25));
            }
        }

        let label = format!("ch {channel}");
        let measure = measure_text(&label, None, 24.0 as u16, 1.0);
        // Label
        draw_text(&label, x, y + h + measure.height * 1.5, 24.0, WHITE);
    }

    pub fn display_channels_panel(&self, start_x: f32, start_y: f32, w: f32, h: f32) {
        // Show only non-frozen channels: RW visible + hidden
        let tot_rw_chs = VIS_CHS + HID_CHS;
        let start = VIS_CHS;

        if tot_rw_chs == 0 || w <= 0.0 || h <= 0.0 {
            return;
        }

        // Approx square grid layout
        let cols = (tot_rw_chs as f32).sqrt().ceil() as usize;
        let cols = cols.max(1);
        let rows = tot_rw_chs.div_ceil(cols);

        let pad = (w.min(h) * 0.02).max(2.0);
        let cell_w = (w - pad * (cols.saturating_sub(1) as f32)) / cols as f32;
        let cell_h_total = (h - pad * (rows.saturating_sub(1) as f32)) / rows as f32;

        let font_size = (cell_h_total * 0.8).clamp(10.0, 24.0);
        let grid_h = (cell_h_total - font_size).max(1.0);
        let grid_w = cell_w;

        for i in 0..tot_rw_chs {
            let row = i / cols;
            let col = i % cols;

            let cx = start_x + (col as f32) * (cell_w + pad);
            let cy = start_y + (row as f32) * (cell_h_total + pad);

            // Draw the channel grid inside its cell area (excluding label slice)
            self.display_channel_grid(cx, cy, grid_w, grid_h, start + i);
        }
    }
}

pub fn draw_params(x: f32, y: f32, w: f32, h: f32, nca_id: usize, nca: &mut NCA) {
    draw_rectangle_lines(x, y, w, h, 1.0, WHITE.with_alpha(0.5));

    let shape = (OUT_CHS, INP_DIM);
    draw_text(
        &format!(
            "nca_id={}, weights={}, biases={}, shape={:?}",
            nca_id,
            nca.weights.len(),
            nca.biases.len(),
            shape
        ),
        x,
        y + h + 20.0,
        24.0,
        WHITE,
    );

    let w_max = nca.weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let b_max = nca.biases.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let param_max = w_max.max(b_max).max(1e-5);

    let n_rows = OUT_CHS;
    let n_cols = INP_DIM;
    let p_h = if n_rows > 0 { h / n_rows as f32 } else { h };
    let p_w = if n_cols > 0 { w / n_cols as f32 } else { w };

    for yi in 0..n_rows {
        for xi in 0..n_cols {
            let idx = yi * INP_DIM + xi;
            if idx >= nca.weights.len() {
                continue;
            }
            let val = nca.weights[idx] / param_max;
            draw_rectangle(
                x + xi as f32 * p_w,
                y + yi as f32 * p_h,
                p_w,
                p_h,
                if val > 0.0 {
                    BLUE.with_alpha(val.abs())
                } else {
                    RED.with_alpha(val.abs())
                },
            );
        }
    }
}

pub fn draw_tooltip(x: f32, y: f32, lines: &[&str]) {
    let pad = 8.0;
    let font_size = 18.0;
    let line_h = font_size * 1.25;

    // Compute bounding box
    let mut max_w = 0.0_f32;
    for l in lines {
        let m = measure_text(l, None, font_size as u16, 1.0);
        if m.width > max_w {
            max_w = m.width;
        }
    }
    let w = max_w + 2.0 * pad;
    let h = lines.len() as f32 * line_h + 2.0 * pad;

    // Keep on-screen
    let sw = screen_width();
    let sh = screen_height();
    let mut bx = x;
    let mut by = y;
    if bx + w > sw {
        bx = (sw - w - 2.0).max(0.0);
    }
    if by + h > sh {
        by = (sh - h - 2.0).max(0.0);
    }

    // Draw
    draw_rectangle(bx, by, w, h, BLACK.with_alpha(0.85));
    draw_rectangle_lines(bx, by, w, h, 1.0, WHITE.with_alpha(0.85));

    // Text
    let mut ty = by + pad + font_size;
    for l in lines {
        draw_text(l, bx + pad, ty, font_size, WHITE);
        ty += line_h;
    }
}

pub fn draw_metrics(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    dataset: &Dataset,
    metrics: &[(String, TaskReport)],
    selected_task_id: Option<&str>,
) -> (Option<String>, Option<(f32, f32, Vec<String>)>) {
    let metrics = metrics.iter().map(|i| (&i.0, &i.1)).collect_vec();
    let metrics_map: HashMap<&String, &TaskReport> = HashMap::from_iter(metrics);
    let n_tasks = dataset.tasks.len();

    let n_rows = 10;
    let n_col = n_tasks.div_ceil(n_rows);

    let bb_h = h / n_rows as f32;
    let bb_w = w / n_col as f32;
    let pad = 2.0;
    let c_r = (bb_h.min(bb_w) / 2.0 - pad).max(4.0);
    let offset_x = bb_w / 2.0;
    let offset_y = bb_h / 2.0;

    draw_rectangle_lines(x, y, w, h, 1.0, WHITE.with_alpha(0.5));

    let (mx, my) = mouse_position();
    let mut clicked: Option<String> = None;
    let mut tooltip: Option<(f32, f32, Vec<String>)> = None;

    for yi in 0..n_rows {
        for xi in 0..n_col {
            let task_idx = yi * n_col + xi;
            if task_idx >= n_tasks {
                continue;
            }

            let task = &dataset.tasks[task_idx];
            let cx = x + xi as f32 * bb_w + offset_x;
            let cy = y + yi as f32 * bb_h + offset_y;

            if let Some(metric) = metrics_map.get(&task.id) {
                let mean_test_acc = metric.test_accs.iter().sum::<f32>() / metric.test_accs.len() as f32;
                let mean_train_acc = metric.train_accs.iter().sum::<f32>() / metric.train_accs.len() as f32;

                let test_color = if mean_test_acc == 1.0 {
                    GREEN
                } else {
                    BLUE.with_alpha(mean_test_acc)
                };
                let train_color = if mean_train_acc == 1.0 {
                    GREEN
                } else {
                    BLUE.with_alpha(mean_train_acc)
                };

                // Filled test-acc circle
                draw_circle(cx, cy, c_r, test_color);
                // Train-acc ring
                draw_circle_lines(cx, cy, c_r - 2.0, 2.0, train_color);

                // Highlight if selected
                if let Some(sel_id) = selected_task_id
                    && sel_id == task.id
                {
                    draw_circle_lines(cx, cy, c_r + 1.0, 2.0, YELLOW.with_alpha(0.8));
                }

                // Hover detection
                let dx = mx - cx;
                let dy = my - cy;
                if dx * dx + dy * dy <= c_r * c_r {
                    // Tooltip with all accuracies
                    let train_str = metric.train_accs.iter().map(|v| format!("{:.2}", v)).join(", ");
                    let test_str = metric.test_accs.iter().map(|v| format!("{:.2}", v)).join(", ");
                    let lines_owned = vec![
                        format!("Task: {}", task.id),
                        format!("Train accs: [{}]", train_str),
                        format!("Test accs:  [{}]", test_str),
                    ];
                    tooltip = Some((mx + 12.0, my + 12.0, lines_owned));

                    // Click to select
                    if is_mouse_button_pressed(MouseButton::Left) {
                        clicked = Some(task.id.clone());
                    }
                }
            } else {
                // No metrics available: draw placeholder
                draw_circle(cx, cy, c_r, GRAY.with_alpha(0.1));
                draw_circle_lines(cx, cy, c_r - 2.0, 2.0, GRAY.with_alpha(0.1));

                if let Some(sel_id) = selected_task_id
                    && sel_id == task.id
                {
                    draw_circle_lines(cx, cy, c_r + 3.0, 3.0, YELLOW);
                }

                // Hover detection for placeholder
                let dx = mx - cx;
                let dy = my - cy;
                if dx * dx + dy * dy <= c_r * c_r {
                    let lines_owned = vec![format!("Task: {}", task.id), "No metrics available".to_string()];
                    tooltip = Some((mx + 12.0, my + 12.0, lines_owned));

                    if is_mouse_button_pressed(MouseButton::Left) {
                        clicked = Some(task.id.clone());
                    }
                }
            }
        }
    }

    (clicked, tooltip)
}

/// ARC colors
pub const COLOR_MAP: &[Color] = &[
    Color::from_hex(0x000000), // 0: black #000000
    Color::from_hex(0x0074D9), // 1: blue #0074D9
    Color::from_hex(0xFF4136), // 2: red #FF4136
    Color::from_hex(0x2ECC40), // 3: green #2ECC40
    Color::from_hex(0xFFDC00), // 4: yellow #FFDC00
    Color::from_hex(0xAAAAAA), // 5: grey #AAAAAA
    Color::from_hex(0xF012BE), // 6: fuchsia #F012BE
    Color::from_hex(0xFF851B), // 7: orange #FF851B
    Color::from_hex(0x7FDBFF), // 8: teal #7FDBFF
    Color::from_hex(0x870C25), // 9: brown #870C25
];
