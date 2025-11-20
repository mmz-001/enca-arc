use crate::metrics::TrainMetrics;
use plotters::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs,
};

pub fn plot_metrics(metrics: &TrainMetrics, out_dir: &str, task_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let task_plot_dir = format!("{}/plots/{}", out_dir, task_id);
    fs::create_dir_all(&task_plot_dir)
        .unwrap_or_else(|e| panic!("Failed to create plot directory '{}': {}", task_plot_dir, e));

    let fitness_path = format!("{}/plots/{}/fitness.png", out_dir, task_id);
    let accuracy_path = format!("{}/plots/{}/accuracy.png", out_dir, task_id);

    let mut fitness_data: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    let mut accuracy_data: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    let mut all_ids = HashSet::new();
    let mut max_epoch = 0;
    let mut min_fitness = f32::MAX;
    let mut max_fitness = f32::MIN;
    let mut min_acc = f32::MAX;
    let mut max_acc = f32::MIN;

    // Extract data from metrics
    for epoch_metrics in &metrics.epoch_metrics {
        if epoch_metrics.epoch > max_epoch {
            max_epoch = epoch_metrics.epoch;
        }
        for ind in &epoch_metrics.individual_metrics {
            all_ids.insert(ind.id);

            fitness_data
                .entry(ind.id)
                .or_default()
                .push((epoch_metrics.epoch, ind.fitness));

            accuracy_data
                .entry(ind.id)
                .or_default()
                .push((epoch_metrics.epoch, ind.mean_acc));

            if ind.fitness < min_fitness {
                min_fitness = ind.fitness;
            }
            if ind.fitness > max_fitness {
                max_fitness = ind.fitness;
            }
            if ind.mean_acc < min_acc {
                min_acc = ind.mean_acc;
            }
            if ind.mean_acc > max_acc {
                max_acc = ind.mean_acc;
            }
        }
    }

    if all_ids.is_empty() {
        return Ok(());
    }

    let mut sorted_ids: Vec<usize> = all_ids.into_iter().collect();
    sorted_ids.sort();

    // Generate unique colors based on ID
    let get_color = |id: usize| -> RGBAColor {
        let idx = sorted_ids.iter().position(|&x| x == id).unwrap_or(0);
        // Use golden angle approximation to distribute colors evenly
        let hue = (idx as f64 * 0.618033988749895) % 1.0;
        HSLColor(hue, 0.7, 0.5).to_rgba()
    };

    // Ensure max_epoch is at least 1 for plot range
    let x_max = max_epoch.max(1);

    // Plot Fitness
    let root = BitMapBackend::new(&fitness_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Add some margin to the Y-axis range
    let y_margin = (max_fitness - min_fitness).abs() * 0.05;
    let y_range = (min_fitness - y_margin)..(max_fitness + y_margin);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Task {} Fitness", task_id), ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..x_max, y_range)?;

    chart.configure_mesh().x_desc("Epoch").y_desc("Fitness").draw()?;

    for id in &sorted_ids {
        if let Some(series) = fitness_data.get(id) {
            let color = get_color(*id);
            chart
                .draw_series(LineSeries::new(series.clone(), &color))?
                .label(format!("ID {}", id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Plot Mean Accuracy
    let root = BitMapBackend::new(&accuracy_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Add some margin to the Y-axis range
    let y_margin = (max_acc - min_acc).abs() * 0.05;
    let y_range = (min_acc - y_margin)..(max_acc + y_margin);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Task {} Mean Accuracy", task_id), ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..x_max, y_range)?;

    chart.configure_mesh().x_desc("Epoch").y_desc("Mean Accuracy").draw()?;

    for id in &sorted_ids {
        if let Some(series) = accuracy_data.get(id) {
            let color = get_color(*id);
            chart
                .draw_series(LineSeries::new(series.clone(), &color))?
                .label(format!("ID {}", id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
